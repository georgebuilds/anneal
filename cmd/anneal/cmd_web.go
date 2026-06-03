//go:build !js

// anneal web — serve the studio (the local browser surface).
//
// Foundation (W0): static embed only, plus stub endpoints for the API surfaces
// that subsequent W steps fill in. The studio's HTML/CSS/JS/worker scaffold
// boots; the WASM build is deferred to a later W step (anneal.wasm is not
// embedded here; see TODO below).
//
// Architecture spec: notes/anneal_web_spec.md
// Visual / interaction invariants: DESIGN.md

package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"path"
	"strings"

	"github.com/georgebuilds/anneal/internal/bundle"
	"github.com/georgebuilds/anneal/viz"
	"github.com/georgebuilds/anneal/web"
)

// webCmd is the entry point wired into cmd/anneal/main.go.
func webCmd(args []string) int {
	addr := ":3001"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		addr = args[0]
	}

	mux := serveMux()
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Printf("anneal web: listen %s: %v\n", addr, err)
		return 1
	}
	url := "http://" + ln.Addr().String()
	fmt.Printf("anneal studio\n")
	fmt.Printf("listening: %s\n", url)
	fmt.Printf("open %s to load the studio.\n", url)
	fmt.Printf("(W0 foundation: routing, theme, keyboard live; views land in W1+.)\n")

	if err := http.Serve(ln, mux); err != nil {
		fmt.Printf("anneal web: serve: %v\n", err)
		return 1
	}
	return 0
}

// serveMux constructs the studio's HTTP mux. Split out so tests can spin it
// up via httptest.NewServer without binding a port.
func serveMux() *http.ServeMux {
	mux := http.NewServeMux()

	staticFS := web.FS()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			// Studio uses History API routing; every non-/static path returns
			// the shell so the client-side router can resolve it.
			if strings.HasPrefix(r.URL.Path, "/static/") ||
				strings.HasPrefix(r.URL.Path, "/visualize/") ||
				strings.HasPrefix(r.URL.Path, "/api/") ||
				strings.HasPrefix(r.URL.Path, "/sse/") {
				http.NotFound(w, r)
				return
			}
		}
		serveStudioHTML(w, r, staticFS)
	})

	mux.Handle("/static/", http.StripPrefix("/static/", staticFileServer(staticFS)))

	// W4 visualize embed: serve the viz artifact (verbatim) under
	// /visualize/static/* and the embed-wrapper HTML at /visualize/embed.
	// The standalone `anneal viz` command is unaffected; it still mounts
	// viz/static at /. Spec: notes/anneal_web_spec.md §5.2.
	mux.HandleFunc("/visualize/embed", serveVisualizeEmbed(staticFS))
	mux.HandleFunc("/visualize/api/graph",    visualizeAPIHandler())
	mux.HandleFunc("/visualize/api/timeline", visualizeAPIHandler())
	if vizFS, err := viz.StaticFS(); err == nil {
		mux.Handle("/visualize/static/", http.StripPrefix("/visualize/static/", staticFileServer(vizFS)))
	}

	mux.HandleFunc("/api/device",         stubJSON("device probe not yet implemented"))
	mux.HandleFunc("/api/runs",           runsHandler())
	mux.HandleFunc("/api/runs/",          runsHandler())
	mux.HandleFunc("/api/compile/tuned",  stubJSON("native BEAM compile not yet implemented"))
	mux.HandleFunc("/sse/train",          handleSSETrain)
	mux.HandleFunc("/sse/generate",       stubJSON("generate SSE not yet implemented"))

	return mux
}

// serveVisualizeEmbed serves the W4 iframe wrapper HTML (web/visualize_embed.html).
// The page loads the viz artifact from /visualize/static/, intercepts the
// app.js wasm fetch (which uses a relative URL) to point at the right path,
// and adds the iframe-side postMessage bridge that maps node clicks to the
// studio's drawer. The standalone `anneal viz` command never serves this.
func serveVisualizeEmbed(staticFS fs.FS) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, err := fs.ReadFile(staticFS, "visualize_embed.html")
		if err != nil {
			http.Error(w, "visualize_embed.html: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		_, _ = w.Write(b)
	}
}

// visualizeAPIHandler answers the REST fallback the embedded viz uses when
// anneal.wasm is not present. It mirrors viz.Serve's /api/graph and
// /api/timeline endpoints under the /visualize/ prefix so the studio's
// iframe and the standalone `anneal viz` command can both reach them.
func visualizeAPIHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("name")
		if name == "" {
			name = "mlp"
		}
		var b []byte
		var err error
		switch r.URL.Path {
		case "/visualize/api/graph":
			var g *viz.GraphData
			g, err = viz.BuildGraph(name)
			if err == nil {
				b, err = g.ToJSON()
			}
		case "/visualize/api/timeline":
			var t *viz.TimelineData
			t, err = viz.BuildTimeline(name)
			if err == nil {
				b, err = t.ToJSON()
			}
		default:
			http.NotFound(w, r)
			return
		}
		if err != nil {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_, _ = w.Write(b)
	}
}

// runsHandler dispatches the /api/runs[/...] family (W1). The split is:
//   - GET  /api/runs              -> list bundles
//   - POST /api/runs              -> 501 (save current run lands in W6+)
//   - GET  /api/runs/{id}         -> manifest JSON
//   - GET  /api/runs/{id}/graph.json
//   - GET  /api/runs/{id}/schedule.json
//   - GET  /api/runs/{id}/loss.csv
//   - GET  /api/runs/{id}/generation.ndjson
//   - GET  /api/runs/{id}/kernels/{name}
//
// All paths emit the {"error","detail"} JSON shape on failure. The bundle
// reader does its own path-containment check; the handler also Cleans the
// sub-path so escape attempts return 404 not panic.
func runsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Strip the /api/runs prefix; what remains is "" (list) or
		// "<id>" or "<id>/<file>" or "<id>/kernels/<name>".
		rest := strings.TrimPrefix(r.URL.Path, "/api/runs")
		rest = strings.TrimPrefix(rest, "/")

		// POST /api/runs — reserved for "save current run" in W6+.
		if rest == "" && r.Method == http.MethodPost {
			writeJSONError(w, http.StatusNotImplemented, "phase ID pending",
				"save current run is not yet wired (lands in W6+)")
			return
		}

		root, err := bundle.EnvOrDefault()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "run cache error", err.Error())
			return
		}

		// GET /api/runs — list all bundles.
		if rest == "" {
			summaries, err := bundle.ListBundles(root)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "list bundles failed", err.Error())
				return
			}
			if summaries == nil {
				summaries = []bundle.BundleSummary{}
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			_ = json.NewEncoder(w).Encode(summaries)
			return
		}

		// Split id from sub-path.
		id, sub, _ := strings.Cut(rest, "/")
		// Clean the sub-path so attempts like "graph.json/../../etc" can
		// not reach the reader with a malformed name.
		sub = path.Clean("/" + sub)
		sub = strings.TrimPrefix(sub, "/")

		rdr, err := bundle.OpenBundleIn(root, id)
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "bundle not found",
				fmt.Sprintf("id=%q: %v", id, err))
			return
		}

		switch {
		case sub == "":
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			_ = json.NewEncoder(w).Encode(rdr.Manifest())

		case sub == "graph.json":
			b, err := rdr.Graph()
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "graph not in bundle", err.Error())
				return
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			_, _ = w.Write(b)

		case sub == "schedule.json":
			b, err := rdr.Schedule()
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "schedule not in bundle", err.Error())
				return
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			_, _ = w.Write(b)

		case sub == "loss.csv":
			rows, err := rdr.Loss()
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "loss.csv not readable", err.Error())
				return
			}
			w.Header().Set("Content-Type", "text/csv; charset=utf-8")
			_, _ = fmt.Fprintln(w, bundle.LossCSVHeader)
			for _, row := range rows {
				_, _ = fmt.Fprintln(w, row.CSVRow())
			}

		case sub == "generation.ndjson":
			rows, err := rdr.Generation()
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "generation.ndjson not readable", err.Error())
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson; charset=utf-8")
			enc := json.NewEncoder(w)
			for _, row := range rows {
				_ = enc.Encode(row)
			}

		case strings.HasPrefix(sub, "kernels/"):
			name := strings.TrimPrefix(sub, "kernels/")
			wgsl, err := rdr.Kernel(name)
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "kernel not found", err.Error())
				return
			}
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			_, _ = w.Write([]byte(wgsl))

		default:
			writeJSONError(w, http.StatusNotFound, "unknown bundle file", sub)
		}
	}
}

// writeJSONError emits a consistent JSON error body shape:
//
//	{"error":"...","detail":"..."}
func writeJSONError(w http.ResponseWriter, status int, errMsg, detail string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":  errMsg,
		"detail": detail,
	})
}

// serveStudioHTML writes studio.html with an explicit Content-Type. Using the
// generic http.FileServer would also work for /, but this path is also taken
// by every History-API deep link (e.g. /v/mlp) which is not a real file.
func serveStudioHTML(w http.ResponseWriter, r *http.Request, staticFS fs.FS) {
	b, err := fs.ReadFile(staticFS, "studio.html")
	if err != nil {
		http.Error(w, "studio.html: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	_, _ = w.Write(b)
}

// staticFileServer wraps http.FileServer with explicit Content-Type overrides
// for the file kinds the studio actually serves. Go's mime package on darwin
// occasionally serves .js as text/plain; pin the types here.
func staticFileServer(staticFS fs.FS) http.Handler {
	inner := http.FileServer(http.FS(staticFS))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ".html"):
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
		case strings.HasSuffix(r.URL.Path, ".css"):
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
		case strings.HasSuffix(r.URL.Path, ".js"):
			w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		case strings.HasSuffix(r.URL.Path, ".wasm"):
			w.Header().Set("Content-Type", "application/wasm")
		}
		inner.ServeHTTP(w, r)
	})
}

// stubJSON returns a handler that responds 501 with a consistent JSON body.
// Subsequent web phases swap these out for real handlers.
func stubJSON(reason string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusNotImplemented)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":  "phase ID pending",
			"detail": reason,
		})
	}
}
