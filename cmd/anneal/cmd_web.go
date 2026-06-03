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
	"strings"

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
				strings.HasPrefix(r.URL.Path, "/api/") ||
				strings.HasPrefix(r.URL.Path, "/sse/") {
				http.NotFound(w, r)
				return
			}
		}
		serveStudioHTML(w, r, staticFS)
	})

	mux.Handle("/static/", http.StripPrefix("/static/", staticFileServer(staticFS)))

	mux.HandleFunc("/api/device",         stubJSON("device probe not yet implemented"))
	mux.HandleFunc("/api/runs",           stubJSON("run cache reader not yet implemented"))
	mux.HandleFunc("/api/compile/tuned",  stubJSON("native BEAM compile not yet implemented"))
	mux.HandleFunc("/sse/train",          stubJSON("train SSE not yet implemented"))
	mux.HandleFunc("/sse/generate",       stubJSON("generate SSE not yet implemented"))

	return mux
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
