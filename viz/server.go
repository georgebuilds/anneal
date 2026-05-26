//go:build !js

package viz

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
)

//go:embed static
var staticFiles embed.FS

// Serve starts an HTTP server on addr that serves the visualizer SPA and a
// REST API endpoint at /api/graph?name=<model>.
//
// The browser first tries the WASM path (anneal.wasm in static/), falling back
// to the REST endpoint so the viz works whether or not the WASM binary has been
// compiled. Build the WASM binary with:
//
//	GOOS=js GOARCH=wasm go build -o viz/static/anneal.wasm ./viz/wasm/
func Serve(addr string) error {
	mux := http.NewServeMux()

	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("viz: embed static: %w", err)
	}
	mux.Handle("/", http.FileServer(http.FS(staticFS)))

	mux.HandleFunc("/api/graph", func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("name")
		if name == "" {
			name = "mlp"
		}
		g, err := BuildGraph(name)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		b, err := g.ToJSON()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		_, _ = w.Write(b)
	})

	return http.ListenAndServe(addr, mux)
}
