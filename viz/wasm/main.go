//go:build js && wasm

// Command viz/wasm is the WASM entry point for the anneal visualizer.
// It compiles the real anneal compiler (frontend + rewrite + scheduler) to
// WebAssembly and exposes annealGetGraph(name) to the browser.
//
// Build with:
//
//	GOOS=js GOARCH=wasm go build -o viz/static/anneal.wasm ./viz/wasm/
//
// Copy wasm_exec.js alongside the binary:
//
//	cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" viz/static/
package main

import (
	"syscall/js"

	_ "github.com/georgebuilds/anneal/examples" // registers mlp, conv via init()
	"github.com/georgebuilds/anneal/viz"
)

func main() {
	// Expose annealGetGraph(name string) string (JSON) on the global JS object.
	js.Global().Set("annealGetGraph", js.FuncOf(getGraph))
	// Expose annealGetTimeline(name string) string (JSON) — multi-stage scrub.
	js.Global().Set("annealGetTimeline", js.FuncOf(getTimeline))
	// W2: kernels view — annealGetKernels(name string) string (JSON).
	// Returns kernel set: id, op_count, buffers_in/out, shape, wgsl, fusion_spans.
	// Spec: notes/anneal_web_spec.md §4, §5.3.
	js.Global().Set("annealGetKernels", js.FuncOf(getKernels))

	// Block forever so the Go runtime stays alive while the page is open.
	select {}
}

func getGraph(_ js.Value, args []js.Value) any {
	name := "mlp"
	if len(args) > 0 && args[0].Type() == js.TypeString {
		name = args[0].String()
	}
	g, err := viz.BuildGraph(name)
	if err != nil {
		return js.ValueOf(`{"error":"` + err.Error() + `"}`)
	}
	b, err := g.ToJSON()
	if err != nil {
		return js.ValueOf(`{"error":"json marshal: ` + err.Error() + `"}`)
	}
	return js.ValueOf(string(b))
}

func getTimeline(_ js.Value, args []js.Value) any {
	name := "mlp"
	if len(args) > 0 && args[0].Type() == js.TypeString {
		name = args[0].String()
	}
	t, err := viz.BuildTimeline(name)
	if err != nil {
		return js.ValueOf(`{"error":"` + err.Error() + `"}`)
	}
	b, err := t.ToJSON()
	if err != nil {
		return js.ValueOf(`{"error":"json marshal: ` + err.Error() + `"}`)
	}
	return js.ValueOf(string(b))
}

// getKernels is the JS-facing wrapper for viz.BuildKernels. Returns one JSON
// document per model name; the studio's kernels-view renderer consumes it.
func getKernels(_ js.Value, args []js.Value) any {
	name := "mlp"
	if len(args) > 0 && args[0].Type() == js.TypeString {
		name = args[0].String()
	}
	k, err := viz.BuildKernels(name)
	if err != nil {
		return js.ValueOf(`{"error":"` + err.Error() + `"}`)
	}
	b, err := k.ToJSON()
	if err != nil {
		return js.ValueOf(`{"error":"json marshal: ` + err.Error() + `"}`)
	}
	return js.ValueOf(string(b))
}
