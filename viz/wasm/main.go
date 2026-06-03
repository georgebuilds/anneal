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
	// W4: node-inspector backend — annealNodeDetail(graphId, nodeId) → JSON.
	// Returns op/dtype/shape/parents/children/arg for one node of one
	// rendered graph. Spec: notes/anneal_web_spec.md §4, §5.2.
	js.Global().Set("annealNodeDetail", js.FuncOf(getNodeDetail))
	// W3: explain view — annealExplainOp(opName string) string (JSON).
	// Returns the explain payload: description, symbolic rules from
	// rewrite/rules/symbolic.upat, gradient rule from tensor.Gradient, and a
	// before/after mini-graph. Spec: notes/anneal_web_spec.md §4, §5.4.
	js.Global().Set("annealExplainOp", js.FuncOf(explainOp))

	// Block forever so the Go runtime stays alive while the page is open.
	select {}
}

// explainOp is the JS-facing wrapper for viz.BuildExplain. Returns one JSON
// payload per canonical op name; the studio's explain-view renderer consumes
// it. An unknown op name returns {"error":"..."} so the renderer can surface
// the blameless error message inline.
func explainOp(_ js.Value, args []js.Value) any {
	name := ""
	if len(args) > 0 && args[0].Type() == js.TypeString {
		name = args[0].String()
	}
	e, err := viz.BuildExplain(name)
	if err != nil {
		return js.ValueOf(`{"error":"` + err.Error() + `"}`)
	}
	b, err := e.ToJSON()
	if err != nil {
		return js.ValueOf(`{"error":"json marshal: ` + err.Error() + `"}`)
	}
	return js.ValueOf(string(b))
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

// getNodeDetail is the JS-facing wrapper for viz.BuildNodeDetail. The studio
// posts {graphId, nodeId} from the embedded viz iframe; this handler returns
// the JSON the drawer renders. graphId defaults to "mlp" and nodeId to "n0"
// so a parameter-less call still produces something useful for debugging.
func getNodeDetail(_ js.Value, args []js.Value) any {
	graphID, nodeID := "mlp", "n0"
	if len(args) > 0 && args[0].Type() == js.TypeString {
		graphID = args[0].String()
	}
	if len(args) > 1 && args[1].Type() == js.TypeString {
		nodeID = args[1].String()
	}
	nd, err := viz.BuildNodeDetail(graphID, nodeID)
	if err != nil {
		return js.ValueOf(`{"error":"` + err.Error() + `"}`)
	}
	b, err := nd.ToJSON()
	if err != nil {
		return js.ValueOf(`{"error":"json marshal: ` + err.Error() + `"}`)
	}
	return js.ValueOf(string(b))
}
