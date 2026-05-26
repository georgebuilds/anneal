package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

func kernelsCmd(args []string) int {
	return kernelsCmdW(args, os.Stdout)
}

func kernelsCmdW(args []string, w io.Writer) int {
	_, rest, err := parseFlags("kernels", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}

	if len(rest) == 0 {
		fmt.Fprintln(w, "usage: anneal kernels <model>")
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "available models:")
		for _, e := range examples.All() {
			fmt.Fprintf(w, "  %-12s  %s\n", e.Name, e.Summary)
		}
		return 1
	}

	name := rest[0]
	ex, err := examples.Get(name)
	if err != nil {
		fmt.Fprintln(w, formatError(err.Error()))
		return 1
	}

	result, err := ex.Build("webgpu")
	if err != nil {
		fmt.Fprintf(w, "build error: %v\n", err)
		return 1
	}

	// Build SINK and run schedule — no GPU required.
	a := result.Arena
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{result.Output.Node()}, nil, nil)
	items := schedule.CreateSchedule(sink, "webgpu")

	fmt.Fprintf(w, "kernels: %s — %s\n", ex.Name, ex.Summary)
	fmt.Fprintf(w, "device: webgpu\n")
	fmt.Fprintf(w, "kernels: %d\n", len(items))
	fmt.Fprintln(w)

	for i, item := range items {
		wgsl := codegen.RenderWGSL(item)

		ktype := kernelType(wgsl)
		fusedOps := countFusedOps(wgsl)

		fmt.Fprintf(w, "--- kernel %d ---\n", i)
		fmt.Fprintf(w, "type:    %s (%d fused ops)\n", ktype, fusedOps)

		// Output buffer info.
		if len(item.Bufs) > 0 {
			out := item.Bufs[0]
			fmt.Fprintf(w, "output:  buf[%d] %s  %v  %d elements\n",
				out.UOpIdx, dtTypeName(out.DType), out.Shape, out.Size)
		}

		// Input buffer info.
		if len(item.Bufs) > 1 {
			fmt.Fprintf(w, "inputs:  ")
			for j, buf := range item.Bufs[1:] {
				if j > 0 {
					fmt.Fprintf(w, ", ")
				}
				fmt.Fprintf(w, "buf[%d] %s %v", buf.UOpIdx, dtTypeName(buf.DType), buf.Shape)
				if buf.Slot == -1 {
					fmt.Fprintf(w, " (leaf)")
				}
			}
			fmt.Fprintln(w)
		}

		fmt.Fprintln(w)
		fmt.Fprint(w, wgsl)
		fmt.Fprintln(w)
	}

	return 0
}

// kernelType detects whether a WGSL shader is a reduction or elementwise kernel.
// A shader containing "for (var" has a loop body characteristic of reductions.
func kernelType(wgsl string) string {
	if strings.Contains(wgsl, "for (var") {
		return "reduction"
	}
	return "elementwise"
}

// countFusedOps counts the number of intermediate let-bindings in a WGSL shader,
// which approximates the number of fused operations.
func countFusedOps(wgsl string) int {
	return strings.Count(wgsl, "let t")
}
