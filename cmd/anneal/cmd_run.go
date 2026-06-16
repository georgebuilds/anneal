package main

import (
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
)

// runExecOpener is the seam that opens the executor `anneal run` realizes on.
// Production opens WebGPU; tests substitute a CPU executor so the build +
// realize body runs in CI without a GPU. It honours the requested device tag
// (cpu vs webgpu) so `--device=cpu` Just Works on the CLI too.
var runExecOpener = openRunExec

// openRunExec opens the executor for the requested device tag.
func openRunExec(device string) (backend.Executor, func(), error) {
	if device == "cpu" {
		dev, err := cpu.Open()
		if err != nil {
			return nil, nil, err
		}
		return dev, func() { dev.Close() }, nil
	}
	dev, err := webgpu.Open()
	if err != nil {
		return nil, nil, err
	}
	return dev, func() { dev.Close() }, nil
}

func runCmd(args []string) int {
	return runCmdW(args, os.Stdout)
}

//nolint:errcheck // best-effort write to stdout/stderr
func runCmdW(args []string, w io.Writer) int {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	flags, rest, err := parseFlags("run", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}

	if len(rest) == 0 {
		fmt.Fprintln(w, "usage: anneal run <model>")
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

	device := flags.device

	exec, closeExec, err := runExecOpener(device)
	if err != nil {
		fmt.Fprint(w, noAdapterError())
		return 1
	}
	defer closeExec()

	tensor.DefaultExecutor = exec
	defer func() { tensor.DefaultExecutor = nil }()

	result, err := ex.Build(device)
	if err != nil {
		fmt.Fprintf(w, "build error: %v\n", err)
		return 1
	}

	if err := tensor.Realize(result.Output); err != nil {
		fmt.Fprintf(w, "realize error: %v\n", err)
		return 1
	}

	data := result.Output.Data()
	shape := result.Output.Shape()

	fmt.Fprintf(w, "model: %s\n", ex.Name)
	fmt.Fprintf(w, "shape: %v\n", shape)

	n := len(data)
	if n > 8 {
		n = 8
	}
	fmt.Fprintf(w, "output (first %d values):", n)
	for i := 0; i < n; i++ {
		fmt.Fprintf(w, " %.6f", data[i])
	}
	fmt.Fprintln(w)

	return 0
}
