package main

import (
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
)

func runCmd(args []string) int {
	return runCmdW(args, os.Stdout)
}

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

	dev, err := webgpu.Open()
	if err != nil {
		fmt.Fprint(w, noAdapterError())
		return 1
	}
	defer dev.Close()

	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	device := flags.device

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
