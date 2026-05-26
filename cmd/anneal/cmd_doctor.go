package main

import (
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/georgebuilds/anneal/backend/webgpu"
)

func doctorCmd(args []string) int {
	return doctorCmdW(args, os.Stdout)
}

func doctorCmdW(args []string, w io.Writer) int {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	flags, _, err := parseFlags("doctor", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}
	_ = flags

	dev, err := webgpu.Open()
	if err != nil {
		fmt.Fprint(w, doctorFailureMsg())
		return 1
	}
	defer dev.Close()

	name := dev.AdapterName()
	backend := detectBackend()

	fmt.Fprintf(w, "device: %s\n", name)
	fmt.Fprintf(w, "backend: %s\n", backend)
	fmt.Fprintf(w, "status: %s\n", bold("ready"))
	return 0
}

// detectBackend returns a human-readable backend name based on the OS.
// The actual WebGPU backend is selected at runtime by the driver; this is a
// best-effort display hint.
func detectBackend() string {
	switch runtime.GOOS {
	case "darwin":
		return "Metal"
	case "linux":
		return "Vulkan"
	case "windows":
		return "D3D12"
	default:
		return "unknown"
	}
}
