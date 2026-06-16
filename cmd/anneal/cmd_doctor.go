package main

import (
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/georgebuilds/anneal/backend/webgpu"
)

// doctorProbe reads the diagnostic fields `anneal doctor` prints. Production
// opens a transient WebGPU device; tests inject a fake so the report-rendering
// body runs in CI without a GPU.
type doctorProbeResult struct {
	adapterName string
	shaderF16   bool
}

var doctorProbeFn = doctorProbeWebGPU

// doctorProbeWebGPU opens a transient WebGPU device and reads its fields.
func doctorProbeWebGPU() (doctorProbeResult, error) {
	dev, err := webgpu.Open()
	if err != nil {
		return doctorProbeResult{}, err
	}
	defer dev.Close()
	return doctorProbeResult{adapterName: dev.AdapterName(), shaderF16: dev.HasShaderF16}, nil
}

func doctorCmd(args []string) int {
	return doctorCmdW(args, os.Stdout)
}

//nolint:errcheck // best-effort write to stdout/stderr
func doctorCmdW(args []string, w io.Writer) int {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	flags, _, err := parseFlags("doctor", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}
	_ = flags

	probe, err := doctorProbeFn()
	if err != nil {
		fmt.Fprint(w, doctorFailureMsg())
		return 1
	}

	name := probe.adapterName
	backend := detectBackend()
	shaderF16 := "NO"
	if probe.shaderF16 {
		shaderF16 = "yes"
	}

	fmt.Fprintf(w, "device: %s\n", name)
	fmt.Fprintf(w, "backend: %s\n", backend)
	fmt.Fprintf(w, "shader-f16: %s\n", shaderF16)
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
