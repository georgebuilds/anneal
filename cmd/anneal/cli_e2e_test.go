package main

import (
	"os"
	"runtime"
	"testing"

	"github.com/rogpeppe/go-internal/testscript"

	"github.com/georgebuilds/anneal/backend/webgpu"
)

// TestMain makes the test binary double as the `anneal` binary when invoked
// from inside a testscript .txtar script. testscript copies this binary into
// $PATH under each registered name; when re-executed under that name we run
// the real CLI entry point.
func TestMain(m *testing.M) {
	// testscript.Main races with -coverprofile (the test binary writes coverage
	// to GOCOVERDIR before the parent harness has set it up), so we stay on
	// RunMain. The deprecation warning is intentional.
	os.Exit(testscript.RunMain(m, map[string]func() int{ //nolint:staticcheck // see comment
		"anneal": func() int { return run(os.Args[1:]) },
	}))
}

// TestCLIScripts runs every .txtar script under testdata/script. Each script
// is one end-to-end test: it invokes the real `anneal` binary and asserts on
// stdout, stderr, and exit code. The `[gpu]` condition lets a script opt in
// to GPU paths and skip cleanly on machines without a WebGPU adapter.
func TestCLIScripts(t *testing.T) {
	testscript.Run(t, testscript.Params{
		Dir: "testdata/script",
		Condition: func(cond string) (bool, error) {
			if cond == "gpu" {
				return hasGPU(), nil
			}
			return false, nil
		},
	})
}

// hasGPU returns true when a WebGPU adapter is available. The probe opens and
// immediately closes a device on a locked OS thread.
func hasGPU() bool {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := webgpu.Open()
	if err != nil {
		return false
	}
	dev.Close()
	return true
}
