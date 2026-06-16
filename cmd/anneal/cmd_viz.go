//go:build !js

package main

import (
	"fmt"
	"os/exec"
	"runtime"

	"github.com/georgebuilds/anneal/viz"
)

// vizServe is the seam that lets tests substitute the blocking viz.Serve
// call (which binds a port and never returns on success) with a fake.
var vizServe = viz.Serve

// browserOpener is the seam that lets tests substitute the real
// openBrowser (which spawns an OS browser process) with a no-op.
var browserOpener = openBrowser

func vizCmd(args []string) int {
	addr := ":3000"
	url := "http://localhost:3000"
	fmt.Printf("anneal viz — static UOp graph visualizer\n")
	fmt.Printf("server: %s\n", url)
	fmt.Printf("\nopen %s in a browser to view the graph.\n", url)
	fmt.Printf("build the WASM binary for in-browser compilation:\n")
	fmt.Printf("  GOOS=js GOARCH=wasm go build -o viz/static/anneal.wasm ./viz/wasm/\n")
	fmt.Printf("  cp \"$(go env GOROOT)/misc/wasm/wasm_exec.js\" viz/static/\n\n")
	browserOpener(url)
	if err := vizServe(addr); err != nil {
		fmt.Printf("viz server error: %v\n", err)
		return 1
	}
	return 0
}

// browserRunner spawns the OS open command. Hoisted to a var so tests can
// assert openBrowser dispatches correctly without launching a real browser.
var browserRunner = func(name string, args ...string) error {
	return exec.Command(name, args...).Start()
}

func openBrowser(url string) {
	var cmd string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
	case "linux":
		cmd = "xdg-open"
	default:
		return
	}
	_ = browserRunner(cmd, url)
}
