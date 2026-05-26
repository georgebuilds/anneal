//go:build !js

package main

import (
	"fmt"
	"os/exec"
	"runtime"

	"github.com/georgebuilds/anneal/viz"
)

func vizCmd(args []string) int {
	addr := ":3000"
	url := "http://localhost:3000"
	fmt.Printf("anneal viz — static UOp graph visualizer\n")
	fmt.Printf("server: %s\n", url)
	fmt.Printf("\nopen %s in a browser to view the graph.\n", url)
	fmt.Printf("build the WASM binary for in-browser compilation:\n")
	fmt.Printf("  GOOS=js GOARCH=wasm go build -o viz/static/anneal.wasm ./viz/wasm/\n")
	fmt.Printf("  cp \"$(go env GOROOT)/misc/wasm/wasm_exec.js\" viz/static/\n\n")
	openBrowser(url)
	if err := viz.Serve(addr); err != nil {
		fmt.Printf("viz server error: %v\n", err)
		return 1
	}
	return 0
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
	_ = exec.Command(cmd, url).Start()
}
