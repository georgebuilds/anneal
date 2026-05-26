package main

import (
	"fmt"
	"os"

	// Import examples package to trigger init() registrations.
	_ "github.com/georgebuilds/anneal/examples"
)

const version = "0.0.0-dev"

const usageText = `anneal — a tensor compiler. Gradients are just rewrites.

usage:
  anneal <command> [flags]

commands:
  run       realize and execute a graph
  train     training loop with live TUI dashboard (--plain for text output)
  viz       open the UOp graph visualizer in a browser
  graph     dump the UOp DAG in textual form
  kernels   show generated WGSL with fusion boundaries annotated
  explain   show the rewrite rules that fire for one op
  doctor    WebGPU / backend environment check

flags (global):
  --device=<name>   target device (default: webgpu)
  --debug=<n>       debug verbosity level (0–3); also: DEBUG=n
  --viz             enable graph visualization; also: VIZ=1

run 'anneal help <command>' for more information on a command.
`

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		fmt.Print(usageText)
		return 0
	}

	verb := args[0]
	rest := args[1:]

	switch verb {
	case "--version", "version":
		fmt.Println("anneal " + version)
		return 0

	case "--help", "-h", "help":
		if len(rest) > 0 {
			return verbHelp(rest[0])
		}
		fmt.Print(usageText)
		return 0

	case "run":
		return runCmd(rest)

	case "train":
		return trainCmd(rest)

	case "graph":
		return graphCmd(rest)

	case "kernels":
		return kernelsCmd(rest)

	case "explain":
		return explainCmd(rest)

	case "doctor":
		return doctorCmd(rest)

	case "viz":
		return vizCmd(rest)

	default:
		fmt.Fprintf(os.Stderr, "anneal: unknown command %q\n", verb)
		fmt.Fprintf(os.Stderr, "run 'anneal help' for usage\n")
		return 1
	}
}

// verbHelp prints per-verb help text and returns 0.
func verbHelp(verb string) int {
	switch verb {
	case "run":
		fmt.Print(`usage: anneal run <model> [flags]

realize and execute a named example graph on the GPU.

flags:
  --device=<name>   target device (default: webgpu)
  --debug=<n>       debug verbosity level

`)
	case "train":
		fmt.Print(`usage: anneal train <model> [flags]

run a training loop with a live bubbletea dashboard (TUI). automatically
falls back to plain text when stdout is not a terminal, NO_COLOR is set,
or --plain is passed.

flags:
  --device=<name>   target device (default: webgpu)
  --steps=<n>       number of training steps (default: 100)
  --lr=<f>          learning rate (default: 0.050)
  --log-every=<n>   log loss every N steps (default: 10)
  --plain           plain text output; disables the TUI
  --debug=<n>       debug verbosity level

`)
	case "graph":
		fmt.Print(`usage: anneal graph <model> [flags]

dump the UOp DAG in textual form. does not require a GPU.

flags:
  --device=<name>   target device for graph annotation (default: webgpu)
  --debug=<n>       debug verbosity level

`)
	case "kernels":
		fmt.Print(`usage: anneal kernels <model> [flags]

show generated WGSL compute shaders with fusion boundaries annotated.
does not require a GPU.

flags:
  --device=<name>   target device (default: webgpu)
  --debug=<n>       debug verbosity level

`)
	case "explain":
		fmt.Print(`usage: anneal explain <op>

show the symbolic and gradient rewrite rules that fire for a given op.
does not require a GPU.

examples:
  anneal explain add
  anneal explain matmul.backward
  anneal explain symbolic

`)
	case "viz":
		fmt.Print(`usage: anneal viz

open the static UOp graph visualizer in a browser. starts an HTTP server on
:3000 and serves the forward + backward graph for mlp and conv examples.
for in-browser compilation (WASM dogfood), build anneal.wasm first:

  GOOS=js GOARCH=wasm go build -o viz/static/anneal.wasm ./viz/wasm/
  cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" viz/static/

without the WASM binary the server falls back to a REST API that runs the
real compiler natively and returns JSON — both paths produce real graphs.

`)
	case "doctor":
		fmt.Print(`usage: anneal doctor [flags]

check the WebGPU / backend environment and report device status.

flags:
  --device=<name>   target device to probe (default: webgpu)
  --debug=<n>       debug verbosity level

`)
	default:
		fmt.Printf("anneal: unknown command %q — run 'anneal help' for usage\n", verb)
		return 1
	}
	return 0
}
