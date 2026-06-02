package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples/gpt2"
	"github.com/georgebuilds/anneal/tensor"
)

// gpt2Cmd is the top-level "anneal gpt2" verb dispatcher. It exposes one
// subcommand for now (sample); future GPT-2 work (perplexity, prompt
// completions from stdin, etc.) slots in here.
func gpt2Cmd(args []string) int {
	return gpt2CmdW(args, os.Stdout)
}

func gpt2CmdW(args []string, w io.Writer) int {
	if len(args) == 0 {
		printGPT2Usage(w)
		return 1
	}
	sub := args[0]
	rest := args[1:]
	switch sub {
	case "sample":
		return gpt2SampleCmdW(rest, w)
	case "-h", "--help", "help":
		printGPT2Usage(w)
		return 0
	default:
		fmt.Fprintf(w, "anneal gpt2: unknown subcommand %q\n", sub)
		printGPT2Usage(w)
		return 1
	}
}

// gpt2SampleCmdW parses the sample-specific flags, opens the WebGPU
// adapter, loads weights, and dispatches to gpt2.RunSampleCLI. It returns
// a process exit code so the parent dispatcher can propagate it directly.
//
//nolint:errcheck // best-effort writes to stdout/stderr
func gpt2SampleCmdW(args []string, w io.Writer) int {
	// Metal NSAutoreleasePool is thread-local; pin this goroutine to its OS
	// thread so pool create and drain always happen on the same thread.
	// Matches the cmd_run and cmd_train conventions.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	fs := flag.NewFlagSet("gpt2 sample", flag.ContinueOnError)
	device := fs.String("device", "webgpu", "target device")
	maxTokens := fs.Int("max-tokens", 20, "number of tokens to generate")
	temperature := fs.Float64("temperature", 1.0, "softmax temperature")
	topK := fs.Int("top-k", 40, "top-k filter (<=0 disables)")
	greedy := fs.Bool("greedy", false, "use argmax sampling (deterministic)")
	plain := fs.Bool("plain", false, "plain output (no header)")

	// Allow the prompt to appear before or after the flags. The first
	// non-flag arg becomes the prompt; remaining non-flag args are joined
	// with single spaces so users can write `anneal gpt2 sample The quick
	// brown fox` without quoting.
	parseArgs := make([]string, 0, len(args))
	var promptParts []string
	for _, a := range args {
		if a != "" && a[0] == '-' {
			parseArgs = append(parseArgs, a)
			continue
		}
		promptParts = append(promptParts, a)
	}
	if err := fs.Parse(parseArgs); err != nil {
		fmt.Fprintln(w, err)
		return 1
	}
	if len(promptParts) == 0 {
		// Also accept trailing args after flags (Go's flag package stops at
		// the first non-flag, but we already pre-extracted them above).
		promptParts = fs.Args()
	}
	if len(promptParts) == 0 {
		fmt.Fprintln(w, "usage: anneal gpt2 sample <prompt> [--max-tokens=N] [--temperature=F] [--top-k=K] [--greedy] [--plain]")
		return 1
	}
	prompt := strings.Join(promptParts, " ")

	dev, err := webgpu.Open()
	if err != nil {
		fmt.Fprint(w, noAdapterError())
		return 1
	}
	defer dev.Close()

	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	opts := gpt2.SampleOptions{
		MaxTokens:   *maxTokens,
		Temperature: float32(*temperature),
		TopK:        *topK,
		Greedy:      *greedy,
	}

	if err := gpt2.RunSampleCLI(w, *device, prompt, opts, *plain); err != nil {
		fmt.Fprintf(w, "gpt2: %v\n", err)
		// Make the asset-cache hint discoverable. ANNEAL_OFFLINE=1 with a
		// missing cache surfaces the URL the user can fetch manually; the
		// underlying error message from internal/assets already names it.
		if isOfflineMissingAsset(err) {
			fmt.Fprintln(w, "")
			fmt.Fprintln(w, "to fetch the assets manually (~550 MB), see the README \"GPT-2 sample\" section,")
			fmt.Fprintln(w, "or unset ANNEAL_OFFLINE to let anneal download them on first run.")
		}
		return 1
	}
	return 0
}

// printGPT2Usage prints the canonical help text for the gpt2 verb.
func printGPT2Usage(w io.Writer) {
	//nolint:errcheck // best-effort
	fmt.Fprint(w, `usage: anneal gpt2 <subcommand> [flags]

subcommands:
  sample <prompt>   sample text from GPT-2-small (HuggingFace weights)

run 'anneal gpt2 sample --help' for sample-specific flags.
`)
}

// isOfflineMissingAsset matches the canonical error string emitted by
// internal/assets.Get when ANNEAL_OFFLINE=1 is set and the cache is empty.
// Used only to decide whether to print the extra "fetch manually" hint.
func isOfflineMissingAsset(err error) bool {
	return err != nil && strings.Contains(err.Error(), "ANNEAL_OFFLINE=1")
}
