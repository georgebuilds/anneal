package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
)

func trainCmd(args []string) int {
	return trainCmdW(args, os.Stdout)
}

func trainCmdW(args []string, w io.Writer) int {
	// Metal NSAutoreleasePool is thread-local; pin this goroutine to its OS
	// thread so pool create and drain always happen on the same thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	device := fs.String("device", "webgpu", "target device")
	debug := fs.Int("debug", 0, "debug verbosity level (0–3)")
	viz := fs.Bool("viz", false, "enable graph visualization")
	steps := fs.Int("steps", 100, "number of training steps")
	lr := fs.Float64("lr", 0.05, "learning rate")
	logEvery := fs.Int("log-every", 10, "log loss every N steps")
	plain := fs.Bool("plain", false, "plain text output (disables the TUI)")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintln(w, err)
		return 1
	}

	explicitlySet := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { explicitlySet[f.Name] = true })
	if !explicitlySet["viz"] {
		if v := os.Getenv("VIZ"); v == "1" {
			*viz = true
		}
	}
	if !explicitlySet["debug"] {
		if v := os.Getenv("DEBUG"); v != "" {
			var n int
			if _, err := fmt.Sscanf(v, "%d", &n); err == nil {
				*debug = n
			}
		}
	}
	_ = *debug
	_ = *viz

	rest := fs.Args()
	if len(rest) == 0 {
		fmt.Fprintln(w, "usage: anneal train <model> [--steps=N] [--lr=F] [--log-every=N] [--plain]")
		fmt.Fprintln(w)
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

	adapterName := dev.AdapterName()
	backend := detectBackend()

	cfg := examples.TrainConfig{
		Steps:    *steps,
		LR:       float32(*lr),
		LogEvery: *logEvery,
	}

	// Activate the TUI when writing to an interactive TTY, NO_COLOR is not set,
	// and --plain was not requested. The plain path is the CI/pipe/test path.
	if !*plain && !noColor() && isTerminalWriter(w) {
		return trainWithTUI(ex, cfg, adapterName, backend, *device)
	}

	// Plain output path: used for --plain, NO_COLOR, non-TTY output, and tests.
	fmt.Fprintf(w, "training %s — %s\n", ex.Name, ex.Summary)
	fmt.Fprintf(w, "device: %s (%s)\n", backend, adapterName)
	fmt.Fprintf(w, "steps: %d · lr: %.3f · batch: auto\n", cfg.Steps, cfg.LR)
	fmt.Fprintln(w)

	logFn := func(step int, loss float32) {
		fmt.Fprintf(w, "step %d: loss=%.6f\n", step, loss)
	}
	if err := ex.Train(*device, cfg, logFn); err != nil {
		fmt.Fprintf(w, "train error: %v\n", err)
		return 1
	}
	fmt.Fprintf(w, "\ndone — %d steps\n", cfg.Steps)
	return 0
}

// trainWithTUI runs the training loop with the bubbletea dashboard.
// This goroutine is OS-locked for Metal and runs training directly; the TUI
// runs in a separate goroutine receiving updates via tea.Program.Send.
func trainWithTUI(
	ex *examples.Example,
	cfg examples.TrainConfig,
	adapterName, backend, device string,
) int {
	m := tui.New(tui.Config{
		Device:     adapterName,
		Backend:    backend,
		ModelName:  ex.Name,
		TotalSteps: cfg.Steps,
	})

	p := tea.NewProgram(m, tea.WithAltScreen())

	// TUI renders in a background goroutine (no Metal calls there).
	tuiDone := make(chan error, 1)
	go func() {
		_, err := p.Run()
		tuiDone <- err
	}()

	// Wire schedule stats so every Realize call pushes live compiler counts.
	tui.SetStatsHook(p)
	defer tui.ClearStatsHook()

	// Per-step callback: smooth progress bar without loss-eval overhead.
	cfg.OnStep = func(step int) {
		p.Send(tui.StepMsg{Step: step})
	}

	// Loss callback: sent every LogEvery steps.
	logFn := func(step int, loss float32) {
		p.Send(tui.LossMsg{Step: step, Loss: loss})
	}

	var trainErr error
	if err := ex.Train(device, cfg, logFn); err != nil {
		p.Send(tui.ErrMsg{Err: err})
		trainErr = err
	} else {
		p.Send(tui.DoneMsg{})
	}

	// Wait for user to press q (or TUI to exit for any reason).
	if tuiErr := <-tuiDone; tuiErr != nil && trainErr == nil {
		fmt.Fprintf(os.Stderr, "tui error: %v\n", tuiErr)
	}

	if trainErr != nil {
		return 1
	}
	return 0
}

// isTerminalWriter reports whether w is an interactive terminal.
func isTerminalWriter(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}
