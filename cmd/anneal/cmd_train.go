package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/georgebuilds/anneal/backend/cpu"
	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/internal/bundle"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
)

func trainCmd(args []string) int {
	return trainCmdW(args, os.Stdout)
}

//nolint:errcheck // best-effort write to stdout/stderr
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
	batch := fs.Int64("batch", 16, "batch size for dynamic-batch models; static models ignore this")
	// --bundle is opt-in (default OFF) per spec §6: CLI runs do not write
	// to ~/.cache/anneal/runs/ unless explicitly asked. ANNEAL_BUNDLE=1
	// is the env equivalent. anneal web wires this sink unconditionally
	// (see notes/web_progress.md decision 4).
	enableBundle := fs.Bool("bundle", false, "write a run bundle to "+bundle.EnvVar+
		" (or default cache); default OFF; also: ANNEAL_BUNDLE=1")

	// Extract the model name from anywhere in args so callers can write either
	// "anneal train --steps=N nanogpt" or the more natural
	// "anneal train nanogpt --steps=N". The Go flag package stops at the
	// first non-flag, so without this swap the trailing flags are silently
	// dropped (defaults win). The first non-flag arg is treated as the model.
	parseArgs := make([]string, 0, len(args))
	var model string
	for _, a := range args {
		if model == "" && a != "" && a[0] != '-' {
			model = a
			continue
		}
		parseArgs = append(parseArgs, a)
	}
	if err := fs.Parse(parseArgs); err != nil {
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
	if !explicitlySet["bundle"] {
		if v := os.Getenv("ANNEAL_BUNDLE"); v == "1" {
			*enableBundle = true
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

	// If no positional model was provided in pre-parse, fall back to fs.Args()
	// (in case the caller passed the model AFTER the flags via the older
	// "anneal train --steps=N <model>" form).
	if model == "" {
		rest := fs.Args()
		if len(rest) == 0 {
			fmt.Fprintln(w, "usage: anneal train <model> [--steps=N] [--lr=F] [--log-every=N] [--batch=N] [--plain]")
			fmt.Fprintln(w)
			fmt.Fprintln(w, "available models:")
			for _, e := range examples.All() {
				_, _ = fmt.Fprintf(w, "  %-12s  %s\n", e.Name, e.Summary)
			}
			return 1
		}
		model = rest[0]
	}

	ex, err := examples.Get(model)
	if err != nil {
		fmt.Fprintln(w, formatError(err.Error()))
		return 1
	}

	var (
		adapterName string
		backend     string
	)
	switch *device {
	case "webgpu":
		dev, openErr := webgpu.Open()
		if openErr != nil {
			fmt.Fprint(w, noAdapterError())
			return 1
		}
		defer dev.Close()
		tensor.DefaultExecutor = dev
		defer func() { tensor.DefaultExecutor = nil }()
		adapterName = dev.AdapterName()
		backend = detectBackend()
	case "cpu":
		dev, openErr := cpu.Open()
		if openErr != nil {
			_, _ = fmt.Fprintf(w, "cpu backend: %v\n", openErr)
			return 1
		}
		defer dev.Close()
		tensor.DefaultExecutor = dev
		defer func() { tensor.DefaultExecutor = nil }()
		adapterName = "cpu (pure Go interpreter)"
		backend = "cpu"
	default:
		_, _ = fmt.Fprintf(w, "unsupported --device %q; choose webgpu or cpu\n", *device)
		return 1
	}

	cfg := examples.TrainConfig{
		Steps:    *steps,
		LR:       float32(*lr),
		LogEvery: *logEvery,
		Batch:    *batch,
	}
	// LogText sink routes arbitrary text (e.g. a nanoGPT generation sample
	// at end of training) into the same writer used for loss lines. Plain
	// path: write to w (which may be a bytes.Buffer in tests). TUI path:
	// wired below in trainWithTUI.
	cfg.LogText = func(s string) { fmt.Fprint(w, s) }

	// Optional bundle sink (W1). Writer is nil unless --bundle / env so
	// a default CLI run produces no disk side effects.
	bw := maybeOpenBundle(w, *enableBundle, ex.Name, adapterName, backend, *device, cfg)
	defer func() {
		if bw != nil {
			_ = bw.Close()
		}
	}()

	// Activate the TUI when writing to an interactive TTY, NO_COLOR is not set,
	// and --plain was not requested. The plain path is the CI/pipe/test path.
	if !*plain && !noColor() && isTerminalWriter(w) {
		return trainWithTUI(ex, cfg, adapterName, backend, *device, bw)
	}

	// Plain output path: used for --plain, NO_COLOR, non-TTY output, and tests.
	_, _ = fmt.Fprintf(w, "training %s — %s\n", ex.Name, ex.Summary)
	_, _ = fmt.Fprintf(w, "device: %s (%s)\n", backend, adapterName)
	_, _ = fmt.Fprintf(w, "steps: %d · lr: %.3f · batch: %d\n", cfg.Steps, cfg.LR, cfg.Batch)
	if bw != nil {
		_, _ = fmt.Fprintf(w, "bundle: %s\n", bw.Path())
	}
	fmt.Fprintln(w)

	// W5 plain path: the trainer pushes one Snapshot per logged step into
	// cfg.SnapshotFn; the plain renderer reads from it. The bundle sink
	// reads the same Snapshot. This is the same single-source design the
	// TUI path uses, just rendered as one line per step instead of into a
	// bubbletea Model.
	startWall := time.Now()
	baseSnap := tui.Snapshot{
		MaxSteps:     cfg.Steps,
		AdapterName:  adapterName,
		BackendName:  backend,
		DeviceTag:    *device,
		LearningRate: float32(*lr),
		BatchSize:    *batch,
	}
	cfg.SnapshotFn = func(snap tui.Snapshot) {
		// Byte-identical to the pre-W5 logFn line: "step %d: loss=%.6f\n".
		// Anything else the Snapshot carries (compiler stats, sample text)
		// is invisible on the plain path by design — the CLI line format
		// is pinned by spec §11.5.
		_, _ = fmt.Fprintf(w, "step %d: loss=%.6f\n", snap.Step, snap.Loss)
		if bw != nil {
			_ = bw.AppendLoss(bundle.LossRow{
				Step:   snap.Step,
				Loss:   snap.Loss,
				WallMs: snap.WallMs,
			})
		}
	}
	logFn := snapshotShimLogFn(&baseSnap, startWall, cfg.SnapshotFn)
	if err := ex.Train(*device, cfg, logFn); err != nil {
		_, _ = fmt.Fprintf(w, "train error: %v\n", err)
		if bw != nil {
			_ = bw.Finalize(time.Since(startWall).Milliseconds())
		}
		return 1
	}
	_, _ = fmt.Fprintf(w, "\ndone — %d steps\n", cfg.Steps)
	if bw != nil {
		_ = bw.Finalize(time.Since(startWall).Milliseconds())
	}
	return 0
}

// snapshotShimLogFn is the W5 back-compat shim that lets older callers
// (and existing Example.Train methods) keep their `(step int, loss float32)`
// interface while emitting Snapshots through the new channel.
//
// The shim copies `base` (the static run-level fields filled at startup),
// patches Step/Loss/HasLoss/WallMs from the per-step pair, and hands the
// resulting Snapshot to sink. When sink is nil the shim is a no-op so a
// caller wiring only the legacy contract still trains successfully.
func snapshotShimLogFn(base *tui.Snapshot, startWall time.Time, sink func(tui.Snapshot)) func(int, float32) {
	return func(step int, loss float32) {
		if sink == nil {
			return
		}
		snap := *base
		snap.Step = step
		snap.Loss = loss
		snap.HasLoss = true
		snap.WallMs = time.Since(startWall).Milliseconds()
		// The shim only knows "a step was logged" — it does not know that
		// training has completed. PhaseDone is set by the orchestrator
		// after Example.Train returns; here we always emit PhaseTraining
		// so the "done" status only flips once on the explicit DoneMsg.
		snap.Phase = tui.PhaseTraining
		sink(snap)
	}
}

// maybeOpenBundle opens a bundle.Writer when enable is true (per --bundle
// or ANNEAL_BUNDLE=1), or returns nil. A failure to open is logged but
// does not block training — the bundle is a sink, not a precondition.
func maybeOpenBundle(w io.Writer, enable bool, model, adapter, backendName, device string, cfg examples.TrainConfig) *bundle.Writer {
	if !enable {
		return nil
	}
	root, err := bundle.EnvOrDefault()
	if err != nil {
		_, _ = fmt.Fprintf(w, "bundle: skip (cannot resolve root): %v\n", err)
		return nil
	}
	bw, err := bundle.NewWriter(root, model, bundle.KindTrain)
	if err != nil {
		_, _ = fmt.Fprintf(w, "bundle: skip (cannot create writer): %v\n", err)
		return nil
	}
	// Provenance: version + adapter + backend. WGSL hash and git rev are
	// left empty here — the trainer-side stats hook will fill them in a
	// future W step once the headless snapshot channel exists.
	if err := bw.SetProvenance(version, "", adapter, backendName, "", nil); err != nil {
		_, _ = fmt.Fprintf(w, "bundle: warn (set provenance): %v\n", err)
	}
	if err := bw.WriteConfig(bundle.Config{
		Model:  model,
		Device: device,
		Hyperparams: map[string]any{
			"steps":     cfg.Steps,
			"lr":        cfg.LR,
			"log_every": cfg.LogEvery,
			"batch":     cfg.Batch,
		},
	}); err != nil {
		_, _ = fmt.Fprintf(w, "bundle: warn (write config): %v\n", err)
	}
	return bw
}

// trainWithTUI runs the training loop with the bubbletea dashboard.
// This goroutine is OS-locked for Metal and runs training directly; the TUI
// runs in a separate goroutine receiving updates via tea.Program.Send.
//
// bw, when non-nil, captures loss rows into the optional run bundle (W1).
func trainWithTUI(
	ex *examples.Example,
	cfg examples.TrainConfig,
	adapterName, backend, device string,
	bw *bundle.Writer,
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

	// Route arbitrary text emissions (e.g. nanoGPT's final sample) to
	// stderr while the TUI owns stdout. The user sees the sample after the
	// TUI exits.
	cfg.LogText = func(s string) { fmt.Fprint(os.Stderr, s) }

	// W5 TUI path: the trainer's per-step Snapshot flows into the TUI as a
	// SnapshotMsg and into the bundle writer via the same SnapshotFn. The
	// byte-identical render guarantee comes from Model.Update mapping
	// SnapshotMsg fields onto the same internal state the legacy LossMsg
	// path would have produced.
	startWall := time.Now()
	baseSnap := tui.Snapshot{
		MaxSteps:     cfg.Steps,
		AdapterName:  adapterName,
		BackendName:  backend,
		DeviceTag:    device,
		LearningRate: cfg.LR,
		BatchSize:    cfg.Batch,
	}
	cfg.SnapshotFn = func(snap tui.Snapshot) {
		p.Send(tui.SnapshotMsg{Snapshot: snap})
		if bw != nil {
			_ = bw.AppendLoss(bundle.LossRow{
				Step:   snap.Step,
				Loss:   snap.Loss,
				WallMs: snap.WallMs,
			})
		}
	}
	logFn := snapshotShimLogFn(&baseSnap, startWall, cfg.SnapshotFn)

	var trainErr error
	if err := ex.Train(device, cfg, logFn); err != nil {
		p.Send(tui.ErrMsg{Err: err})
		trainErr = err
	} else {
		p.Send(tui.DoneMsg{})
	}

	if bw != nil {
		_ = bw.Finalize(time.Since(startWall).Milliseconds())
	}

	// Wait for user to press q (or TUI to exit for any reason).
	if tuiErr := <-tuiDone; tuiErr != nil && trainErr == nil {
		_, _ = fmt.Fprintf(os.Stderr, "tui error: %v\n", tuiErr)
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
