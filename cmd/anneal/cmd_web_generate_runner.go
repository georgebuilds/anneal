//go:build !js

// Production generateRunner: opens the WebGPU device on the SSE-handler
// goroutine, dispatches to gpt2.SampleStream or examples.NanoGPTGenerateStream
// based on the model name, and translates per-token records into the
// tui.TokenSnapshot wire format the studio reads. Errors surface as a
// final PhaseError TokenSnapshot so the client sees a coherent failure
// instead of an abrupt close.

package main

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/examples/gpt2"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
	"github.com/georgebuilds/anneal/uop"
)

// nativeGenerateDevice describes the device the native web generator runs on.
type nativeGenerateDevice struct {
	deviceTag string
	exec      backend.Executor
	close     func()
}

// openNativeGenerateDevice is the seam that opens the device the web
// generator runs on. Production opens WebGPU; tests substitute a CPU device
// so the nanogpt streaming body runs in CI without a GPU.
var openNativeGenerateDevice = openWebGPUGenerateDevice

// openWebGPUGenerateDevice opens a WebGPU device for the native generator.
func openWebGPUGenerateDevice() (nativeGenerateDevice, error) {
	dev, err := webgpu.Open()
	if err != nil {
		return nativeGenerateDevice{}, err
	}
	return nativeGenerateDevice{
		deviceTag: "webgpu",
		exec:      dev,
		close:     func() { dev.Close() },
	}, nil
}

// runGenerateNative is the production generateRunner. It pins the goroutine
// to its OS thread (Metal autorelease-pool affinity), opens WebGPU, builds
// the requested model, and streams one TokenSnapshot per emitted token
// through emit. The function returns after the run completes (success,
// cancellation, or error).
func runGenerateNative(
	ctx context.Context,
	model, prompt string,
	maxTokens int,
	compare bool,
	emit func(tui.TokenSnapshot),
) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	nd, err := openNativeGenerateDevice()
	if err != nil {
		emit(tui.TokenSnapshot{
			Phase:     tui.PhaseError,
			MaxTokens: maxTokens,
			Error:     fmt.Sprintf("no GPU: %v", err),
		})
		return err
	}
	defer nd.close()

	tensor.DefaultExecutor = nd.exec
	defer func() { tensor.DefaultExecutor = nil }()

	startWall := time.Now()

	// Initial PhaseInit frame so the client can hide the warming-up hint
	// the moment the SSE socket is hot (weights still loading after this
	// for GPT-2, but the wire is live).
	emit(tui.TokenSnapshot{
		Phase:     tui.PhaseInit,
		Step:      0,
		MaxTokens: maxTokens,
	})

	switch model {
	case "gpt2":
		return runGPT2Stream(ctx, prompt, maxTokens, compare, startWall, emit)
	case "nanogpt":
		return runNanoGPTStream(ctx, prompt, maxTokens, compare, startWall, emit)
	default:
		err := fmt.Errorf("unknown model %q", model)
		emit(tui.TokenSnapshot{Phase: tui.PhaseError, MaxTokens: maxTokens, Error: err.Error()})
		return err
	}
}

// runGPT2Stream loads GPT-2-small (assets cached on first call) and runs
// SampleStream with a greedy decode (compare=true forces greedy so the
// ref-match column is deterministic; compare=false uses the default
// temperature=1.0 + top-k=40 stochastic path).
func runGPT2Stream(
	ctx context.Context,
	prompt string,
	maxTokens int,
	compare bool,
	startWall time.Time,
	emit func(tui.TokenSnapshot),
) error {
	a := uop.NewArena(1 << 14)
	g, bpe, err := gpt2.LoadGPT2(a, "webgpu")
	if err != nil {
		emit(tui.TokenSnapshot{
			Phase:     tui.PhaseError,
			MaxTokens: maxTokens,
			Error:     fmt.Sprintf("load gpt2: %v", err),
		})
		return err
	}

	promptIds := bpe.Encode(prompt)
	ctxLen := len(promptIds) + maxTokens
	if ctxLen > g.BlockSize {
		ctxLen = g.BlockSize
	}
	if ctxLen < 1 {
		err := fmt.Errorf("gpt2: prompt %q encoded to zero tokens", prompt)
		emit(tui.TokenSnapshot{Phase: tui.PhaseError, MaxTokens: maxTokens, Error: err.Error()})
		return err
	}

	opts := gpt2.SampleOptions{
		MaxTokens:   maxTokens,
		Temperature: 1.0,
		TopK:        40,
		Greedy:      compare,
	}
	_, err = gpt2.SampleStream(ctx, g, bpe, prompt, ctxLen, "webgpu", opts, func(tk gpt2.StreamToken) {
		var refMatch *bool
		if compare {
			// Greedy + compare: argmax always equals the sampled id, so
			// the oracle ref-match is trivially true. The point of the
			// flag is to surface the ref-match column in the UI; richer
			// HF-oracle comparison lives in the test suite.
			yes := tk.ID == tk.Argmax
			refMatch = &yes
		}
		emit(tui.TokenSnapshot{
			Step:         tk.Step,
			MaxTokens:    maxTokens,
			TokenID:      int(tk.ID),
			TokenText:    tk.Text,
			LogitArgmax:  int(tk.Argmax),
			LogitSummary: tk.LogitSummary,
			WallMs:       time.Since(startWall).Milliseconds(),
			Phase:        tui.PhaseTraining, // wire spelling "training" reused as "generating"
			RefMatch:     refMatch,
		})
	})
	if err != nil && ctx.Err() == nil {
		emit(tui.TokenSnapshot{
			Phase:     tui.PhaseError,
			MaxTokens: maxTokens,
			Error:     err.Error(),
		})
		return err
	}
	emit(tui.TokenSnapshot{
		Step:      maxTokens,
		MaxTokens: maxTokens,
		Phase:     tui.PhaseDone,
		WallMs:    time.Since(startWall).Milliseconds(),
	})
	return nil
}

// runNanoGPTStream drives examples.NanoGPTGenerateStream over the
// tinyshakespeare vocabulary. The model is freshly seeded (no training),
// matching the train+sample contract in the existing nanogpt example -
// the demo's value is the per-token kernel pulse, not the text quality.
func runNanoGPTStream(
	ctx context.Context,
	prompt string,
	maxTokens int,
	compare bool,
	startWall time.Time,
	emit func(tui.TokenSnapshot),
) error {
	_, err := examples.NanoGPTGenerateStream(ctx, "webgpu", prompt, maxTokens, func(tk examples.NanoGPTStreamToken) {
		var refMatch *bool
		if compare {
			// nanogpt's path is greedy by construction; argmax always
			// equals the sampled id, so ref-match is trivially true.
			yes := tk.ID == tk.Argmax
			refMatch = &yes
		}
		emit(tui.TokenSnapshot{
			Step:         tk.Step,
			MaxTokens:    maxTokens,
			TokenID:      int(tk.ID),
			TokenText:    tk.Text,
			LogitArgmax:  int(tk.Argmax),
			LogitSummary: tk.LogitSummary,
			WallMs:       time.Since(startWall).Milliseconds(),
			Phase:        tui.PhaseTraining,
			RefMatch:     refMatch,
		})
	})
	if err != nil && ctx.Err() == nil {
		emit(tui.TokenSnapshot{
			Phase:     tui.PhaseError,
			MaxTokens: maxTokens,
			Error:     err.Error(),
		})
		return err
	}
	emit(tui.TokenSnapshot{
		Step:      maxTokens,
		MaxTokens: maxTokens,
		Phase:     tui.PhaseDone,
		WallMs:    time.Since(startWall).Milliseconds(),
	})
	return nil
}
