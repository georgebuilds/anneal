//go:build !js

// Production trainRunner: opens the WebGPU device on the SSE-handler
// goroutine, builds the registered example's TrainConfig with the W5
// snapshot channel, and drives examples.Train. Errors from the device
// open or the train loop surface to the handler so the caller can choose
// to emit them as `event: error` SSE frames.

package main

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
)

// runTrainNative is the production trainRunner. It opens the WebGPU adapter
// and runs the requested example end to end, pushing one Snapshot per
// logged step into snap (the SSE handler's bridge into the wire writer).
//
// Lifecycle:
//  1. Metal's NSAutoreleasePool is thread-local; LockOSThread keeps the
//     pool create/drain on the same OS thread.
//  2. Open the WebGPU device; on error fabricate a final error snapshot so
//     the client sees a coherent failure rather than an abrupt close.
//  3. Build the per-step Snapshot template (model name, device, hyperparams)
//     and wire cfg.SnapshotFn = snap; the snapshot shim translates the
//     legacy logFn into Snapshot puts (same plumbing as cmd_train.go).
//  4. Call examples.Train. After it returns, emit one PhaseDone snapshot
//     so the client's UI flips to "done" before the channel closes.
func runTrainNative(ctx context.Context, model string, steps int, snap func(tui.Snapshot)) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	ex, err := examples.Get(model)
	if err != nil {
		// Already validated by handleSSETrain, but guard against drift.
		snap(tui.Snapshot{Phase: tui.PhaseError, Error: err.Error()})
		return err
	}

	dev, err := webgpu.Open()
	if err != nil {
		snap(tui.Snapshot{
			Phase: tui.PhaseError,
			Error: fmt.Sprintf("no GPU: %v", err),
		})
		return err
	}
	defer dev.Close()

	tensor.DefaultExecutor = dev
	defer func() { tensor.DefaultExecutor = nil }()

	adapterName := dev.AdapterName()
	backend := detectBackend()

	cfg := examples.TrainConfig{
		Steps:    steps,
		LR:       0.05,
		LogEvery: 1,
		Batch:    16,
	}

	startWall := time.Now()
	base := tui.Snapshot{
		MaxSteps:     cfg.Steps,
		AdapterName:  adapterName,
		BackendName:  backend,
		LearningRate: cfg.LR,
		BatchSize:    cfg.Batch,
	}
	cfg.SnapshotFn = func(s tui.Snapshot) {
		// Decorate per-snapshot fields that the shim doesn't know about
		// (model name, device tag) so the wire frame is self-describing.
		if s.AdapterName == "" {
			s.AdapterName = adapterName
		}
		if s.BackendName == "" {
			s.BackendName = backend
		}
		s.MaxSteps = cfg.Steps
		snap(s)
	}
	logFn := snapshotShimLogFn(&base, startWall, cfg.SnapshotFn)

	if err := ex.Train("webgpu", cfg, logFn); err != nil {
		snap(tui.Snapshot{
			Phase: tui.PhaseError,
			Step:  steps,
			Error: err.Error(),
		})
		return err
	}

	// One final PhaseDone snapshot so the dashboard can flip the buttons
	// before the channel closes (the SSE `event: done` comes after).
	snap(tui.Snapshot{
		Step:        steps,
		MaxSteps:    steps,
		Phase:       tui.PhaseDone,
		AdapterName: adapterName,
		BackendName: backend,
		WallMs:      time.Since(startWall).Milliseconds(),
	})
	return nil
}
