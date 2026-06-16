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

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/backend/webgpu"
	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tui"
)

// nativeTrainDevice describes the device the native web trainer runs on:
// the wire device tag, an adapter display name, the executor, and a closer.
type nativeTrainDevice struct {
	deviceTag   string
	adapterName string
	exec        backend.Executor
	close       func()
}

// openNativeTrainDevice is the seam that opens the device the web trainer
// runs on. Production opens WebGPU; tests substitute a CPU device so the full
// train body runs in CI without a GPU.
var openNativeTrainDevice = openWebGPUTrainDevice

// openWebGPUTrainDevice opens a WebGPU device for the native web trainer.
func openWebGPUTrainDevice() (nativeTrainDevice, error) {
	dev, err := webgpu.Open()
	if err != nil {
		return nativeTrainDevice{}, err
	}
	return nativeTrainDevice{
		deviceTag:   "webgpu",
		adapterName: dev.AdapterName(),
		exec:        dev,
		close:       func() { dev.Close() },
	}, nil
}

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

	nd, err := openNativeTrainDevice()
	if err != nil {
		snap(tui.Snapshot{
			Phase: tui.PhaseError,
			Error: fmt.Sprintf("no GPU: %v", err),
		})
		return err
	}
	defer nd.close()

	tensor.DefaultExecutor = nd.exec
	defer func() { tensor.DefaultExecutor = nil }()

	adapterName := nd.adapterName
	backendName := detectBackend()

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
		BackendName:  backendName,
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
			s.BackendName = backendName
		}
		s.MaxSteps = cfg.Steps
		snap(s)
	}
	logFn := snapshotShimLogFn(&base, startWall, cfg.SnapshotFn)

	if err := ex.Train(nd.deviceTag, cfg, logFn); err != nil {
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
		BackendName: backendName,
		WallMs:      time.Since(startWall).Milliseconds(),
	})
	return nil
}
