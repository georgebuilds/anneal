package webgpu

import (
	"errors"
	"fmt"
	"os"
	"runtime"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

// Device holds an open WebGPU device and its associated queue.
//
// ── Threading model (load-bearing - do not bypass) ───────────────────────────
//
// Every Metal-touching operation MUST run on the single GPU-owner goroutine,
// which is locked to one OS thread for its entire lifetime (see gpuOwnerLoop).
// The public entry points (Run, RunSymbolic, DispatchSymKernel, CompileSymKernel,
// Open, Close) funnel their work through that goroutine via d.onGPU; the
// unexported *Locked helpers they call assume they already run there and must
// never call onGPU themselves (that would deadlock the owner on itself).
//
// Why this is required: gogpu/wgpu's Metal HAL drives every native call inside
// an NSAutoreleasePool that is created and drained within a single HAL function
// (e.g. Device.WaitIdle). An ObjC autorelease pool is thread-affine - it must be
// drained on the OS thread that created it. Go's scheduler freely migrates a
// goroutine across OS threads at blocking syscalls (Metal's waitUntilCompleted
// is one), so a pool created on thread A can be drained on thread B → SIGSEGV in
// the Metal runtime. Pinning all Metal work to one never-unlocked OS thread makes
// every create/drain pair happen on that same thread, eliminating the migration.
//
// readBuffer additionally avoids wgpu's Buffer.Map, which spawns its OWN unpinned
// goroutine to run Poll(PollWait)→WaitIdle. Locking the owner goroutine does not
// pin a goroutine the library spawns internally, so readBuffer drives the map
// resolution itself (MapAsync + Poll(PollWait)) on the owner thread instead.
type Device struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	// HasShaderF16 is true when the adapter exposed the shader-f16 extension and the
	// device was opened with it. Kernels that use f16 dtypes require this; the
	// executor fails closed if it is false (no silent f32 fallback).
	HasShaderF16 bool

	// Collaborators (see renderer.go / compiler.go / allocator.go / program.go
	// / buffer.go). The Renderer is stateless; the Compiler owns the static and
	// symbolic pipeline caches; Allocators are created per Run / RunSymbolic /
	// Dispatch call and reset at the end so per-run buffers are dedup'd and
	// released cleanly.
	renderer renderer
	compiler *compiler

	// jobs delivers closures to the GPU-owner goroutine. Closed by Close, which
	// terminates the owner goroutine (and thereby its locked OS thread).
	jobs chan gpuJob

	// ── Stateful realize cache (opt-in, set up by BeginRealizeScope) ──────────
	// When realizeOn is set for a Run, intermediate (slotted) buffers are
	// allocated persistently (one per node, NO slot reuse) and cached by node
	// UOpIdx in realizeCache, so a later Run in the same scope reuses them and
	// skips their producer kernels. The cache is scoped by (realizeScopeID,
	// realizeScopeGen): a change frees the whole cache. All fields below are
	// touched only on the GPU-owner goroutine except realizeOn / pendingScope*,
	// which BeginRealizeScope sets just before the Run that consumes them (the
	// jobs-channel send to the owner goroutine is the happens-before barrier).
	realizeOn       bool
	pendingScopeID  uint64
	pendingScopeGen uint64
	realizeScopeID  uint64
	realizeScopeGen uint64
	realizeCache    map[uint32]*deviceBuffer
}

// gpuJob is a unit of Metal-touching work handed to the GPU-owner goroutine.
type gpuJob struct {
	fn   func() error
	done chan error
}

// gpuOwnerLoop is the body of the single GPU-owner goroutine. On darwin it
// locks itself to one OS thread permanently and never unlocks: when the loop
// returns (jobs closed) while still locked, the Go runtime terminates the
// underlying OS thread, which is the desired teardown. Every job runs to
// completion on this one thread, so all NSAutoreleasePool create/drain pairs
// share it.
//
// On non-darwin platforms (Vulkan, DX12), wgpu does not need this thread
// pinning: there are no autorelease pools, and Vulkan worker threads spawned
// by wgpu are independent of any goroutine's OS thread. Locking on Linux is
// actively harmful - terminating the locked OS thread on Close() leaves
// Vulkan worker threads (e.g. Mesa llvmpipe's workers) racing process exit,
// which manifests as a SIGSEGV in native code with no Go traceback.
func (d *Device) gpuOwnerLoop() {
	if runtime.GOOS == "darwin" {
		runtime.LockOSThread()
	}
	for j := range d.jobs {
		j.done <- j.fn()
	}
}

// onGPU runs fn on the GPU-owner goroutine and blocks until it completes,
// returning fn's error. Must NOT be called from within a job already running on
// the owner goroutine (it would deadlock).
func (d *Device) onGPU(fn func() error) error {
	j := gpuJob{fn: fn, done: make(chan error, 1)}
	d.jobs <- j
	return <-j.done
}

// Open acquires a WebGPU adapter and device from the system.
// Returns an error with actionable guidance if no adapter is found.
//
// The adapter/device acquisition runs on the GPU-owner goroutine because it also
// drives Metal and creates autorelease pools.
func Open() (*Device, error) {
	if os.Getenv("ANNEAL_NO_GPU") == "1" {
		return nil, errors.New("webgpu: disabled via ANNEAL_NO_GPU=1")
	}
	d := &Device{jobs: make(chan gpuJob)}
	go d.gpuOwnerLoop()

	err := d.onGPU(func() error {
		inst, err := wgpu.CreateInstance(nil)
		if err != nil {
			return fmt.Errorf("webgpu: CreateInstance: %w - is a native GPU runtime available? Metal on macOS, Vulkan on Linux", err)
		}
		adapter, err := inst.RequestAdapter(nil)
		if err != nil {
			inst.Release()
			return fmt.Errorf("webgpu: no GPU adapter found: %w - run `anneal doctor` for hardware diagnostics", err)
		}
		var devDesc *wgpu.DeviceDescriptor
		if adapter.Features().Contains(gputypes.FeatureShaderF16) {
			var feats gputypes.Features
			feats.Insert(gputypes.FeatureShaderF16)
			devDesc = &wgpu.DeviceDescriptor{RequiredFeatures: feats}
			d.HasShaderF16 = true
		}
		dev, err := adapter.RequestDevice(devDesc)
		if err != nil {
			adapter.Release()
			inst.Release()
			return fmt.Errorf("webgpu: RequestDevice: %w", err)
		}
		d.instance = inst
		d.adapter = adapter
		d.device = dev
		d.queue = dev.Queue()
		return nil
	})
	if err != nil {
		close(d.jobs) // tear down the owner goroutine / its OS thread
		return nil, err
	}
	d.renderer = renderer{}
	d.compiler = newCompiler(d)
	return d, nil
}

// Close releases all GPU resources and terminates the GPU-owner goroutine.
func (d *Device) Close() {
	_ = d.onGPU(func() error {
		d.freeRealizeCacheLocked()
		if d.compiler != nil {
			d.compiler.releaseAll()
		}
		if d.device != nil {
			d.device.Release()
		}
		if d.adapter != nil {
			d.adapter.Release()
		}
		if d.instance != nil {
			d.instance.Release()
		}
		return nil
	})
	close(d.jobs) // owner goroutine exits → its locked OS thread is terminated
}

// AdapterName returns the GPU adapter name for diagnostics.
func (d *Device) AdapterName() string {
	if d.adapter == nil {
		return "<none>"
	}
	return d.adapter.Info().Name
}

// MaxStorageBufferBindingSize returns the device's maximum size in bytes for a
// single storage buffer binding. The WebGPU spec minimum is 128 MiB. Real
// hardware typically exposes much more (M3 reports several GiB); software
// adapters (Mesa llvmpipe in CI) tend to honour only the spec minimum. Tests
// that need a single buffer larger than 128 MiB (e.g. GPT-2 scale embeddings,
// roughly 147 MiB) should consult this and skip when the limit is too low,
// otherwise CreateBindGroup fails at realize time.
func (d *Device) MaxStorageBufferBindingSize() uint64 {
	if d.device == nil {
		return 0
	}
	return d.device.Limits().MaxStorageBufferBindingSize
}
