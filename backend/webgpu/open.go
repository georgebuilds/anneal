package webgpu

import (
	"fmt"
	"runtime"

	"github.com/gogpu/wgpu"
)

// Device holds an open WebGPU device and its associated queue.
//
// ── Threading model (load-bearing — do not bypass) ───────────────────────────
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
// (e.g. Device.WaitIdle). An ObjC autorelease pool is thread-affine — it must be
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
	// symCache maps compiled WGSL source → handle for compile-once symbolic kernels.
	// Only touched on the GPU-owner goroutine, so it needs no synchronization.
	symCache map[string]*SymKernelHandle

	// jobs delivers closures to the GPU-owner goroutine. Closed by Close, which
	// terminates the owner goroutine (and thereby its locked OS thread).
	jobs chan gpuJob
}

// gpuJob is a unit of Metal-touching work handed to the GPU-owner goroutine.
type gpuJob struct {
	fn   func() error
	done chan error
}

// gpuOwnerLoop is the body of the single GPU-owner goroutine. It locks itself to
// one OS thread permanently and never unlocks: when the loop returns (jobs
// closed) while still locked, the Go runtime terminates the underlying OS thread,
// which is the desired teardown. Every job runs to completion on this one thread,
// so all NSAutoreleasePool create/drain pairs share it.
func (d *Device) gpuOwnerLoop() {
	runtime.LockOSThread()
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
	d := &Device{jobs: make(chan gpuJob)}
	go d.gpuOwnerLoop()

	err := d.onGPU(func() error {
		inst, err := wgpu.CreateInstance(nil)
		if err != nil {
			return fmt.Errorf("webgpu: CreateInstance: %w — is a native GPU runtime available? Metal on macOS, Vulkan on Linux", err)
		}
		adapter, err := inst.RequestAdapter(nil)
		if err != nil {
			inst.Release()
			return fmt.Errorf("webgpu: no GPU adapter found: %w — run `anneal doctor` for hardware diagnostics", err)
		}
		dev, err := adapter.RequestDevice(nil)
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
	d.symCache = make(map[string]*SymKernelHandle)
	return d, nil
}

// Close releases all GPU resources and terminates the GPU-owner goroutine.
func (d *Device) Close() {
	_ = d.onGPU(func() error {
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
