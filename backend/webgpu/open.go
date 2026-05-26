package webgpu

import (
	"fmt"

	"github.com/gogpu/wgpu"
)

// Device holds an open WebGPU device and its associated queue.
type Device struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
}

// Open acquires a WebGPU adapter and device from the system.
// Returns an error with actionable guidance if no adapter is found.
func Open() (*Device, error) {
	inst, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: CreateInstance: %w — is a native GPU runtime available? Metal on macOS, Vulkan on Linux", err)
	}
	adapter, err := inst.RequestAdapter(nil)
	if err != nil {
		inst.Release()
		return nil, fmt.Errorf("webgpu: no GPU adapter found: %w — run `anneal doctor` for hardware diagnostics", err)
	}
	dev, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("webgpu: RequestDevice: %w", err)
	}
	return &Device{
		instance: inst,
		adapter:  adapter,
		device:   dev,
		queue:    dev.Queue(),
	}, nil
}

// Close releases all GPU resources.
func (d *Device) Close() {
	if d.device != nil {
		d.device.Release()
	}
	if d.adapter != nil {
		d.adapter.Release()
	}
	if d.instance != nil {
		d.instance.Release()
	}
}

// AdapterName returns the GPU adapter name for diagnostics.
func (d *Device) AdapterName() string {
	if d.adapter == nil {
		return "<none>"
	}
	return d.adapter.Info().Name
}
