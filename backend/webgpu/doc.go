// Package webgpu implements the anneal backend using gogpu/wgpu — a pure-Go
// WebGPU implementation that uses goffi for Metal/Vulkan/DX12 HAL calls.
//
// Build requirement: CGO_ENABLED=0 (goffi's dynamic-linking mechanism
// requires CGO to be disabled; see github.com/go-webgpu/goffi#requirements).
//
// Usage:
//
//	dev, err := webgpu.Open()
//	if err != nil {
//	    log.Fatalf("no GPU: %v", err)
//	}
//	defer dev.Close()
//
//	outputs, err := dev.Run(items, inputs)
package webgpu
