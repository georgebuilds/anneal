// Package backend defines the device abstraction and hosts backend implementations.
// WebGPU (via gogpu/wgpu zero-CGO dynamic linking) is the v1 target.
package backend

import "github.com/georgebuilds/anneal/schedule"

// Executor executes a compiled schedule on a device.
// inputs maps Buffer.UOpIdx → flat float32 data for leaf (input) buffers.
// Returns output data keyed by Buffer.UOpIdx for final output buffers.
type Executor interface {
	Run(items []schedule.ExecItem, inputs map[uint32][]float32) (map[uint32][]float32, error)
	Close()
}

// SymbolicExecutor is an optional interface implemented by backends that support
// runtime-bound symbolic kernels. Kernels with symbolic dims are compiled once
// and dispatched multiple times with different concrete bindings.
// binding maps DefineVar name → concrete int64 value for this dispatch.
type SymbolicExecutor interface {
	RunSymbolic(items []schedule.ExecItem, inputs map[uint32][]float32, binding map[string]int64) (map[uint32][]float32, error)
}
