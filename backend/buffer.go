package backend

import "github.com/georgebuilds/anneal/uop"

// DeviceBuffer is the backend-agnostic handle for one GPU buffer.
//
// Threading: every method that touches native GPU state (Write, Read, Release)
// must be called from the backend's GPU-owner goroutine — on backends where one
// exists (the WebGPU/Metal implementation pins all native calls to a single
// runtime.LockOSThread'd goroutine, see backend/webgpu/open.go). Calling these
// methods from anywhere else risks the same NSAutoreleasePool migration crash
// that the owner-goroutine pattern was introduced to fix.
//
// Read returns the buffer contents as raw little-endian bytes; the caller
// decodes per DType().
//
// The name DeviceBuffer (rather than just Buffer) avoids collision with
// schedule.Buffer, which is the static schedule-side descriptor of a buffer
// slot rather than a runtime GPU handle.
type DeviceBuffer interface {
	Size() int64
	DType() *uop.DType
	Write(data []byte) error
	Read() ([]byte, error)
	Release()
}
