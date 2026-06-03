//go:build !js

// W9 — /api/device: the native binary's adapter / device probe surfaced as
// JSON. Spec: notes/anneal_web_spec.md §5.8.
//
// The doctor view is server-tier: the native binary knows its adapter, the
// browser does not have introspection into the goffi-driven WebGPU runtime.
// The studio's doctor page fetches /api/device for the native card and runs
// navigator.gpu.requestAdapter() in-page for the browser card. The two are
// independent enumerations on the same machine; the browser card carries a
// "diagnostic only" caveat per spec §5.8.
//
// Implementation note: opening the WebGPU device on every probe is expensive
// (and on darwin it pins an OS thread for the device's lifetime). The handler
// opens-then-closes a transient device for each request; this matches what
// `anneal doctor` already does on the CLI and keeps the server stateless. A
// test seam (deviceProbeFn) lets the wire contract tests run without a real
// GPU adapter.

package main

import (
	"encoding/json"
	"net/http"
	"runtime"

	"github.com/georgebuilds/anneal/backend/webgpu"
)

// DeviceInfo is the JSON payload returned by /api/device. Field shape is
// pinned by cmd_web_doctor_test.go (TestAPIDevice_RequiredFields). New fields
// added here MUST default to a JSON-friendly zero value so older studio
// builds still parse the response.
type DeviceInfo struct {
	AdapterName               string `json:"adapter_name"`
	Backend                   string `json:"backend"`
	OS                        string `json:"os"`
	Arch                      string `json:"arch"`
	AnnealVersion             string `json:"anneal_version"`
	ShaderF16                 bool   `json:"shader_f16"`
	MaxStorageBufferBindingSize uint64 `json:"max_storage_buffer_binding_size"`

	// Error is populated when the native binary cannot reach a GPU adapter
	// (e.g. headless CI without Mesa, missing Metal). The fields above are
	// still filled in with what we know (os/arch/version/backend); the studio
	// renders the error inline so the doctor page is never blank.
	Error string `json:"error,omitempty"`
}

// deviceProbe is the function signature for opening a WebGPU device and
// reading its diagnostic fields. Production wires probeWebGPU; tests can
// replace it with a no-GPU stub via withStubDeviceProbe.
type deviceProbe func() (adapterName string, shaderF16 bool, maxBuf uint64, err error)

var deviceProbeFn deviceProbe = probeWebGPU

// probeWebGPU opens a transient WebGPU device, reads its diagnostic fields,
// and closes it. Errors are surfaced verbatim through the JSON envelope.
func probeWebGPU() (string, bool, uint64, error) {
	// Lock to an OS thread for the open/close — webgpu.Open spins up a GPU
	// owner goroutine on darwin that holds an OS thread, but the locking
	// here protects callers that hop threads between Open and Close.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	dev, err := webgpu.Open()
	if err != nil {
		return "", false, 0, err
	}
	defer dev.Close()
	return dev.AdapterName(), dev.HasShaderF16, dev.MaxStorageBufferBindingSize(), nil
}

// handleAPIDevice writes the device info JSON. Returns 200 even on probe
// failure — the studio expects a parseable JSON body and renders the
// embedded error string inline rather than treating a missing adapter as
// an HTTP error.
func handleAPIDevice(w http.ResponseWriter, r *http.Request) {
	info := DeviceInfo{
		Backend:       detectBackend(),
		OS:            runtime.GOOS,
		Arch:          runtime.GOARCH,
		AnnealVersion: version,
	}
	name, f16, maxBuf, err := deviceProbeFn()
	if err != nil {
		info.Error = err.Error()
		info.AdapterName = "<unavailable>"
	} else {
		info.AdapterName = name
		info.ShaderF16 = f16
		info.MaxStorageBufferBindingSize = maxBuf
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(info)
}
