// Tensor-inspect view data (W9). Pure-Go: dispatches a byte slice + format
// hint to the tensor/npy or tensor/safetensors byte readers and returns a
// JSON document the studio's home dropzone renders.
//
// DD2 / privacy property: bytes never leave the tab. The WASM annealInspectTensor
// bridge consumes a Uint8Array directly from the browser dropzone, calls
// BuildInspect, and returns a JSON string. The native server NEVER touches
// the payload (there is no /api/inspect endpoint).
//
// Spec: notes/anneal_web_spec.md §5.1.
package viz

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/georgebuilds/anneal/tensor/npy"
	"github.com/georgebuilds/anneal/tensor/safetensors"
)

// previewLen is how many leading float32 values BuildInspect surfaces per
// tensor. The dropzone shows them as a tiny preview ("first N values").
const previewLen = 16

// InspectResult is the top-level JSON payload returned by annealInspectTensor.
// Mirrors the contract in notes/anneal_web_spec.md §5.1.
type InspectResult struct {
	Format  string       `json:"format"`          // "npy" | "npz" | "safetensors"
	Tensors []TensorInfo `json:"tensors"`         // npz/safetensors have multiple; npy has one
	Error   string       `json:"error,omitempty"` // set on parse failure
}

// TensorInfo is one entry in InspectResult.Tensors. Preview holds up to
// previewLen leading float32 values (or fewer for tiny tensors / scalars).
// Bytes is the on-wire payload size of the encoded tensor (i.e. n*itemSize
// before any dtype conversion); it lets the user see "this U64 tensor was
// 12 MB on disk but is 6 MB after Float32 conversion".
type TensorInfo struct {
	Name    string    `json:"name"`
	Shape   []int64   `json:"shape"`
	DType   string    `json:"dtype"`
	Numel   int       `json:"numel"`
	Bytes   int       `json:"bytes"`
	Preview []float32 `json:"preview"`
}

// ToJSON serializes r to canonical JSON bytes.
func (r *InspectResult) ToJSON() ([]byte, error) { return json.Marshal(r) }

// BuildInspect dispatches on format ("npy"|"npz"|"safetensors") and returns
// shape, dtype, numel, byte size, and a 16-element preview for every tensor.
// Unknown formats return an InspectResult whose Error field is populated so
// the studio's renderer can surface the blameless error inline.
func BuildInspect(format string, payload []byte) *InspectResult {
	f := strings.ToLower(strings.TrimSpace(format))
	switch f {
	case "npy":
		entry, err := npy.ReadBytes(payload)
		if err != nil {
			return &InspectResult{Format: "npy", Error: err.Error()}
		}
		return &InspectResult{
			Format:  "npy",
			Tensors: []TensorInfo{npyTensorInfo("array", entry)},
		}
	case "npz":
		names, entries, err := npy.ReadZBytes(payload)
		if err != nil {
			return &InspectResult{Format: "npz", Error: err.Error()}
		}
		ts := make([]TensorInfo, 0, len(names))
		for _, n := range names {
			ts = append(ts, npyTensorInfo(n, entries[n]))
		}
		return &InspectResult{Format: "npz", Tensors: ts}
	case "safetensors":
		entries, err := safetensors.ReadBytes(payload)
		if err != nil {
			return &InspectResult{Format: "safetensors", Error: err.Error()}
		}
		ts := make([]TensorInfo, 0, len(entries))
		for _, e := range entries {
			ts = append(ts, safetensorsTensorInfo(e))
		}
		return &InspectResult{Format: "safetensors", Tensors: ts}
	default:
		return &InspectResult{
			Format: format,
			Error:  fmt.Sprintf("unknown format %q (expected npy, npz, or safetensors)", format),
		}
	}
}

func npyTensorInfo(name string, e npy.InspectEntry) TensorInfo {
	numel := 1
	for _, s := range e.Shape {
		numel *= int(s)
	}
	if numel < 0 {
		numel = 0
	}
	return TensorInfo{
		Name:    name,
		Shape:   e.Shape,
		DType:   e.DType,
		Numel:   numel,
		Bytes:   numel * npyItemSize(e.DType),
		Preview: preview(e.Data),
	}
}

func safetensorsTensorInfo(e safetensors.InspectEntry) TensorInfo {
	numel := 1
	for _, s := range e.Shape {
		numel *= int(s)
	}
	if numel < 0 {
		numel = 0
	}
	return TensorInfo{
		Name:    e.Name,
		Shape:   e.Shape,
		DType:   e.DType,
		Numel:   numel,
		Bytes:   numel * stItemSize(e.DType),
		Preview: preview(e.Data),
	}
}

// preview returns the leading slice of data, truncated to previewLen, as a
// freshly-allocated slice so the caller cannot mutate the parser's payload.
// Tiny tensors return their full content; an empty tensor returns an empty
// (not nil) slice so the JSON renders as `[]`.
func preview(data []float32) []float32 {
	n := previewLen
	if len(data) < n {
		n = len(data)
	}
	out := make([]float32, n)
	copy(out, data)
	return out
}

// npyItemSize returns bytes-per-element for a numpy dtype descriptor.
// Returns 0 for unknown descriptors (the Bytes field then reads 0 rather
// than a wrong value).
func npyItemSize(descr string) int {
	if descr == "" {
		return 0
	}
	// Strip the byte-order prefix if present.
	if descr[0] == '<' || descr[0] == '>' || descr[0] == '|' || descr[0] == '=' {
		descr = descr[1:]
	}
	if len(descr) < 2 {
		return 0
	}
	switch descr[0] {
	case 'f', 'i', 'u', 'b', 'c':
		var n int
		for j := 1; j < len(descr); j++ {
			if descr[j] < '0' || descr[j] > '9' {
				return 0
			}
			n = n*10 + int(descr[j]-'0')
		}
		return n
	}
	return 0
}

func stItemSize(dtype string) int {
	switch dtype {
	case "BOOL", "I8", "U8":
		return 1
	case "F16", "BF16", "I16", "U16":
		return 2
	case "F32", "I32", "U32":
		return 4
	case "F64", "I64", "U64":
		return 8
	}
	return 0
}
