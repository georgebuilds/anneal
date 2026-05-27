// Package safetensors provides pure-Go save and load of safetensors checkpoints.
//
// # Naming convention
//
// Parameter.Name is never populated by the nn package (it is always ""),
// so this package uses an explicit map[string]*nn.Parameter as its primary
// API — the caller owns the name mapping:
//
//	params := map[string]*nn.Parameter{
//	    "l1.weight": linear1.Weight,
//	    "l1.bias":   linear1.Bias,
//	    "l2.weight": linear2.Weight,
//	}
//	safetensors.Save("checkpoint.safetensors", params)
//	safetensors.Load("checkpoint.safetensors", params)
//
// This is design option (a): caller owns names, no silent reliance on empty .Name.
// A positional-index approach would be fragile across layer additions; explicit
// names travel safely across model refactors.
//
// # Dtype policy (mirrors tensor/npy)
//
// Files are always written as F32 (anneal's native float carry type — lossless).
// On load, the following conversions are applied:
//
//   - F32                      → float32, bit-exact.
//   - F64                      → float32 via float32(); precision loss possible.
//   - F16 (IEEE-754 half)      → float32; see f16ToF32; precision loss possible.
//   - BF16 (brain float16)     → float32 by zero-filling the low mantissa bits.
//   - I8 / I16                 → float32 by integer value; always exact.
//   - I32 / I64                → float32 by value; error if |v| > 2^24.
//   - U8 / U16                 → float32 by integer value; always exact.
//   - U32 / U64                → float32 by value; error if v > 2^24.
//   - BOOL                     → 0.0 (false) or 1.0 (true).
//   - F8_E4M3, F8_E5M2, …      → error naming the dtype.
//
// # File format
//
// safetensors layout: 8-byte little-endian u64 header length N, N bytes of
// UTF-8 JSON describing each tensor (dtype, shape, data_offsets into the
// data section), then raw tensor bytes (all little-endian). A optional
// "__metadata__" entry (string→string map) is silently ignored on load.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/georgebuilds/anneal/tensor/nn"
)

// maxExactInt is the largest integer (absolute value) exactly representable as
// float32 (2^24 = 16,777,216 — 23 explicit + 1 implicit mantissa bit).
// Mirrors the constant in tensor/npy for consistent behaviour.
const maxExactInt = int64(1 << 24)

// Entry is a decoded tensor from a safetensors file.
type Entry struct {
	Data  []float32
	Shape []int64
}

// ── Save ─────────────────────────────────────────────────────────────────────

// Save writes the named Parameters to a safetensors checkpoint at path.
// Each parameter's Value slice is saved as F32. The map keys become tensor
// names in the file; Parameter.Name is not used.
//
// Tensor names are written in lexicographic order for deterministic output.
func Save(path string, params map[string]*nn.Parameter) error {
	names := sortedKeys(params)

	// Compute per-tensor byte sizes and cumulative data offsets (F32 = 4 bytes).
	type tensorMeta struct {
		DType       string   `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	hdrMap := make(map[string]tensorMeta, len(params))
	var off int64
	for _, name := range names {
		p := params[name]
		sh := p.T.Shape()
		n := int64(1)
		for _, s := range sh {
			n *= s
		}
		byteLen := n * 4
		hdrMap[name] = tensorMeta{
			DType:       "F32",
			Shape:       sh,
			DataOffsets: [2]int64{off, off + byteLen},
		}
		off += byteLen
	}

	// Encode and pad JSON header to a multiple of 8 bytes (safetensors convention).
	hdrJSON, err := json.Marshal(hdrMap)
	if err != nil {
		return fmt.Errorf("safetensors: encode header: %w", err)
	}
	if rem := len(hdrJSON) % 8; rem != 0 {
		pad := make([]byte, 8-rem)
		for i := range pad {
			pad[i] = ' '
		}
		hdrJSON = append(hdrJSON, pad...)
	}

	// Build raw data buffer: F32 little-endian, tensors in name order.
	var totalElems int
	for _, name := range names {
		totalElems += len(params[name].Value)
	}
	dataBuf := make([]byte, totalElems*4)
	pos := 0
	for _, name := range names {
		for _, v := range params[name].Value {
			binary.LittleEndian.PutUint32(dataBuf[pos:], math.Float32bits(v))
			pos += 4
		}
	}

	// Write: [u64 header_len] [padded header JSON] [data buffer].
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("safetensors: create %q: %w", path, err)
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(hdrJSON))); err != nil {
		return fmt.Errorf("safetensors: write header length: %w", err)
	}
	if _, err := f.Write(hdrJSON); err != nil {
		return fmt.Errorf("safetensors: write header: %w", err)
	}
	if _, err := f.Write(dataBuf); err != nil {
		return fmt.Errorf("safetensors: write data: %w", err)
	}
	return nil
}

// ── Load ─────────────────────────────────────────────────────────────────────

// Load reads a safetensors checkpoint and restores the Value slice of each
// Parameter in into. For every key in into the file must contain a tensor
// with the same name and shape — a mismatch returns a clear error.
//
// After Load, call p.Load(arena) to seed a fresh leaf tensor from the
// restored Value before the next forward pass.
func Load(path string, into map[string]*nn.Parameter) error {
	tensors, err := LoadTensors(path)
	if err != nil {
		return err
	}
	for name, p := range into {
		entry, ok := tensors[name]
		if !ok {
			return fmt.Errorf("safetensors: tensor %q not found in %q", name, path)
		}
		pShape := p.T.Shape()
		if !shapesEqual(pShape, entry.Shape) {
			return fmt.Errorf("safetensors: tensor %q shape mismatch: file has %v, parameter expects %v",
				name, entry.Shape, pShape)
		}
		copy(p.Value, entry.Data)
	}
	return nil
}

// LoadTensors parses a safetensors file and returns all tensors as float32
// data + shape. Useful for loading weights into non-Parameter contexts (e.g.
// inspecting a HuggingFace checkpoint).
//
// The "__metadata__" header entry is silently skipped.
func LoadTensors(path string) (map[string]Entry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %q: %w", path, err)
	}
	defer f.Close()

	// 8-byte little-endian u64: header byte length.
	var hLen uint64
	if err := binary.Read(f, binary.LittleEndian, &hLen); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}

	hdrBytes := make([]byte, hLen)
	if _, err := io.ReadFull(f, hdrBytes); err != nil {
		return nil, fmt.Errorf("safetensors: read header (%d bytes): %w", hLen, err)
	}

	// Data buffer follows the header immediately.
	dataBuf, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read data: %w", err)
	}

	// Parse JSON using raw messages so "__metadata__" (different shape) is handled.
	var rawHdr map[string]json.RawMessage
	if err := json.Unmarshal(hdrBytes, &rawHdr); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}

	type tensorEntry struct {
		DType       string   `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	out := make(map[string]Entry, len(rawHdr))
	for name, raw := range rawHdr {
		if name == "__metadata__" {
			continue
		}
		var te tensorEntry
		if err := json.Unmarshal(raw, &te); err != nil {
			return nil, fmt.Errorf("safetensors: parse entry %q: %w", name, err)
		}
		start, end := te.DataOffsets[0], te.DataOffsets[1]
		if start < 0 || end > int64(len(dataBuf)) || start > end {
			return nil, fmt.Errorf("safetensors: tensor %q data_offsets [%d,%d] out of range (data buffer is %d bytes)",
				name, start, end, len(dataBuf))
		}
		f32, err := decodeToFloat32(dataBuf[start:end], te.DType, te.Shape, name)
		if err != nil {
			return nil, err
		}
		out[name] = Entry{Data: f32, Shape: te.Shape}
	}
	return out, nil
}

// ── dtype decoding ────────────────────────────────────────────────────────────

// decodeToFloat32 converts a raw safetensors byte slice to []float32.
// Applies the same dtype policy as tensor/npy.
func decodeToFloat32(raw []byte, dtype string, shape []int64, name string) ([]float32, error) {
	nElems := 1
	for _, s := range shape {
		nElems *= int(s)
	}

	iSize, err := dtypeItemSize(dtype)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}
	if len(raw) != nElems*iSize {
		return nil, fmt.Errorf("safetensors: tensor %q: dtype %s shape %v expects %d bytes, got %d",
			name, dtype, shape, nElems*iSize, len(raw))
	}

	out := make([]float32, nElems)
	switch dtype {

	case "F32":
		for i := 0; i < nElems; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}

	case "F64": // float64 → float32: precision loss is documented and expected
		for i := 0; i < nElems; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}

	case "F16":
		for i := 0; i < nElems; i++ {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}

	case "BF16":
		for i := 0; i < nElems; i++ {
			out[i] = bf16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}

	case "I8": // [-128, 127]: always fits in float32
		for i := 0; i < nElems; i++ {
			out[i] = float32(int8(raw[i]))
		}

	case "I16": // [-32768, 32767]: always fits in float32
		for i := 0; i < nElems; i++ {
			out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:])))
		}

	case "I32":
		for i := 0; i < nElems; i++ {
			v := int64(int32(binary.LittleEndian.Uint32(raw[i*4:])))
			if v > maxExactInt || v < -maxExactInt {
				return nil, fmt.Errorf("safetensors: tensor %q: I32 value %d at index %d is outside float32 exact integer range [%d, %d]; use a narrower dtype or filter values in Python before loading",
					name, v, i, -maxExactInt, maxExactInt)
			}
			out[i] = float32(v)
		}

	case "I64":
		for i := 0; i < nElems; i++ {
			v := int64(binary.LittleEndian.Uint64(raw[i*8:]))
			if v > maxExactInt || v < -maxExactInt {
				return nil, fmt.Errorf("safetensors: tensor %q: I64 value %d at index %d is outside float32 exact integer range [%d, %d]; use a narrower dtype or filter values in Python before loading",
					name, v, i, -maxExactInt, maxExactInt)
			}
			out[i] = float32(v)
		}

	case "U8": // [0, 255]: always fits in float32
		for i := 0; i < nElems; i++ {
			out[i] = float32(raw[i])
		}

	case "U16": // [0, 65535]: always fits in float32
		for i := 0; i < nElems; i++ {
			out[i] = float32(binary.LittleEndian.Uint16(raw[i*2:]))
		}

	case "U32":
		for i := 0; i < nElems; i++ {
			v := uint64(binary.LittleEndian.Uint32(raw[i*4:]))
			if v > uint64(maxExactInt) {
				return nil, fmt.Errorf("safetensors: tensor %q: U32 value %d at index %d exceeds float32 exact integer range [0, %d]; use a narrower dtype or filter values in Python before loading",
					name, v, i, maxExactInt)
			}
			out[i] = float32(v)
		}

	case "U64":
		for i := 0; i < nElems; i++ {
			v := binary.LittleEndian.Uint64(raw[i*8:])
			if v > uint64(maxExactInt) {
				return nil, fmt.Errorf("safetensors: tensor %q: U64 value %d at index %d exceeds float32 exact integer range [0, %d]; use a narrower dtype or filter values in Python before loading",
					name, v, i, maxExactInt)
			}
			out[i] = float32(v)
		}

	case "BOOL": // 1 byte per element: 0 = false → 0.0, nonzero = true → 1.0
		for i := 0; i < nElems; i++ {
			if raw[i] != 0 {
				out[i] = 1.0
			}
		}

	default:
		return nil, fmt.Errorf("safetensors: tensor %q: dtype %q is not supported; supported: F32, F64, F16, BF16, I8, I16, I32, I64, U8, U16, U32, U64, BOOL",
			name, dtype)
	}
	return out, nil
}

// dtypeItemSize returns the bytes per element for a safetensors dtype string.
func dtypeItemSize(dtype string) (int, error) {
	switch dtype {
	case "BOOL", "I8", "U8":
		return 1, nil
	case "F16", "BF16", "I16", "U16":
		return 2, nil
	case "F32", "I32", "U32":
		return 4, nil
	case "F64", "I64", "U64":
		return 8, nil
	default:
		return 0, fmt.Errorf("dtype %q is not supported; supported: F32, F64, F16, BF16, I8, I16, I32, I64, U8, U16, U32, U64, BOOL", dtype)
	}
}

// ── float16 / bfloat16 conversion ─────────────────────────────────────────────

// f16ToF32 converts an IEEE-754 float16 bit pattern to the nearest float32.
// Handles zero, subnormals (denormals), infinities, and NaN correctly.
func f16ToF32(bits uint16) float32 {
	sign := uint32(bits>>15) << 31
	exp16 := int32((bits >> 10) & 0x1f)
	mant := uint32(bits&0x3ff) << 13

	var result uint32
	switch exp16 {
	case 0: // zero or subnormal
		if mant == 0 {
			result = sign // ±0
		} else {
			// Normalise the subnormal: shift until the implicit leading bit is set.
			e := uint32(127 - 14) // F32 bias 127, F16 subnormal effective exp = 1-15 = -14
			m := mant
			for m&0x800000 == 0 {
				m <<= 1
				e--
			}
			result = sign | (e << 23) | (m & 0x7fffff)
		}
	case 0x1f: // infinity or NaN
		result = sign | 0x7f800000 | mant
	default: // normal value
		exp32 := uint32(exp16 - 15 + 127)
		result = sign | (exp32 << 23) | mant
	}
	return math.Float32frombits(result)
}

// bf16ToF32 converts a bfloat16 bit pattern to float32.
// BF16 is the high 16 bits of a float32 IEEE-754 encoding; the low 16 mantissa
// bits are simply zero-filled, making the conversion lossless in the other
// direction (float32 → BF16 is truncation, not rounding).
func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// ── helpers ───────────────────────────────────────────────────────────────────

func shapesEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sortedKeys(m map[string]*nn.Parameter) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
