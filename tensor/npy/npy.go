// Package npy provides pure-Go loading of NumPy .npy and .npz files into
// anneal Tensors.
//
// All output tensors carry dtype Float32. Supported dtype conversions:
//
//   - numpy float32  → Float32, bit-exact.
//   - numpy float64  → Float32; downcasts via float32() (IEEE-754 round-to-nearest-even).
//     Precision loss is possible and expected for values not representable as float32.
//   - numpy int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64
//     → Float32 by integer value. float32 exactly represents all integers whose
//     absolute value is ≤ 2^24 = 16,777,216. Values outside that range return
//     an error instead of silently dropping low-order bits.
//   - numpy bool (|b1) → Float32: false → 0.0, true → 1.0.
//   - All other dtypes (complex, float16, structured, …) → a descriptive error.
//
// Fortran-order (column-major) arrays are transposed to C (row-major) order.
// Parse is pure Go — no cgo, no runtime Python.
package npy

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// maxExactInt is the largest integer (by absolute value) exactly representable
// as float32. float32 has 23 explicit mantissa bits + 1 implicit = 24 bits of
// integer precision → 2^24 = 16,777,216.
const maxExactInt = int64(1 << 24)

// Load loads a .npy file at path and returns a Float32 Tensor on device.
// See the package doc for dtype conversion rules and error conditions.
func Load(a *uop.Arena, path string, device string) (*tensor.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("npy: read %q: %w", path, err)
	}
	return parseNPY(a, data, device)
}

// LoadNPZ loads a .npz file (ZIP of named .npy entries) and returns a map of
// array name → Tensor. The same dtype rules as Load apply to each entry.
func LoadNPZ(a *uop.Arena, path string, device string) (map[string]*tensor.Tensor, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("npy: open npz %q: %w", path, err)
	}
	defer r.Close()

	out := make(map[string]*tensor.Tensor, len(r.File))
	for _, f := range r.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("npy: open npz entry %q: %w", f.Name, err)
		}
		data, readErr := io.ReadAll(rc)
		rc.Close()
		if readErr != nil {
			return nil, fmt.Errorf("npy: read npz entry %q: %w", f.Name, readErr)
		}
		t, err := parseNPY(a, data, device)
		if err != nil {
			return nil, fmt.Errorf("npy: parse npz entry %q: %w", f.Name, err)
		}
		out[name] = t
	}
	return out, nil
}

// ── internal parser ───────────────────────────────────────────────────────────

func parseNPY(a *uop.Arena, data []byte, device string) (*tensor.Tensor, error) {
	if len(data) < 10 {
		return nil, fmt.Errorf("npy: file too short (%d bytes)", len(data))
	}
	// Magic: 0x93 'N' 'U' 'M' 'P' 'Y'
	if data[0] != 0x93 || data[1] != 'N' || data[2] != 'U' ||
		data[3] != 'M' || data[4] != 'P' || data[5] != 'Y' {
		return nil, fmt.Errorf("npy: not a NumPy file (bad magic bytes)")
	}
	major := int(data[6])

	var headerLen, dataOffset int
	switch major {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
		dataOffset = 10 + headerLen
	case 2, 3:
		if len(data) < 12 {
			return nil, fmt.Errorf("npy: file too short for v%d.x header length field", major)
		}
		headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
		dataOffset = 12 + headerLen
	default:
		return nil, fmt.Errorf("npy: unsupported format version %d.%d", major, int(data[7]))
	}

	if dataOffset > len(data) {
		return nil, fmt.Errorf("npy: declared header length %d exceeds file length", headerLen)
	}
	// Header string: trim trailing spaces and newline used for alignment padding.
	hdrBytes := data[dataOffset-headerLen : dataOffset]
	hdr := strings.TrimRight(string(hdrBytes), " \n")

	descr, fortranOrder, shape, err := parseHeader(hdr)
	if err != nil {
		return nil, fmt.Errorf("npy: header parse: %w", err)
	}

	byteOrder, typeChar, itemSize, err := parseDescr(descr)
	if err != nil {
		return nil, err
	}

	nElems := 1
	for _, s := range shape {
		nElems *= int(s)
	}

	expectedBytes := nElems * itemSize
	if len(data)-dataOffset < expectedBytes {
		return nil, fmt.Errorf("npy: data payload truncated: need %d bytes, have %d",
			expectedBytes, len(data)-dataOffset)
	}
	raw := data[dataOffset : dataOffset+expectedBytes]

	f32, err := convertToFloat32(raw, byteOrder, typeChar, itemSize, nElems)
	if err != nil {
		return nil, err
	}

	if fortranOrder && len(shape) > 1 {
		f32 = fortranToC(f32, shape)
	}

	t := tensor.NewLeaf(a, shape, uop.Dtypes.Float32, device)
	t.SetData(f32)
	return t, nil
}

// ── header parsing ────────────────────────────────────────────────────────────

func parseHeader(hdr string) (descr string, fortranOrder bool, shape []int64, err error) {
	descr, err = hdrStringField(hdr, "descr")
	if err != nil {
		return
	}
	fortranOrder, err = hdrBoolField(hdr, "fortran_order")
	if err != nil {
		return
	}
	shape, err = hdrShapeField(hdr)
	return
}

// hdrStringField extracts the value of a string-valued key from a Python dict
// literal, e.g. {'descr': '<f4', ...} → returns "<f4" for key "descr".
func hdrStringField(hdr, key string) (string, error) {
	needle := "'" + key + "'"
	idx := strings.Index(hdr, needle)
	if idx < 0 {
		return "", fmt.Errorf("key %q not found", key)
	}
	rest := strings.TrimSpace(hdr[idx+len(needle):])
	if !strings.HasPrefix(rest, ":") {
		return "", fmt.Errorf("expected ':' after key %q", key)
	}
	rest = strings.TrimSpace(rest[1:])
	if len(rest) == 0 || rest[0] != '\'' {
		return "", fmt.Errorf("expected quoted string value for key %q", key)
	}
	end := strings.Index(rest[1:], "'")
	if end < 0 {
		return "", fmt.Errorf("unterminated string value for key %q", key)
	}
	return rest[1 : end+1], nil
}

// hdrBoolField extracts a boolean field (True/False) from a Python dict literal.
func hdrBoolField(hdr, key string) (bool, error) {
	needle := "'" + key + "'"
	idx := strings.Index(hdr, needle)
	if idx < 0 {
		return false, fmt.Errorf("key %q not found", key)
	}
	rest := strings.TrimSpace(hdr[idx+len(needle):])
	if !strings.HasPrefix(rest, ":") {
		return false, fmt.Errorf("expected ':' after key %q", key)
	}
	rest = strings.TrimSpace(rest[1:])
	switch {
	case strings.HasPrefix(rest, "True"):
		return true, nil
	case strings.HasPrefix(rest, "False"):
		return false, nil
	default:
		snip := rest
		if len(snip) > 20 {
			snip = snip[:20]
		}
		return false, fmt.Errorf("expected True or False for key %q, got %q", key, snip)
	}
}

// hdrShapeField extracts the shape tuple from a Python dict literal.
// Returns []int64{} for scalar arrays (shape = ()).
func hdrShapeField(hdr string) ([]int64, error) {
	const needle = "'shape'"
	idx := strings.Index(hdr, needle)
	if idx < 0 {
		return nil, fmt.Errorf("key 'shape' not found")
	}
	rest := strings.TrimSpace(hdr[idx+len(needle):])
	if !strings.HasPrefix(rest, ":") {
		return nil, fmt.Errorf("expected ':' after 'shape'")
	}
	rest = strings.TrimSpace(rest[1:])
	if !strings.HasPrefix(rest, "(") {
		return nil, fmt.Errorf("expected '(' for shape tuple, got %q", rest[:min(10, len(rest))])
	}
	end := strings.Index(rest, ")")
	if end < 0 {
		return nil, fmt.Errorf("unterminated shape tuple")
	}
	inner := strings.TrimSpace(rest[1:end])
	if inner == "" {
		return []int64{}, nil // scalar: shape ()
	}
	var shape []int64
	for _, tok := range strings.Split(inner, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue // trailing comma in single-element tuple: (5,)
		}
		v, err := strconv.ParseInt(tok, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid shape dimension %q: %w", tok, err)
		}
		shape = append(shape, v)
	}
	return shape, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ── dtype descriptor ──────────────────────────────────────────────────────────

// parseDescr parses a numpy dtype descriptor string such as '<f4', '|b1', '>i8', '=u2'.
// Returns (byteOrder, typeChar, itemSizeBytes, error).
// byteOrder: '<' = little-endian, '>' = big-endian, '|' or '=' = not-applicable/native.
func parseDescr(descr string) (byteOrder byte, typeChar byte, itemSize int, err error) {
	if len(descr) == 0 {
		err = fmt.Errorf("npy: empty dtype descriptor")
		return
	}
	switch descr[0] {
	case '<', '>', '|', '=':
		byteOrder = descr[0]
		descr = descr[1:]
	default:
		byteOrder = '|'
	}
	if len(descr) < 2 {
		err = fmt.Errorf("npy: dtype descriptor too short (after stripping byte-order prefix): %q", descr)
		return
	}
	typeChar = descr[0]
	sz, parseErr := strconv.Atoi(descr[1:])
	if parseErr != nil {
		err = fmt.Errorf("npy: invalid item size in dtype descriptor: %q", descr)
		return
	}
	itemSize = sz
	return
}

// ── data conversion ───────────────────────────────────────────────────────────

func convertToFloat32(raw []byte, byteOrder, typeChar byte, itemSize, n int) ([]float32, error) {
	// All byte orders other than big-endian are treated as little-endian.
	// '=' (native) is little-endian on all platforms we target (x86, ARM).
	// '|' (not applicable) is used for 1-byte types where order is irrelevant.
	isLE := byteOrder != '>'

	u16 := func(i int) uint16 {
		b := raw[i*2:]
		if isLE {
			return binary.LittleEndian.Uint16(b)
		}
		return binary.BigEndian.Uint16(b)
	}
	u32 := func(i int) uint32 {
		b := raw[i*4:]
		if isLE {
			return binary.LittleEndian.Uint32(b)
		}
		return binary.BigEndian.Uint32(b)
	}
	u64 := func(i int) uint64 {
		b := raw[i*8:]
		if isLE {
			return binary.LittleEndian.Uint64(b)
		}
		return binary.BigEndian.Uint64(b)
	}

	out := make([]float32, n)
	switch typeChar {

	case 'f': // floating-point
		switch itemSize {
		case 4: // float32 → exact bit-for-bit copy
			for i := 0; i < n; i++ {
				out[i] = math.Float32frombits(u32(i))
			}
		case 8: // float64 → float32: precision loss is documented and expected
			for i := 0; i < n; i++ {
				out[i] = float32(math.Float64frombits(u64(i)))
			}
		default:
			return nil, fmt.Errorf("npy: unsupported float dtype float%d (only float32 and float64 are supported)", itemSize*8)
		}

	case 'i': // signed integer
		switch itemSize {
		case 1: // int8: range [-128, 127]; all values fit exactly in float32
			for i := 0; i < n; i++ {
				out[i] = float32(int8(raw[i]))
			}
		case 2: // int16: range [-32768, 32767]; all values fit exactly in float32
			for i := 0; i < n; i++ {
				out[i] = float32(int16(u16(i)))
			}
		case 4: // int32: values outside ±2^24 cannot be represented exactly
			for i := 0; i < n; i++ {
				v := int64(int32(u32(i)))
				if v > maxExactInt || v < -maxExactInt {
					return nil, fmt.Errorf("npy: int32 value %d at index %d is outside float32 exact integer range [%d, %d]; use a smaller dtype or filter the array in numpy first",
						v, i, -maxExactInt, maxExactInt)
				}
				out[i] = float32(v)
			}
		case 8: // int64: values outside ±2^24 cannot be represented exactly
			for i := 0; i < n; i++ {
				v := int64(u64(i))
				if v > maxExactInt || v < -maxExactInt {
					return nil, fmt.Errorf("npy: int64 value %d at index %d is outside float32 exact integer range [%d, %d]; use a smaller dtype or filter the array in numpy first",
						v, i, -maxExactInt, maxExactInt)
				}
				out[i] = float32(v)
			}
		default:
			return nil, fmt.Errorf("npy: unsupported signed integer dtype int%d", itemSize*8)
		}

	case 'u': // unsigned integer
		switch itemSize {
		case 1: // uint8: range [0, 255]; all values fit exactly in float32
			for i := 0; i < n; i++ {
				out[i] = float32(raw[i])
			}
		case 2: // uint16: range [0, 65535]; all values fit exactly in float32
			for i := 0; i < n; i++ {
				out[i] = float32(u16(i))
			}
		case 4: // uint32: values > 2^24 cannot be represented exactly
			for i := 0; i < n; i++ {
				v := u32(i)
				if int64(v) > maxExactInt {
					return nil, fmt.Errorf("npy: uint32 value %d at index %d exceeds float32 exact integer range [0, %d]; use a smaller dtype or filter the array in numpy first",
						v, i, maxExactInt)
				}
				out[i] = float32(v)
			}
		case 8: // uint64: values > 2^24 cannot be represented exactly
			for i := 0; i < n; i++ {
				v := u64(i)
				if v > uint64(maxExactInt) {
					return nil, fmt.Errorf("npy: uint64 value %d at index %d exceeds float32 exact integer range [0, %d]; use a smaller dtype or filter the array in numpy first",
						v, i, maxExactInt)
				}
				out[i] = float32(v)
			}
		default:
			return nil, fmt.Errorf("npy: unsupported unsigned integer dtype uint%d", itemSize*8)
		}

	case 'b': // bool (numpy dtype string |b1)
		if itemSize != 1 {
			return nil, fmt.Errorf("npy: unexpected bool element size %d (expected 1)", itemSize)
		}
		for i := 0; i < n; i++ {
			if raw[i] != 0 {
				out[i] = 1.0
			}
			// zero slots already initialised to 0.0 by make()
		}

	case 'c': // complex
		return nil, fmt.Errorf("npy: complex%d dtype is not supported; extract a real component in numpy before loading (e.g. arr.real)", itemSize*8)

	default:
		return nil, fmt.Errorf("npy: unsupported dtype kind %q (%d bytes/element); supported: float32/64, int8/16/32/64, uint8/16/32/64, bool",
			string(typeChar), itemSize)
	}
	return out, nil
}

// ── Fortran-order → C-order transposition ─────────────────────────────────────

// fortranToC converts a flat slice stored in Fortran (column-major) order to
// C (row-major) order. shape must match len(src). 0-D and 1-D inputs are
// returned unchanged (both orders are identical).
func fortranToC(src []float32, shape []int64) []float32 {
	rank := len(shape)
	if rank <= 1 {
		return src
	}

	n := len(src)
	dst := make([]float32, n)

	// fStrides[d] = stride of dimension d in Fortran (column-major) layout.
	// The first dimension has stride 1; each subsequent stride multiplies by
	// the size of the previous dimension.
	fStrides := make([]int64, rank)
	fStrides[0] = 1
	for d := 1; d < rank; d++ {
		fStrides[d] = fStrides[d-1] * shape[d-1]
	}

	// Walk all multi-indices in C order (last dimension varies fastest).
	// The C-order flat index is simply the outer loop counter.
	idx := make([]int64, rank)
	for cFlat := 0; cFlat < n; cFlat++ {
		var fFlat int64
		for d := 0; d < rank; d++ {
			fFlat += idx[d] * fStrides[d]
		}
		dst[cFlat] = src[fFlat]

		// Advance multi-index in C order.
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return dst
}
