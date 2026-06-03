// bytes.go — byte-based inspection APIs for the WASM tensor-inspect view (W9).
//
// The Load / LoadNPZ entry points read from disk and return Float32 Tensors;
// they erase the original dtype string. The inspect dropzone needs:
//   - the raw dtype label so the studio can show "F64" or "<i4" instead of
//     always saying "Float32",
//   - byte-level input (the studio holds the file content in a Uint8Array),
//   - no Arena dependency (the WASM inspector does not allocate a tensor).
//
// ReadBytes / ReadZBytes mirror parseNPY / LoadNPZ but skip the Tensor
// construction and surface the original dtype descriptor verbatim. They share
// the same internal parsing helpers (parseHeader, parseDescr, convertToFloat32,
// fortranToC) so a fix in those functions reaches both the inspector and the
// existing arena-backed loaders.
package npy

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"strings"
)

// InspectEntry is one decoded array surfaced by ReadBytes / ReadZBytes.
//
// DType is the raw numpy dtype descriptor ("<f4", "<i8", "|b1", …) — what the
// .npy header literally carries. Shape is the row-major dimensions (Fortran
// arrays are transposed in Data). Data is the same Float32 payload Load
// returns, dequantized through the documented dtype policy.
type InspectEntry struct {
	DType string
	Shape []int64
	Data  []float32
}

// ReadBytes parses a .npy file from an in-memory byte slice and returns the
// raw dtype descriptor, shape, and dequantized float32 payload. No Arena, no
// Tensor.
func ReadBytes(data []byte) (InspectEntry, error) {
	descr, shape, f32, err := parseNPYBytes(data)
	if err != nil {
		return InspectEntry{}, err
	}
	return InspectEntry{DType: descr, Shape: shape, Data: f32}, nil
}

// ReadZBytes parses a .npz archive from an in-memory byte slice and returns
// one InspectEntry per stored array. Entry names retain insertion order via
// the returned name slice (npz uses ZIP storage, which preserves order).
func ReadZBytes(data []byte) (names []string, entries map[string]InspectEntry, err error) {
	r, zerr := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if zerr != nil {
		return nil, nil, fmt.Errorf("npy: open npz bytes: %w", zerr)
	}
	entries = make(map[string]InspectEntry, len(r.File))
	names = make([]string, 0, len(r.File))
	for _, f := range r.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, oerr := f.Open()
		if oerr != nil {
			return nil, nil, fmt.Errorf("npy: open npz entry %q: %w", f.Name, oerr)
		}
		raw, rerr := io.ReadAll(rc)
		_ = rc.Close()
		if rerr != nil {
			return nil, nil, fmt.Errorf("npy: read npz entry %q: %w", f.Name, rerr)
		}
		descr, shape, f32, perr := parseNPYBytes(raw)
		if perr != nil {
			return nil, nil, fmt.Errorf("npy: parse npz entry %q: %w", f.Name, perr)
		}
		names = append(names, name)
		entries[name] = InspectEntry{DType: descr, Shape: shape, Data: f32}
	}
	return names, entries, nil
}

// parseNPYBytes is the byte-only path through the existing parser. It re-uses
// parseHeader / parseDescr / convertToFloat32 / fortranToC so the dtype policy
// stays single-sourced.
func parseNPYBytes(data []byte) (descr string, shape []int64, f32 []float32, err error) {
	if len(data) < 10 {
		err = fmt.Errorf("npy: file too short (%d bytes)", len(data))
		return
	}
	if data[0] != 0x93 || data[1] != 'N' || data[2] != 'U' ||
		data[3] != 'M' || data[4] != 'P' || data[5] != 'Y' {
		err = fmt.Errorf("npy: not a NumPy file (bad magic bytes)")
		return
	}
	major := int(data[6])

	var headerLen, dataOffset int
	switch major {
	case 1:
		headerLen = int(uint16Le(data[8:10]))
		dataOffset = 10 + headerLen
	case 2, 3:
		if len(data) < 12 {
			err = fmt.Errorf("npy: file too short for v%d.x header length field", major)
			return
		}
		headerLen = int(uint32Le(data[8:12]))
		dataOffset = 12 + headerLen
	default:
		err = fmt.Errorf("npy: unsupported format version %d.%d", major, int(data[7]))
		return
	}
	if dataOffset > len(data) {
		err = fmt.Errorf("npy: declared header length %d exceeds file length", headerLen)
		return
	}
	hdr := strings.TrimRight(string(data[dataOffset-headerLen:dataOffset]), " \n")

	var fortranOrder bool
	descr, fortranOrder, shape, err = parseHeader(hdr)
	if err != nil {
		err = fmt.Errorf("npy: header parse: %w", err)
		return
	}

	byteOrder, typeChar, itemSize, derr := parseDescr(descr)
	if derr != nil {
		err = derr
		return
	}

	nElems := 1
	for _, s := range shape {
		nElems *= int(s)
	}
	expectedBytes := nElems * itemSize
	if len(data)-dataOffset < expectedBytes {
		err = fmt.Errorf("npy: data payload truncated: need %d bytes, have %d",
			expectedBytes, len(data)-dataOffset)
		return
	}
	raw := data[dataOffset : dataOffset+expectedBytes]

	f32, err = convertToFloat32(raw, byteOrder, typeChar, itemSize, nElems)
	if err != nil {
		return
	}
	if fortranOrder && len(shape) > 1 {
		f32 = fortranToC(f32, shape)
	}
	return
}

// Local LE helpers so this file does not need to import encoding/binary just
// for the two-field header read (npy.go already imports it).
func uint16Le(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }
func uint32Le(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
