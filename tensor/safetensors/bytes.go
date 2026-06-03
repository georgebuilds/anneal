// bytes.go — byte-based inspection API for the WASM tensor-inspect view (W9).
//
// LoadTensors takes a path; the studio holds the file in a Uint8Array and
// needs the raw dtype string ("F32", "BF16", "I64") preserved in the result.
// ReadBytes parses an in-memory safetensors byte slice and returns one
// InspectEntry per tensor, sharing the existing decodeToFloat32 path so the
// dtype policy stays single-sourced.
//
// Tensor order in the returned slice follows the lexicographic key order
// safetensors writers use (and that the on-disk JSON header iterates) so the
// inspector's table is deterministic across runs.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// InspectEntry is one decoded tensor surfaced by ReadBytes. DType is the
// raw safetensors dtype label ("F32", "BF16", "I64", …); Shape is the
// row-major dimensions; Data is the dequantized float32 payload using the
// same conversion rules LoadTensors applies.
type InspectEntry struct {
	Name  string
	DType string
	Shape []int64
	Data  []float32
}

// ReadBytes parses a safetensors file from an in-memory byte slice and
// returns an InspectEntry per tensor (excluding the optional
// "__metadata__" entry which is silently skipped).
func ReadBytes(data []byte) ([]InspectEntry, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short (%d bytes)", len(data))
	}
	hLen := binary.LittleEndian.Uint64(data[:8])
	if uint64(len(data)) < 8+hLen {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file length %d", hLen, len(data)-8)
	}
	hdrBytes := data[8 : 8+hLen]
	dataBuf := data[8+hLen:]

	// Preserve key order for a stable inspect table. json.Unmarshal into a
	// map loses order, so we hand-walk the top-level object boundaries to
	// recover insertion order from the JSON text.
	var rawHdr map[string]json.RawMessage
	if err := json.Unmarshal(hdrBytes, &rawHdr); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}
	keys := orderedTopKeys(hdrBytes)
	// orderedTopKeys is best-effort; if it returns nothing (corrupt JSON
	// would have failed Unmarshal above), fall back to map iteration.
	if len(keys) == 0 {
		for k := range rawHdr {
			keys = append(keys, k)
		}
	}

	type tensorEntry struct {
		DType       string   `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	out := make([]InspectEntry, 0, len(keys))
	for _, name := range keys {
		if name == "__metadata__" {
			continue
		}
		raw, ok := rawHdr[name]
		if !ok {
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
		out = append(out, InspectEntry{
			Name:  name,
			DType: te.DType,
			Shape: te.Shape,
			Data:  f32,
		})
	}
	return out, nil
}

// orderedTopKeys walks a JSON object's text and returns top-level keys in
// declaration order. Returns nil on malformed input (the caller already ran
// json.Unmarshal, so structural errors are surfaced through that path).
func orderedTopKeys(hdr []byte) []string {
	var out []string
	depth := 0
	i := 0
	for i < len(hdr) {
		c := hdr[i]
		switch c {
		case '{', '[':
			depth++
			i++
		case '}', ']':
			depth--
			i++
		case '"':
			// String literal. If we are at depth 1 and the next non-ws char
			// after the closing quote is ':', this is a key.
			start := i
			i++
			for i < len(hdr) && hdr[i] != '"' {
				if hdr[i] == '\\' && i+1 < len(hdr) {
					i += 2
					continue
				}
				i++
			}
			if i >= len(hdr) {
				return nil
			}
			str := string(hdr[start+1 : i])
			i++ // past closing quote
			// Look for ':' separator at depth 1.
			if depth == 1 {
				j := i
				for j < len(hdr) && (hdr[j] == ' ' || hdr[j] == '\t' || hdr[j] == '\n' || hdr[j] == '\r') {
					j++
				}
				if j < len(hdr) && hdr[j] == ':' {
					out = append(out, str)
				}
			}
		default:
			i++
		}
	}
	return out
}
