package gpt2

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/tensor/safetensors"
	"github.com/georgebuilds/anneal/uop"
)

// ── applyTensorToParam: transpose + error branches ──────────────────────────

// TestApplyTensorToParamTranspose covers the happy transpose path: an HF
// Conv1D [in=2, out=3] source is transposed into the [out, in] anneal layout.
func TestApplyTensorToParamTranspose(t *testing.T) {
	p := &nn.Parameter{Value: make([]float32, 6)}
	m := paramMap{HFName: "x.c_attn.weight", Param: p, Transpose: true}
	entry := safetensors.Entry{
		// row-major [in=2, out=3]: rows (a b c) (d e f)
		Data:  []float32{1, 2, 3, 4, 5, 6},
		Shape: []int64{2, 3},
	}
	if err := applyTensorToParam(m, entry); err != nil {
		t.Fatalf("applyTensorToParam: %v", err)
	}
	// dst[out=3, in=2] = (1 4)(2 5)(3 6)
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if p.Value[i] != want[i] {
			t.Errorf("dst[%d] = %v, want %v (full %v)", i, p.Value[i], want[i], p.Value)
		}
	}
}

// TestApplyTensorToParamTransposeRankError covers the rank!=2 guard on the
// transpose path.
func TestApplyTensorToParamTransposeRankError(t *testing.T) {
	p := &nn.Parameter{Value: make([]float32, 6)}
	m := paramMap{HFName: "x.c_fc.weight", Param: p, Transpose: true}
	entry := safetensors.Entry{Data: make([]float32, 6), Shape: []int64{6}} // rank 1
	err := applyTensorToParam(m, entry)
	if err == nil || !strings.Contains(err.Error(), "rank-2 Conv1D") {
		t.Errorf("rank-1 transpose source: got %v, want rank-2 error", err)
	}
}

// TestApplyTensorToParamTransposeCountError covers the element-count guard on
// the transpose path (shape product != len(param)).
func TestApplyTensorToParamTransposeCountError(t *testing.T) {
	p := &nn.Parameter{Value: make([]float32, 5)} // expect 6
	m := paramMap{HFName: "x.c_proj.weight", Param: p, Transpose: true}
	entry := safetensors.Entry{Data: make([]float32, 6), Shape: []int64{2, 3}}
	err := applyTensorToParam(m, entry)
	if err == nil || !strings.Contains(err.Error(), "Conv1D transpose source") {
		t.Errorf("count mismatch transpose: got %v, want count error", err)
	}
}

// TestApplyTensorToParamNoTranspose covers the no-transpose happy path: a
// straight copy of matching length.
func TestApplyTensorToParamNoTranspose(t *testing.T) {
	p := &nn.Parameter{Value: make([]float32, 4)}
	m := paramMap{HFName: "ln_f.weight", Param: p, Transpose: false}
	entry := safetensors.Entry{Data: []float32{9, 8, 7, 6}, Shape: []int64{4}}
	if err := applyTensorToParam(m, entry); err != nil {
		t.Fatalf("applyTensorToParam: %v", err)
	}
	for i, want := range []float32{9, 8, 7, 6} {
		if p.Value[i] != want {
			t.Errorf("Value[%d] = %v, want %v", i, p.Value[i], want)
		}
	}
}

// TestApplyTensorToParamNoTransposeCountError covers the element-count guard
// on the no-transpose path.
func TestApplyTensorToParamNoTransposeCountError(t *testing.T) {
	p := &nn.Parameter{Value: make([]float32, 4)}
	m := paramMap{HFName: "wte.weight", Param: p, Transpose: false}
	entry := safetensors.Entry{Data: []float32{1, 2, 3}, Shape: []int64{3}}
	err := applyTensorToParam(m, entry)
	if err == nil || !strings.Contains(err.Error(), "no transpose") {
		t.Errorf("count mismatch no-transpose: got %v, want count error", err)
	}
}

// ── LoadGPT2WeightsInto: tensor-not-found branch ────────────────────────────

// writeEmptySafetensors writes a structurally valid safetensors file whose
// header is an empty JSON object (no tensors). LoadTensors parses it fine but
// every lookup misses, so LoadGPT2WeightsInto hits its "tensor not found"
// branch on the very first mapping entry.
func writeEmptySafetensors(t *testing.T) string {
	t.Helper()
	hdr := []byte("{}")
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	buf := make([]byte, 8+len(hdr))
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(hdr)))
	copy(buf[8:], hdr)
	path := filepath.Join(t.TempDir(), "empty.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write empty safetensors: %v", err)
	}
	return path
}

// writeSafetensorsOneTensor writes a structurally valid safetensors file with
// a single F32 tensor of the given name and element count. Used to drive the
// LoadGPT2WeightsInto per-tensor error path (the named tensor is present so
// the `!ok` check passes, but applyTensorToParam rejects the size).
func writeSafetensorsOneTensor(t *testing.T, name string, nElems int) string {
	t.Helper()
	hdr := []byte(`{"` + name + `":{"dtype":"F32","shape":[` + itoaInt(nElems) + `],"data_offsets":[0,` + itoaInt(nElems*4) + `]}}`)
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	data := make([]byte, nElems*4) // zero-filled F32 payload
	buf := make([]byte, 8+len(hdr)+len(data))
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(hdr)))
	copy(buf[8:], hdr)
	copy(buf[8+len(hdr):], data)
	path := filepath.Join(t.TempDir(), "one.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write one-tensor safetensors: %v", err)
	}
	return path
}

// TestLoadGPT2WeightsIntoApplyError covers the `applyTensorToParam` error
// propagation branch in LoadGPT2WeightsInto: wte.weight is present but has the
// wrong element count, so the per-tensor copy fails and the error bubbles up.
func TestLoadGPT2WeightsIntoApplyError(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)
	// wte.weight is the first mapping entry (no transpose). 3 elements != the
	// real 50257*768, so applyTensorToParam returns the count-mismatch error.
	path := writeSafetensorsOneTensor(t, "wte.weight", 3)
	err := LoadGPT2WeightsInto(g, path)
	if err == nil || !strings.Contains(err.Error(), "no transpose") {
		t.Errorf("LoadGPT2WeightsInto(bad wte size): got %v, want count error", err)
	}
}

// TestLoadGPT2WeightsIntoTensorNotFound covers the `!ok` branch: a canonical
// tied-head GPT loaded against a parseable-but-empty safetensors file fails
// on the first missing tensor (wte.weight).
func TestLoadGPT2WeightsIntoTensorNotFound(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)
	path := writeEmptySafetensors(t)
	err := LoadGPT2WeightsInto(g, path)
	if err == nil || !strings.Contains(err.Error(), "not found in safetensors") {
		t.Errorf("LoadGPT2WeightsInto(empty file): got %v, want tensor-not-found error", err)
	}
	// The first checked tensor is wte.weight.
	if err != nil && !strings.Contains(err.Error(), "wte.weight") {
		t.Errorf("expected wte.weight to be the first missing tensor, got %v", err)
	}
}

// TestNewBPEParseMergesPropagates covers the NewBPE -> parseMerges error
// propagation when the merges blob is malformed (no valid merge rules after
// the header).
func TestNewBPEParseMergesPropagates(t *testing.T) {
	_, err := NewBPE([]byte(tinyVocab), []byte("#version: 0.2\nbadline\n"))
	if err == nil || !strings.Contains(err.Error(), "bad merge line") {
		t.Errorf("NewBPE with bad merges: got %v, want bad-merge-line error", err)
	}
}

// Compile-time assertion that math is referenced (keeps imports honest if the
// helper above is edited later).
var _ = math.Float32bits
