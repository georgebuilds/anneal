package gpt2

import (
	"os"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/internal/assets"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Helpers (no GPU, no network) ─────────────────────────────────────────────

// gpt2WeightsPath returns the cached GPT-2 safetensors path, or "" if the
// file is not on disk. Used by the loader integration tests to skip cleanly
// when the asset has not been fetched.
func gpt2WeightsPath(t *testing.T) string {
	t.Helper()
	t.Setenv("ANNEAL_OFFLINE", "1")
	path, err := assets.Get("gpt2-safetensors")
	if err != nil {
		return ""
	}
	if _, err := os.Stat(path); err != nil {
		return ""
	}
	return path
}

// ── 1. Name-mapping table is structurally correct ───────────────────────────

// TestGPT2MappingTable verifies the HF-name -> Parameter mapping covers
// every Parameter in g.Params() for a tied-head GPT-2-small. Catches drift
// between the mapping table and any future change to NewGPTWithTiedHead /
// nn.Block.Params() ordering.
func TestGPT2MappingTable(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPTWithTiedHead(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)

	mapping := buildGPT2Mapping(g)

	// Every entry must reference a *Parameter that exists in g.Params().
	paramSet := make(map[*nn.Parameter]bool, len(g.Params()))
	for _, p := range g.Params() {
		paramSet[p] = true
	}
	mappedSet := make(map[*nn.Parameter]bool, len(mapping))
	for _, m := range mapping {
		if m.Param == nil {
			t.Fatalf("mapping entry %q has nil Param", m.HFName)
		}
		mappedSet[m.Param] = true
	}

	// Tied-head Params() contains: Wte.Weight, Wpe.Weight, 12*nLayer block
	// params, LNf.{W,B}, and LMHead.Bias (LMHead.Weight is tied to Wte.Weight
	// and only appears via the Wte entry). That is 12*nLayer + 5 unique
	// Parameter pointers.
	wantParams := 12*GPT2NLayer + 5
	if len(g.Params()) != wantParams {
		t.Fatalf("tied-head GPT.Params() count: got %d, want %d", len(g.Params()), wantParams)
	}

	// LMHead.Bias is intentionally NOT in the mapping (HF GPT-2 has no
	// lm_head bias). Every other parameter must be mapped exactly once.
	for _, p := range g.Params() {
		if p == g.LMHead.Bias {
			if mappedSet[p] {
				t.Errorf("LMHead.Bias should be skipped by the mapping (HF GPT-2 has no lm_head.bias)")
			}
			continue
		}
		if !mappedSet[p] {
			t.Errorf("Parameter %p not covered by HF mapping", p)
		}
	}

	// No duplicate mapping entries (each HFName unique, each Param unique).
	seenName := make(map[string]bool, len(mapping))
	seenPtr := make(map[*nn.Parameter]bool, len(mapping))
	for _, m := range mapping {
		if seenName[m.HFName] {
			t.Errorf("duplicate HF name in mapping: %q", m.HFName)
		}
		if seenPtr[m.Param] {
			t.Errorf("duplicate Param pointer in mapping (HF name %q)", m.HFName)
		}
		seenName[m.HFName] = true
		seenPtr[m.Param] = true
	}

	// Spot-check key names: every block has 12 entries (2 LN1 + 4 attn +
	// 2 LN2 + 4 mlp), embeddings have 2, final LN has 2 -> total 12*L + 4.
	wantMapEntries := 12*GPT2NLayer + 4
	if len(mapping) != wantMapEntries {
		t.Fatalf("mapping entry count: got %d, want %d (12*nLayer + 4 root entries)",
			len(mapping), wantMapEntries)
	}

	// Conv1D transpose flags must fire exactly on c_attn, c_proj (attn),
	// c_fc, c_proj (mlp); never on LN/embedding/bias entries.
	for _, m := range mapping {
		shouldTranspose := strings.HasSuffix(m.HFName, "c_attn.weight") ||
			strings.HasSuffix(m.HFName, "c_proj.weight") ||
			strings.HasSuffix(m.HFName, "c_fc.weight")
		if shouldTranspose != m.Transpose {
			t.Errorf("mapping %q: Transpose flag %v but expected %v based on name",
				m.HFName, m.Transpose, shouldTranspose)
		}
	}
}

// ── 2. Conv1D transpose logic ────────────────────────────────────────────────

// TestConv1DTransposeLogic verifies the in-place transpose used by
// LoadGPT2WeightsInto via a tiny synthetic 2x3 source. Catches index-math
// regressions independent of any real weights.
func TestConv1DTransposeLogic(t *testing.T) {
	// HF layout [in=2, out=3]:
	//   src = [a b c
	//          d e f]  (row-major)
	src := []float32{1, 2, 3, 4, 5, 6}

	// Anneal Linear stores [out=3, in=2]:
	//   dst = [a d
	//          b e
	//          c f]  (row-major)
	wantDst := []float32{1, 4, 2, 5, 3, 6}

	in, out := int64(2), int64(3)
	dst := make([]float32, len(src))
	for i := int64(0); i < in; i++ {
		for j := int64(0); j < out; j++ {
			dst[j*in+i] = src[i*out+j]
		}
	}
	for i, want := range wantDst {
		if dst[i] != want {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

// ── 3. Weight tying gives shared *Parameter ─────────────────────────────────

// TestWeightTyingPointerIdentity verifies that NewGPTWithTiedHead aliases
// the LM head's Weight to the embedding Weight as a single *Parameter, and
// that Params() reports the tied parameter exactly once.
func TestWeightTyingPointerIdentity(t *testing.T) {
	a := uop.NewArena(1 << 14)

	// Reference (non-tied) GPT: LMHead.Weight is a distinct Parameter.
	gRef := nn.NewGPT(a, 16, 2, 2, 8, 8)
	if gRef.Wte.Weight == gRef.LMHead.Weight {
		t.Fatalf("non-tied NewGPT: Wte.Weight and LMHead.Weight unexpectedly share pointer")
	}
	if gRef.TieWeights {
		t.Fatalf("non-tied NewGPT: TieWeights should be false")
	}
	wantNonTied := 12*2 + 6 // matches TestGPTParamsCount
	if got := len(gRef.Params()); got != wantNonTied {
		t.Fatalf("non-tied Params(): got %d, want %d", got, wantNonTied)
	}

	// Tied GPT: aliased pointers, Params() drops the duplicate.
	gTied := nn.NewGPTWithTiedHead(a, 16, 2, 2, 8, 8)
	if !gTied.TieWeights {
		t.Fatalf("NewGPTWithTiedHead: TieWeights should be true")
	}
	if gTied.Wte.Weight != gTied.LMHead.Weight {
		t.Fatalf("NewGPTWithTiedHead: Wte.Weight (%p) != LMHead.Weight (%p), expected pointer-identity",
			gTied.Wte.Weight, gTied.LMHead.Weight)
	}
	wantTied := 12*2 + 5 // one fewer than non-tied (LMHead.Weight is skipped)
	if got := len(gTied.Params()); got != wantTied {
		t.Fatalf("tied Params(): got %d, want %d", got, wantTied)
	}

	// Pointer identity carries through Load: after Wte.Weight.Load, both
	// the embedding and LM head see the same fresh leaf.
	a2 := uop.NewArena(1 << 14)
	for _, p := range gTied.Params() {
		p.Load(a2)
	}
	if gTied.Wte.Weight.T != gTied.LMHead.Weight.T {
		t.Fatalf("after Load: Wte.Weight.T (%p) != LMHead.Weight.T (%p), tied leaf identity broken",
			gTied.Wte.Weight.T, gTied.LMHead.Weight.T)
	}

	// Values share the underlying slice: same backing array, same length.
	if &gTied.Wte.Weight.Value[0] != &gTied.LMHead.Weight.Value[0] {
		t.Fatalf("tied Value: Wte.Weight.Value and LMHead.Weight.Value have distinct backing arrays")
	}
}

// ── 4. Tied head: precomputed shape sanity ──────────────────────────────────

// TestTiedHeadShapeIsCompatible documents that Wte.Weight and a
// non-tied LMHead.Weight have the same shape, so the alias is sound.
func TestTiedHeadShapeIsCompatible(t *testing.T) {
	a := uop.NewArena(1 << 14)
	g := nn.NewGPT(a, GPT2Vocab, 1, 1, GPT2NEmbd, GPT2BlockSize)
	wteShape := g.Wte.Weight.T.Shape()
	lmShape := g.LMHead.Weight.T.Shape()
	if len(wteShape) != 2 || len(lmShape) != 2 {
		t.Fatalf("expected rank-2 Wte and LMHead weights; got %v and %v", wteShape, lmShape)
	}
	if wteShape[0] != lmShape[0] || wteShape[1] != lmShape[1] {
		t.Fatalf("tied-head shape mismatch: Wte %v != LMHead %v", wteShape, lmShape)
	}
}
