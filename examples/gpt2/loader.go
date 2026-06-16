// Package gpt2: HuggingFace GPT-2 weight loader (this file).
//
// This file glues an anneal nn.GPT module to the canonical HuggingFace GPT-2
// safetensors checkpoint. It handles three concerns:
//
//  1. Asset acquisition. The vocab.json, merges.txt, and model.safetensors
//     files are resolved through internal/assets, which downloads them on
//     first use and SHA-verifies on every access.
//
//  2. Name mapping. HuggingFace's tensor names (e.g. h.0.attn.c_attn.weight)
//     are mapped onto anneal nn.Parameter pointers (e.g. g.Blocks[0].Attn.QKV.Weight).
//     The HF file shipped by huggingface.co/gpt2 uses unprefixed names: there
//     is no transformer. prefix and there is no lm_head.weight entry: the LM
//     head is tied to wte.weight.
//
//  3. Shape adjustment. HuggingFace's Conv1D layer stores its weight as
//     [in, out], whereas anneal's nn.Linear stores [out, in]. The c_attn,
//     c_proj (attn), c_fc, and c_proj (mlp) tensors are therefore transposed
//     in-place on load. Embeddings and LayerNorm parameters carry no
//     transpose because their HF shapes already match anneal's shapes.
//
// The full pipeline (LoadGPT2) constructs an nn.GPT with TieWeights=true,
// downloads the assets, parses the safetensors blob, copies the mapped
// values (transposed where needed) into each Parameter.Value, and returns
// the wired model plus the BPE tokenizer.
//
// Strict scope: forward-only. The tied lm_head Parameter is shared with
// wte.weight; calling Backward through both paths is not supported in this
// slice and would route two gradient sources into the same leaf without an
// OpShare seam, which is out of scope for Slice O.
package gpt2

import (
	"fmt"
	"os"

	"github.com/georgebuilds/anneal/internal/assets"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/tensor/safetensors"
	"github.com/georgebuilds/anneal/uop"
)

// GPT-2-small canonical hyperparameters. These are the only configuration
// the HuggingFace gpt2 checkpoint supports; loading into a model with any
// other shape would fail every parameter's shape check.
const (
	GPT2Vocab     = 50257
	GPT2NLayer    = 12
	GPT2NHead     = 12
	GPT2NEmbd     = 768
	GPT2BlockSize = 1024
)

// LoadGPT2 constructs a tied-head GPT-2-small nn.GPT, populates every
// parameter from the HuggingFace safetensors checkpoint resolved by
// internal/assets, and returns the model alongside a ready BPE tokenizer.
//
// The function downloads any missing asset on first call (subject to
// ANNEAL_OFFLINE=1, which short-circuits with a clear error). Successful
// return guarantees every parameter listed in g.Params() has been seeded
// from the file; the caller is responsible for calling p.Load(arena) on
// each Parameter before the first forward pass, the same convention every
// other anneal nn module uses.
func LoadGPT2(a *uop.Arena, device string) (*nn.GPT, *BPE, error) {
	vocabPath, err := assets.Get("gpt2-vocab")
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: fetch vocab: %w", err)
	}
	mergesPath, err := assets.Get("gpt2-merges")
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: fetch merges: %w", err)
	}
	weightsPath, err := assets.Get("gpt2-safetensors")
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: fetch weights: %w", err)
	}

	vocabBytes, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: read vocab %s: %w", vocabPath, err)
	}
	mergesBytes, err := os.ReadFile(mergesPath)
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: read merges %s: %w", mergesPath, err)
	}
	bpe, err := NewBPE(vocabBytes, mergesBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("gpt2: build BPE: %w", err)
	}

	g := nn.NewGPTWithTiedHead(a, GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize)
	if err := LoadGPT2WeightsInto(g, weightsPath); err != nil {
		return nil, nil, err
	}
	_ = device // device is captured at module construction; kept here for symmetry with other loaders
	return g, bpe, nil
}

// LoadGPT2WeightsInto reads the HuggingFace gpt2 safetensors file at path
// and copies every tensor into the corresponding nn.Parameter on g, applying
// the Conv1D [in, out] -> [out, in] transpose where the HF layer is a
// Conv1D. g must be a tied-head GPT (TieWeights=true) constructed with the
// canonical GPT-2-small shape; anything else is treated as a programmer
// error and surfaced as a clear shape mismatch on the first parameter that
// fails to align.
//
// The function reads the whole file into memory (the safetensors loader is
// not streaming); the file is ~522 MiB which is comfortable on any host
// that runs anneal's GPU backend.
func LoadGPT2WeightsInto(g *nn.GPT, path string) error {
	if !g.TieWeights {
		return fmt.Errorf("gpt2: LoadGPT2WeightsInto expects a tied-head GPT (built via nn.NewGPTWithTiedHead); got TieWeights=false")
	}
	if g.NLayer != GPT2NLayer || g.NHead != GPT2NHead || g.NEmbd != GPT2NEmbd ||
		g.BlockSize != GPT2BlockSize || g.Vocab != GPT2Vocab {
		return fmt.Errorf("gpt2: expected canonical GPT-2-small config (vocab=%d nLayer=%d nHead=%d nEmbd=%d blockSize=%d); got vocab=%d nLayer=%d nHead=%d nEmbd=%d blockSize=%d",
			GPT2Vocab, GPT2NLayer, GPT2NHead, GPT2NEmbd, GPT2BlockSize,
			g.Vocab, g.NLayer, g.NHead, g.NEmbd, g.BlockSize)
	}

	tensors, err := safetensors.LoadTensors(path)
	if err != nil {
		return fmt.Errorf("gpt2: parse safetensors %s: %w", path, err)
	}

	for _, m := range buildGPT2Mapping(g) {
		entry, ok := tensors[m.HFName]
		if !ok {
			return fmt.Errorf("gpt2: tensor %q not found in safetensors file %s", m.HFName, path)
		}
		if err := applyTensorToParam(m, entry); err != nil {
			return err
		}
	}
	return nil
}

// applyTensorToParam copies one safetensors Entry into the mapped Parameter,
// applying the Conv1D [in, out] -> [out, in] transpose when m.Transpose is
// set. Factored out of LoadGPT2WeightsInto so the per-tensor shape checks and
// the transpose index math are unit-testable with tiny synthetic entries (the
// full GPT-2 checkpoint is ~500 MB, far too large for a host-side test).
func applyTensorToParam(m paramMap, entry safetensors.Entry) error {
	if m.Transpose {
		if len(entry.Shape) != 2 {
			return fmt.Errorf("gpt2: tensor %q: expected rank-2 Conv1D weight, got shape %v", m.HFName, entry.Shape)
		}
		in, out := entry.Shape[0], entry.Shape[1]
		if int64(len(m.Param.Value)) != in*out {
			return fmt.Errorf("gpt2: tensor %q: element count %d != param %d (Conv1D transpose source [%d, %d])",
				m.HFName, len(entry.Data), len(m.Param.Value), in, out)
		}
		// Anneal Linear stores [out, in]; HF Conv1D stores [in, out].
		// Transpose src[i, j] -> dst[j, i]: dst[j*in + i] = src[i*out + j].
		for i := int64(0); i < in; i++ {
			for j := int64(0); j < out; j++ {
				m.Param.Value[j*in+i] = entry.Data[i*out+j]
			}
		}
		return nil
	}
	if len(entry.Data) != len(m.Param.Value) {
		return fmt.Errorf("gpt2: tensor %q: element count %d != param %d (no transpose; HF shape %v)",
			m.HFName, len(entry.Data), len(m.Param.Value), entry.Shape)
	}
	copy(m.Param.Value, entry.Data)
	return nil
}

// paramMap is one entry in the HF -> anneal parameter mapping. Transpose is
// true exactly for HF Conv1D weights (c_attn / c_proj / c_fc), where the HF
// layout is [in, out] and anneal nn.Linear stores [out, in].
type paramMap struct {
	HFName    string
	Param     *nn.Parameter
	Transpose bool
}

// buildGPT2Mapping returns the full HF-name -> anneal-parameter table for a
// tied-head GPT-2-small. Every parameter that participates in forward
// inference is listed exactly once; LMHead.Weight is intentionally absent
// because the HF file omits lm_head.weight entirely (it is tied to wte.weight
// at the source). LMHead.Bias is also absent: HF GPT-2 has no LM-head bias;
// our LMHead.Bias is left at its zero default and adds nothing to the logits.
//
// h.{i}.attn.bias is the HF causal-mask buffer; anneal's CausalSelfAttention
// builds the mask host-side from the blockSize hyperparameter, so the HF
// attn.bias entry is intentionally skipped here. It is not a Parameter on
// our side, just a fixed leaf buffer constructed at Forward time.
func buildGPT2Mapping(g *nn.GPT) []paramMap {
	out := make([]paramMap, 0, 12+12*g.NLayer)

	// Embeddings (no transpose; HF and anneal both store [num, dim]).
	out = append(out,
		paramMap{HFName: "wte.weight", Param: g.Wte.Weight, Transpose: false},
		paramMap{HFName: "wpe.weight", Param: g.Wpe.Weight, Transpose: false},
	)

	// Per-block parameters in HF naming.
	for i := 0; i < g.NLayer; i++ {
		blk := g.Blocks[i]
		prefix := fmt.Sprintf("h.%d.", i)

		// LayerNorm 1 (pre-attention).
		out = append(out,
			paramMap{HFName: prefix + "ln_1.weight", Param: blk.LN1.Weight},
			paramMap{HFName: prefix + "ln_1.bias", Param: blk.LN1.Bias},
		)
		// Attention: c_attn (fused QKV) + c_proj (output projection). Both
		// are Conv1D layers on the HF side, so we transpose the weight on load.
		out = append(out,
			paramMap{HFName: prefix + "attn.c_attn.weight", Param: blk.Attn.QKV.Weight, Transpose: true},
			paramMap{HFName: prefix + "attn.c_attn.bias", Param: blk.Attn.QKV.Bias},
			paramMap{HFName: prefix + "attn.c_proj.weight", Param: blk.Attn.Proj.Weight, Transpose: true},
			paramMap{HFName: prefix + "attn.c_proj.bias", Param: blk.Attn.Proj.Bias},
		)
		// LayerNorm 2 (pre-MLP).
		out = append(out,
			paramMap{HFName: prefix + "ln_2.weight", Param: blk.LN2.Weight},
			paramMap{HFName: prefix + "ln_2.bias", Param: blk.LN2.Bias},
		)
		// MLP: c_fc (FC1) + c_proj (FC2). Both are Conv1D on HF.
		out = append(out,
			paramMap{HFName: prefix + "mlp.c_fc.weight", Param: blk.MLP.FC1.Weight, Transpose: true},
			paramMap{HFName: prefix + "mlp.c_fc.bias", Param: blk.MLP.FC1.Bias},
			paramMap{HFName: prefix + "mlp.c_proj.weight", Param: blk.MLP.FC2.Weight, Transpose: true},
			paramMap{HFName: prefix + "mlp.c_proj.bias", Param: blk.MLP.FC2.Bias},
		)
	}

	// Final LayerNorm.
	out = append(out,
		paramMap{HFName: "ln_f.weight", Param: g.LNf.Weight},
		paramMap{HFName: "ln_f.bias", Param: g.LNf.Bias},
	)

	return out
}
