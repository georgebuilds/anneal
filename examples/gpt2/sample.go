package gpt2

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// ── Sampling configuration ───────────────────────────────────────────────────

// SampleOptions controls the autoregressive sampler. Defaults mirror the
// CLI: 20 new tokens, temperature 1.0, top-k=40, greedy=false. Greedy=true
// is the deterministic argmax path used by the oracle tests; stochastic
// sampling uses a math/rand.Rand seeded by the caller (or a fresh source
// if Rng==nil).
type SampleOptions struct {
	// MaxTokens is the upper bound on generated tokens (in addition to the
	// prompt). Must be > 0.
	MaxTokens int
	// Temperature scales logits before softmax. Values <= 0 are treated as
	// greedy (Temperature is ignored on the greedy path).
	Temperature float32
	// TopK keeps only the K highest-probability tokens before sampling.
	// Set to <= 0 to disable.
	TopK int
	// Greedy forces argmax sampling and ignores Temperature and TopK.
	Greedy bool
	// Rng is the stochastic-sample RNG. When nil and Greedy is false, a
	// fresh math/rand source is created from a fixed seed (1) so results
	// are reproducible by default.
	Rng *rand.Rand
}

// DefaultSampleOptions returns the canonical CLI defaults.
func DefaultSampleOptions() SampleOptions {
	return SampleOptions{
		MaxTokens:   20,
		Temperature: 1.0,
		TopK:        40,
		Greedy:      false,
	}
}

// Sample runs autoregressive generation for opts.MaxTokens steps, starting
// from prompt. The returned text is the decoded concatenation of the prompt
// ids plus all generated ids. ctxLen bounds the per-step context window;
// when the running id sequence is longer than ctxLen, only the trailing
// ctxLen ids are fed to Forward (this is the canonical sliding-window
// inference pattern; the model has no KV cache in this slice). ctxLen must
// be <= g.BlockSize.
//
// The function is forward-only and arena-fresh per step (matching the
// nanogpt sampler precedent): for each new token we allocate a fresh
// uop.Arena, reload every parameter into it, build a fresh forward graph,
// realize the logits, and pull the last-position vector back to the host
// for sampling. This is O(n^2) wall time across n tokens; KV-cache reuse
// is deferred to a later slice.
func Sample(g *nn.GPT, bpe *BPE, prompt string, ctxLen int, device string, opts SampleOptions) (string, error) {
	if opts.MaxTokens <= 0 {
		return "", fmt.Errorf("gpt2: Sample: MaxTokens must be > 0, got %d", opts.MaxTokens)
	}
	if ctxLen <= 0 || ctxLen > g.BlockSize {
		return "", fmt.Errorf("gpt2: Sample: ctxLen %d must be in (0, %d]", ctxLen, g.BlockSize)
	}
	rng := opts.Rng
	if !opts.Greedy && rng == nil {
		rng = rand.New(rand.NewSource(1))
	}

	ids := bpe.Encode(prompt)
	if len(ids) == 0 {
		return "", fmt.Errorf("gpt2: Sample: prompt encoded to zero tokens (%q)", prompt)
	}
	if len(ids) > ctxLen {
		// The model can only attend over ctxLen tokens; we still keep the
		// untruncated ids slice for the final decode so the user sees their
		// full prompt back. Only the rolling-window fed to Forward is
		// truncated.
		// Nothing else to do here; the per-step window selection below
		// already slices the tail.
		_ = ids
	}

	for k := 0; k < opts.MaxTokens; k++ {
		// Build the per-step input window: at most ctxLen trailing ids.
		windowIds := ids
		if len(windowIds) > ctxLen {
			windowIds = windowIds[len(windowIds)-ctxLen:]
		}
		T := int64(len(windowIds))

		a := uop.NewArena(1 << 20)
		for _, p := range g.Params() {
			p.Load(a)
		}
		idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsLeafBits(windowIds))

		logits := g.Forward(idx)
		if err := tensor.Realize(logits); err != nil {
			return "", fmt.Errorf("gpt2: Sample: realize logits at step %d: %w", k, err)
		}

		data := logits.Data()
		V := g.Vocab
		base := (int(T) - 1) * V
		last := data[base : base+V]

		var nextID int32
		if opts.Greedy {
			nextID = argmaxInt32(last)
		} else {
			nextID = sampleFromLogits(last, opts.Temperature, opts.TopK, rng)
		}
		ids = append(ids, nextID)
	}

	return bpe.Decode(ids), nil
}

// SampleArgmaxFirst is a convenience used by the HF-oracle test: encode the
// prompt, run forward once, return the argmax token id from the
// last-position logits. No autoregression, no decoding.
func SampleArgmaxFirst(g *nn.GPT, bpe *BPE, prompt string, ctxLen int, device string) (int32, error) {
	if ctxLen <= 0 || ctxLen > g.BlockSize {
		return 0, fmt.Errorf("gpt2: SampleArgmaxFirst: ctxLen %d must be in (0, %d]", ctxLen, g.BlockSize)
	}
	ids := bpe.Encode(prompt)
	if len(ids) == 0 {
		return 0, fmt.Errorf("gpt2: SampleArgmaxFirst: prompt encoded to zero tokens (%q)", prompt)
	}
	windowIds := ids
	if len(windowIds) > ctxLen {
		windowIds = windowIds[len(windowIds)-ctxLen:]
	}
	T := int64(len(windowIds))

	a := uop.NewArena(1 << 20)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, "webgpu")
	idx.SetData(int32sAsLeafBits(windowIds))

	logits := g.Forward(idx)
	if err := tensor.Realize(logits); err != nil {
		return 0, fmt.Errorf("gpt2: SampleArgmaxFirst: realize: %w", err)
	}
	data := logits.Data()
	V := g.Vocab
	base := (int(T) - 1) * V
	return argmaxInt32(data[base : base+V]), nil
}

// ── Sampling helpers ─────────────────────────────────────────────────────────

// argmaxInt32 returns the index of the maximum element in v. Stable across
// ties: the lowest index wins (matches PyTorch's argmax convention).
func argmaxInt32(v []float32) int32 {
	bestVal := float32(math.Inf(-1))
	best := int32(0)
	for i, x := range v {
		if x > bestVal {
			bestVal = x
			best = int32(i)
		}
	}
	return best
}

// sampleFromLogits applies the standard temperature + top-k + softmax +
// multinomial-sample pipeline. The logits slice is read-only and must have
// length == vocab. When temperature <= 0 the function falls back to argmax
// (treated as greedy); when topK <= 0 every token is eligible.
func sampleFromLogits(logits []float32, temperature float32, topK int, rng *rand.Rand) int32 {
	if temperature <= 0 {
		return argmaxInt32(logits)
	}
	// Temperature-scaled logits.
	scaled := make([]float32, len(logits))
	invT := float32(1.0) / temperature
	for i, v := range logits {
		scaled[i] = v * invT
	}

	// Top-k filter: keep the highest topK values; mask the rest to -inf so
	// softmax assigns them probability zero.
	if topK > 0 && topK < len(scaled) {
		// Find the topK threshold by sorting a copy.
		copyVals := make([]float32, len(scaled))
		copy(copyVals, scaled)
		sort.Slice(copyVals, func(i, j int) bool { return copyVals[i] > copyVals[j] })
		thresh := copyVals[topK-1]
		negInf := float32(math.Inf(-1))
		for i, v := range scaled {
			if v < thresh {
				scaled[i] = negInf
			}
		}
	}

	// Numerically stable softmax (subtract max).
	maxV := float32(math.Inf(-1))
	for _, v := range scaled {
		if v > maxV {
			maxV = v
		}
	}
	probs := make([]float32, len(scaled))
	var sum float32
	for i, v := range scaled {
		// math.Inf(-1) - maxV is still -inf, exp -> 0; safe.
		probs[i] = float32(math.Exp(float64(v - maxV)))
		sum += probs[i]
	}
	if sum == 0 {
		return argmaxInt32(logits)
	}
	// Multinomial sample via cumulative distribution.
	r := float32(rng.Float64()) * sum
	var acc float32
	for i, p := range probs {
		acc += p
		if r <= acc {
			return int32(i)
		}
	}
	// Floating-point slack: fall back to the last index.
	return int32(len(probs) - 1)
}

// int32sAsLeafBits packs []int32 into the float32-bits buffer that
// tensor.NewLeaf(..., Int32, ...).SetData expects. Mirrors the helper used
// in examples/nanogpt_data.go and tensor/nn/embedding_test.go.
func int32sAsLeafBits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// ── CLI entry point ──────────────────────────────────────────────────────────

// RunSampleCLI is invoked by cmd/anneal/cmd_gpt2.go. It encapsulates the
// full GPT-2 sample pipeline: fetch assets, build BPE + GPT, run Sample,
// print the generated text. Splitting the CLI body from cmd/anneal keeps
// the example package self-contained and lets tests exercise the same
// code path the user runs without a process spawn.
//
// w is the output sink (usually os.Stdout). When plain is true, the output
// is just the generated text with a trailing newline; otherwise a short
// header lists the prompt and configuration for readability.
//
//nolint:errcheck // best-effort writes to stdout/stderr
func RunSampleCLI(w io.Writer, device string, prompt string, opts SampleOptions, plain bool) error {
	a := uop.NewArena(1 << 14)
	g, bpe, err := LoadGPT2(a, device)
	if err != nil {
		return err
	}

	// ctxLen: encode the prompt to know how many ids we start with, then
	// pick a context window of min(GPT2BlockSize, len(ids)+MaxTokens). This
	// avoids dispatching a 1024-token forward pass for a single-word prompt.
	promptIds := bpe.Encode(prompt)
	ctxLen := len(promptIds) + opts.MaxTokens
	if ctxLen > g.BlockSize {
		ctxLen = g.BlockSize
	}
	if ctxLen < 1 {
		return fmt.Errorf("gpt2: prompt %q encoded to zero tokens", prompt)
	}

	text, err := Sample(g, bpe, prompt, ctxLen, device, opts)
	if err != nil {
		return err
	}

	if plain {
		fmt.Fprintln(w, text)
		return nil
	}
	mode := "stochastic"
	if opts.Greedy {
		mode = "greedy"
	}
	fmt.Fprintf(w, "prompt: %s\n", prompt)
	fmt.Fprintf(w, "mode: %s · max-tokens: %d · temperature: %.2f · top-k: %d\n",
		mode, opts.MaxTokens, opts.Temperature, opts.TopK)
	fmt.Fprintf(w, "---\n%s\n", text)
	return nil
}
