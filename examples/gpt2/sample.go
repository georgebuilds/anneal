package gpt2

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

// StreamToken is one record emitted by SampleStream - a per-token callback
// payload that carries the id, decoded text fragment, and a logit-summary
// string the studio's generate view renders in the last-token panel.
//
// Argmax is the argmax over the full logit vector at this step. For a
// greedy decode Argmax equals ID; for stochastic sampling Argmax is the
// "what would the deterministic path have picked" and ID is the actual
// sampled id.
type StreamToken struct {
	Step         int     // 0-based token index within this generation
	ID           int32   // sampled token id
	Text         string  // decoded text fragment for this id alone
	Argmax       int32   // argmax id at this step (for ref-match comparisons)
	LogitMax     float32 // value of the max logit
	LogitMaxIdx  int     // index where the max logit lives (same as Argmax)
	LogitSummary string  // pre-formatted "max=X.YZ at idx N"
}

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

// SampleStream is the streaming variant of Sample used by the studio's
// /sse/generate handler. It runs the same autoregressive forward loop but
// invokes onTok(...) after each emitted token with the per-step record;
// the returned string is the same decoded concatenation Sample returns.
//
// The context is checked at the top of every step so a client disconnect
// (caught by the SSE handler) cleanly aborts the generation. The error
// returned in that case is ctx.Err(); the caller can decide whether to
// treat it as a failure (PhaseError) or a graceful stop.
//
// onTok must be non-nil; otherwise the streaming contract is meaningless
// and Sample is the right entry point. Returns the decoded text up to and
// including the last successfully emitted token if the context is
// cancelled mid-stream.
func SampleStream(
	ctx context.Context,
	g *nn.GPT,
	bpe *BPE,
	prompt string,
	ctxLen int,
	device string,
	opts SampleOptions,
	onTok func(StreamToken),
) (string, error) {
	if onTok == nil {
		return "", fmt.Errorf("gpt2: SampleStream: onTok callback must be non-nil")
	}
	if opts.MaxTokens <= 0 {
		return "", fmt.Errorf("gpt2: SampleStream: MaxTokens must be > 0, got %d", opts.MaxTokens)
	}
	if ctxLen <= 0 || ctxLen > g.BlockSize {
		return "", fmt.Errorf("gpt2: SampleStream: ctxLen %d must be in (0, %d]", ctxLen, g.BlockSize)
	}
	rng := opts.Rng
	if !opts.Greedy && rng == nil {
		rng = rand.New(rand.NewSource(1))
	}

	ids := bpe.Encode(prompt)
	if len(ids) == 0 {
		return "", fmt.Errorf("gpt2: SampleStream: prompt encoded to zero tokens (%q)", prompt)
	}
	startLen := len(ids)

	for k := 0; k < opts.MaxTokens; k++ {
		if err := ctx.Err(); err != nil {
			return bpe.Decode(ids), err
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
		idx := tensor.NewLeaf(a, []int64{1, T}, uop.Dtypes.Int32, device)
		idx.SetData(int32sAsLeafBits(windowIds))

		logits := g.Forward(idx)
		if err := tensor.Realize(logits); err != nil {
			return bpe.Decode(ids), fmt.Errorf("gpt2: SampleStream: realize logits at step %d: %w", k, err)
		}

		data := logits.Data()
		V := g.Vocab
		base := (int(T) - 1) * V
		last := data[base : base+V]

		argmaxID := argmaxInt32(last)
		var nextID int32
		if opts.Greedy {
			nextID = argmaxID
		} else {
			nextID = sampleFromLogits(last, opts.Temperature, opts.TopK, rng)
		}

		// Per-step text fragment: the studio prepends the prompt-echo so
		// the SSE payload only carries the newly emitted token text.
		fragment := bpe.Decode([]int32{nextID})
		maxV := last[argmaxID]
		onTok(StreamToken{
			Step:         k,
			ID:           nextID,
			Text:         fragment,
			Argmax:       argmaxID,
			LogitMax:     maxV,
			LogitMaxIdx:  int(argmaxID),
			LogitSummary: fmt.Sprintf("max=%.3f at idx %d", maxV, argmaxID),
		})
		ids = append(ids, nextID)
	}

	_ = startLen
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

// SampleGreedyKV runs autoregressive greedy decoding on top of an explicit
// KV cache, projecting Q/K/V only for the new token each step rather than
// recomputing the whole context. Returns the generated text (excluding the
// original prompt) and the full id sequence (prompt + generated). The K/V
// cache buffer shapes are fixed at maxContext positions so the WGSL kernel
// cache reuses pipelines across every step (no per-step recompilation).
//
// maxContext bounds the total context length (prompt + new tokens). It must
// be in (0, g.BlockSize] and at least len(prompt_ids) + 1. The cache shape
// is fixed at maxContext, regardless of how many tokens are actually drawn.
//
// Greedy decode only: this slice does not support temperature or top-k. The
// temperature / top-k path stays on the existing Sample function until a
// later slice unifies the two paths.
func SampleGreedyKV(g *nn.GPT, bpe *BPE, prompt string, maxContext, maxNewTokens int, device string) (string, []int32, error) {
	if maxContext <= 0 || maxContext > g.BlockSize {
		return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: maxContext %d must be in (0, %d]", maxContext, g.BlockSize)
	}
	if maxNewTokens <= 0 {
		return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: maxNewTokens must be > 0, got %d", maxNewTokens)
	}
	promptIds := bpe.Encode(prompt)
	if len(promptIds) == 0 {
		return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: prompt encoded to zero tokens (%q)", prompt)
	}
	if len(promptIds)+maxNewTokens > maxContext {
		return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: prompt len %d + maxNewTokens %d > maxContext %d",
			len(promptIds), maxNewTokens, maxContext)
	}

	cache := nn.NewKVCache(g.NLayer, g.NHead, g.NEmbd/g.NHead, maxContext)
	ids := make([]int32, 0, len(promptIds)+maxNewTokens)
	ids = append(ids, promptIds...)

	// Prefill: feed every prompt token through the KV step path and capture
	// the logits at the LAST prompt token (used to choose the first generated
	// id). Per-token Arena means each step rebuilds the parameter leaves and
	// the cache leaves; kernel sources are byte-identical so the compiler
	// cache reuses pipelines across steps.
	var lastLogits []float32
	for _, id := range promptIds {
		logits, err := runKVStep(g, cache, id, device)
		if err != nil {
			return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: prefill step %d: %w", cache.Pos, err)
		}
		lastLogits = logits
	}

	// Decode: greedy-pick from the last prefill logits, then loop. Each
	// decode step uses the just-picked id as the input to the next step.
	for n := 0; n < maxNewTokens; n++ {
		nextID := argmaxInt32(lastLogits)
		ids = append(ids, nextID)
		if n == maxNewTokens-1 {
			break
		}
		logits, err := runKVStep(g, cache, nextID, device)
		if err != nil {
			return "", nil, fmt.Errorf("gpt2: SampleGreedyKV: decode step %d: %w", n, err)
		}
		lastLogits = logits
	}

	return bpe.Decode(ids[len(promptIds):]), ids, nil
}

// runKVStep is the per-token GPU pass used by SampleGreedyKV. It builds the
// fresh-arena, reload-params graph, runs Realize twice (once for logits, once
// for the per-layer K_new / V_new buffers that the host-side cache update
// reads), copies K_new and V_new into the cache at slot Pos, then advances
// Pos. Returns the [Vocab]-shaped logits for the single-token output.
func runKVStep(g *nn.GPT, cache *nn.KVCache, id int32, device string) ([]float32, error) {
	a := uop.NewArena(1 << 22)
	for _, p := range g.Params() {
		p.Load(a)
	}
	idx := tensor.NewLeaf(a, []int64{1, 1}, uop.Dtypes.Int32, device)
	idx.SetData([]float32{math.Float32frombits(uint32(id))})

	logits, kNews, vNews := g.ForwardKVStep(idx, cache)
	if err := tensor.Realize(logits); err != nil {
		return nil, fmt.Errorf("realize logits: %w", err)
	}
	out := make([]float32, g.Vocab)
	copy(out, logits.Data())

	// Realize all 2*NLayer kNew/vNew buffers in a single batched call. The K/V
	// projection kernels are genuinely isomorphic (same shape AND structure), so
	// they cannot be told apart by the scheduler's structural-key order. The
	// durable fix (tensor/realize.go assignOutputs over CreateScheduleWithOutputs'
	// per-src output attribution) maps each tensor to ITS OWN output buffer by
	// node identity, so the batched call is now correct - and avoids the
	// 2*NLayer reschedules/step the old single-output workaround incurred.
	kvOutputs := make([]*tensor.Tensor, 0, 2*g.NLayer)
	kvOutputs = append(kvOutputs, kNews...)
	kvOutputs = append(kvOutputs, vNews...)
	if err := tensor.Realize(kvOutputs...); err != nil {
		return nil, fmt.Errorf("realize kv outputs: %w", err)
	}
	for li := 0; li < g.NLayer; li++ {
		cache.StoreLayerKV(li, kNews[li].Data(), vNews[li].Data())
	}
	cache.Advance()
	return out, nil
}

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
	return runSampleWithModel(w, g, bpe, device, prompt, opts, plain)
}

// runSampleWithModel is the post-load body of RunSampleCLI: it computes the
// context window, runs Sample, and renders the output. Factored out so tests
// can drive it with a tiny CPU-backed model instead of the ~550 MB GPT-2
// checkpoint LoadGPT2 fetches.
//
//nolint:errcheck // best-effort writes to stdout/stderr
func runSampleWithModel(w io.Writer, g *nn.GPT, bpe *BPE, device string, prompt string, opts SampleOptions, plain bool) error {
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
