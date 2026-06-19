package examples

// Production entrypoints for GPT-2 fine-tuning that depend on the ~548 MB
// HuggingFace checkpoint and the tinyshakespeare corpus. These are dominated by
// asset download + a GPU realize loop that cannot run in the GPU-less CI runner
// (a single full GPT-2 step on the pure-Go CPU interpreter is impractical), so
// this file is excluded from the coverage signal in codecov.yml. The reusable
// training core they call (runGPT2Finetune, gpt2StableCrossEntropy,
// clipGradsByGlobalNorm, evalGPT2Loss, greedySampleGPT2, sampleTokenBatch) lives
// in gpt2_finetune.go and stays measured.

import (
	"fmt"
	"os"

	"github.com/georgebuilds/anneal/examples/gpt2"
	"github.com/georgebuilds/anneal/internal/assets"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// buildGPT2Finetune constructs a single forward graph at the fine-tune shape so
// `anneal graph` / `anneal kernels` can inspect it without a training run. It
// loads the real tied GPT-2 (priming the asset cache); ANNEAL_OFFLINE with a
// cold cache surfaces a clear error.
func buildGPT2Finetune(device string) (*BuildResult, error) {
	a := uop.NewArena(1 << 20)
	g, _, err := gpt2.LoadGPT2(a, device)
	if err != nil {
		return nil, err
	}
	for _, p := range g.Params() {
		p.Load(a)
	}

	const B = int64(1)
	T := gpt2FinetuneSeqLen
	idxVals := make([]int32, B*T) // all-zero token ids: valid indices in [0, vocab)
	idx := tensor.NewLeaf(a, []int64{B, T}, uop.Dtypes.Int32, device)
	idx.SetData(int32sAsBits(idxVals))

	logits := g.Forward(idx)

	leaves := make([]*tensor.Tensor, 0, len(g.Params()))
	for _, p := range g.Params() {
		leaves = append(leaves, p.T)
	}
	return &BuildResult{Arena: a, Output: logits, Device: device, Leaves: leaves}, nil
}

// trainGPT2Finetune is the production entry: load the real tied GPT-2-small +
// BPE, tokenize tinyshakespeare, and fine-tune via the shared runGPT2Finetune
// core.
func trainGPT2Finetune(device string, cfg TrainConfig, logFn func(step int, loss float32)) error {
	a0 := uop.NewArena(1 << 14)
	g, bpe, err := gpt2.LoadGPT2(a0, device)
	if err != nil {
		return fmt.Errorf("gpt2 finetune: load model: %w", err)
	}

	corpus, err := loadShakespeareText()
	if err != nil {
		return fmt.Errorf("gpt2 finetune: load corpus: %w", err)
	}
	tokens := bpe.Encode(corpus)
	if int64(len(tokens)) < gpt2FinetuneSeqLen+1 {
		return fmt.Errorf("gpt2 finetune: corpus too short after BPE (%d tokens)", len(tokens))
	}

	gptCfg := gpt2FinetuneConfig{
		Vocab:     gpt2.GPT2Vocab,
		NLayer:    gpt2.GPT2NLayer,
		NHead:     gpt2.GPT2NHead,
		NEmbd:     gpt2.GPT2NEmbd,
		BlockSize: gpt2.GPT2BlockSize,
		SeqLen:    gpt2FinetuneSeqLen,
		SampleN:   gpt2FinetuneSampleN,
	}
	encode := func(s string) []int32 { return bpe.Encode(s) }
	decode := func(ids []int32) string { return bpe.Decode(ids) }
	return runGPT2Finetune(device, cfg, logFn, g, tokens, gptCfg, encode, decode, 42)
}

// loadShakespeareText resolves the SHA-pinned tinyshakespeare corpus via
// internal/assets and returns it as raw text for BPE tokenization. (nanoGPT's
// loadShakespeareDataset returns a char-level dataset; GPT-2 needs the bytes.)
func loadShakespeareText() (string, error) {
	path, err := assets.Get("shakespeare")
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read shakespeare corpus %s: %w", path, err)
	}
	return string(b), nil
}
