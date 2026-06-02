package examples

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/georgebuilds/anneal/internal/assets"
)

// ── Char-level dataset for the nanoGPT example ───────────────────────────────
//
// The dataset is the canonical tinyshakespeare corpus (registered in
// internal/assets as "shakespeare"). On first use the asset is downloaded
// and SHA-verified into the project cache; subsequent calls hit the cache.
//
// Tokenization is sorted-unique characters from the corpus, yielding a small
// vocabulary (~65 entries for tinyshakespeare). The whole corpus is encoded
// once to a []int32 buffer; training samples random sequential windows from
// that buffer.
//
// For tests, callers can construct a charDataset directly from an in-memory
// fixture string, avoiding any network or filesystem dependency.

// charDataset is a char-level encoded corpus plus its vocabulary.
type charDataset struct {
	// Vocab is the sorted unique character set; index in this slice is the
	// token id. Vocab[i] is the i-th token's rune value.
	Vocab []rune
	// charToIdx maps a rune to its token id.
	charToIdx map[rune]int32
	// Data holds the encoded corpus as a sequence of token ids.
	Data []int32
}

// newCharDatasetFromString builds a charDataset from an in-memory UTF-8
// string. Useful for tests and the registry-level Build path that does not
// touch the asset cache.
func newCharDatasetFromString(text string) *charDataset {
	if text == "" {
		panic("examples: newCharDatasetFromString: empty corpus")
	}
	seen := map[rune]bool{}
	for _, r := range text {
		seen[r] = true
	}
	vocab := make([]rune, 0, len(seen))
	for r := range seen {
		vocab = append(vocab, r)
	}
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })

	c2i := make(map[rune]int32, len(vocab))
	for i, r := range vocab {
		c2i[r] = int32(i)
	}

	data := make([]int32, 0, len(text))
	for _, r := range text {
		data = append(data, c2i[r])
	}
	return &charDataset{Vocab: vocab, charToIdx: c2i, Data: data}
}

// loadShakespeareDataset resolves the cached tinyshakespeare corpus and
// returns a charDataset. On first call this triggers an HTTP fetch via
// internal/assets; subsequent calls hit the disk cache. Honours
// ANNEAL_OFFLINE=1: when set, missing cache returns the error from
// assets.Get without attempting a network call.
func loadShakespeareDataset() (*charDataset, error) {
	path, err := assets.Get("shakespeare")
	if err != nil {
		return nil, fmt.Errorf("nanogpt: fetch shakespeare asset: %w", err)
	}
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nanogpt: read shakespeare asset %s: %w", path, err)
	}
	return newCharDatasetFromString(string(bytes)), nil
}

// VocabSize returns the number of distinct tokens in the dataset.
func (d *charDataset) VocabSize() int { return len(d.Vocab) }

// Encode maps each rune of s to its token id. Runes not present in the
// vocabulary are silently dropped; callers handing in foreign text should
// not assume round-tripping.
func (d *charDataset) Encode(s string) []int32 {
	out := make([]int32, 0, len(s))
	for _, r := range s {
		if id, ok := d.charToIdx[r]; ok {
			out = append(out, id)
		}
	}
	return out
}

// Decode reverses Encode by looking up each token id in Vocab.
func (d *charDataset) Decode(ids []int32) string {
	out := make([]rune, 0, len(ids))
	for _, id := range ids {
		if id < 0 || int(id) >= len(d.Vocab) {
			continue
		}
		out = append(out, d.Vocab[id])
	}
	return string(out)
}

// SampleBatch draws a random training batch from the corpus. It returns
// xs of length batch*blockSize and ys of the same length, where xs[i] is
// the input window and ys[i] is the next-token target (input shifted by
// one). The two buffers are laid out row-major as [B, T] int32.
//
// The sampler picks B independent uniform start positions in
// [0, len(Data) - blockSize - 1]. When the corpus is shorter than
// blockSize+1, SampleBatch returns nil, nil.
func (d *charDataset) SampleBatch(rng *rand.Rand, batch, blockSize int) (xs, ys []int32) {
	if len(d.Data) < blockSize+1 {
		return nil, nil
	}
	maxStart := len(d.Data) - blockSize - 1
	xs = make([]int32, batch*blockSize)
	ys = make([]int32, batch*blockSize)
	for b := 0; b < batch; b++ {
		start := rng.Intn(maxStart + 1)
		copy(xs[b*blockSize:(b+1)*blockSize], d.Data[start:start+blockSize])
		copy(ys[b*blockSize:(b+1)*blockSize], d.Data[start+1:start+1+blockSize])
	}
	return xs, ys
}

// ── int32 leaf packing ───────────────────────────────────────────────────────
//
// Integer leaf tensors carry their data as float32 bit patterns (the
// dispatch path reads the same bytes back as i32). Mirrors the helper used
// in tensor/nn/embedding_test.idxBitsForLeaf and tensor/nn/gpt_test.gptIdxBitsForLeaf.

// int32sAsBits packs a []int32 into a []float32 by reinterpreting each
// element's bits. The result is what NewLeaf(..., Int32, ...).SetData expects.
func int32sAsBits(vs []int32) []float32 {
	out := make([]float32, len(vs))
	for i, v := range vs {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

// oneHotBits returns a [n, vocab] flat row-major float32 buffer where row i
// has a single 1.0 at column ids[i] and zeros elsewhere. Used for the
// cross-entropy loss via one-hot @ log_softmax (sidesteps gather on the
// last vocab axis, which is not in the Slice D scope).
func oneHotBits(ids []int32, vocab int) []float32 {
	out := make([]float32, len(ids)*vocab)
	for i, id := range ids {
		if id < 0 || int(id) >= vocab {
			continue
		}
		out[i*vocab+int(id)] = 1.0
	}
	return out
}
