package gpt2

import (
	"math"
	"math/rand"
	"testing"
)

// TestBPEEmptyToken covers the `token == ""` short-circuit in bpe, which
// returns nil. The pre-tokenizer never produces an empty piece, so this
// branch is only reachable via a direct call.
func TestBPEEmptyToken(t *testing.T) {
	tok, err := NewBPE([]byte(tinyVocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	if got := tok.bpe(""); got != nil {
		t.Errorf("bpe(\"\") = %v, want nil", got)
	}
}

// TestSampleFromLogitsAllNegInf drives the degenerate all-(-inf) input. With
// every logit -inf the stable-softmax computes v-maxV = (-inf)-(-inf) = NaN,
// so exp() is NaN, sum is NaN (the `sum == 0` guard does NOT fire — NaN != 0),
// and the cumulative-sample comparison r <= acc is never true (NaN compares
// false), so the function returns the trailing FP-slack index len-1. This
// pins the documented fall-through behavior for pathological inputs.
func TestSampleFromLogitsAllNegInf(t *testing.T) {
	neg := float32(math.Inf(-1))
	logits := []float32{neg, neg, neg}
	got := sampleFromLogits(logits, 1.0, 0, rand.New(rand.NewSource(1)))
	if int(got) != len(logits)-1 {
		t.Errorf("all -inf logits: got %d, want %d (FP-slack fallthrough)", got, len(logits)-1)
	}
}

// maxFloatSource is a math/rand source whose Float64 yields the largest value
// strictly below 1.0, so the multinomial-sample cumulative loop lands on the
// final bucket (exercising the trailing FP-slack return).
//
// It must NOT return 1<<63 - 1: that rounds to float64(1<<63), making
// rand.Float64() compute exactly 1.0, which trips its internal `goto again`
// retry loop and spins forever. Subtracting 1<<20 (>> the 2^11 double spacing
// near 2^63) keeps the quotient just under 1.0 with no retry.
type maxFloatSource struct{}

func (maxFloatSource) Int63() int64 { return 1<<63 - 1<<20 }
func (maxFloatSource) Seed(int64)   {}

// TestSampleFromLogitsFPSlackFallback covers the trailing
// `return int32(len(probs) - 1)` fallback. With Float64() == ~1.0 we have
// r ≈ sum; accumulated FP rounding can leave acc strictly below r at the end
// of the loop, so the function returns the last index. Even if the loop does
// catch the last index normally, the result must be the final index, which is
// what we assert.
func TestSampleFromLogitsFPSlackFallback(t *testing.T) {
	rng := rand.New(maxFloatSource{})
	// Several near-equal logits so the cumulative sum spans the whole range;
	// r ≈ sum lands on (or just past) the last bucket.
	logits := []float32{0.0, 0.0, 0.0, 0.0}
	got := sampleFromLogits(logits, 1.0, 0, rng)
	if int(got) != len(logits)-1 {
		t.Errorf("FP-slack sample: got %d, want %d (last index)", got, len(logits)-1)
	}
}

// TestSampleFromLogitsTopKWiderThanVocab covers the `topK >= len(scaled)`
// path where the top-k filter is skipped entirely (every token eligible).
func TestSampleFromLogitsTopKWiderThanVocab(t *testing.T) {
	logits := []float32{0.1, 5.0, 0.2}
	// topK 10 > len 3 -> filter skipped; argmax still dominates probability.
	got := sampleFromLogits(logits, 0.0001, 10, rand.New(rand.NewSource(3)))
	if got != 1 {
		t.Errorf("topK>vocab with tiny temperature: got %d, want 1", got)
	}
}
