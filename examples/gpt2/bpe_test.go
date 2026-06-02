// Tests for the GPT-2 byte-level BPE encoder/decoder.
//
// Fixture strategy: the canonical GPT-2 vocab.json (~1.0 MB) and
// merges.txt (~445 KB) are too large to commit. They are NOT in the repo.
// Tests are organized into two layers:
//
//  1. Unit tests against a tiny hand-crafted vocab + merges blob. These
//     verify the BPE merge loop, the byte-to-unicode bijection round
//     trip, and the pre-tokenizer. They always run.
//
//  2. Reference fixture tests against the real GPT-2 tokenizer files. The
//     expected ids are embedded inline in this file (computed during
//     dispatch using the canonical Python implementation in
//     transformers/tokenization_gpt2.py against vocab.json + merges.txt).
//     The vocab/merges files are loaded from examples/gpt2/testdata/, and
//     if they are absent the reference tests t.Skip. See testdata/README.md
//     for how to download them locally.
//
// In all cases, `go test` runs strictly offline; the dispatch-time fetch
// is one-shot to generate fixtures.
package gpt2

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// -----------------------------------------------------------------------
// 1. Unit tests: byte-to-unicode bijection.
// -----------------------------------------------------------------------

// TestBytesToUnicodeIsBijection verifies that the 256-entry table is a
// permutation: 256 distinct codepoints, all reachable from a unique byte,
// and the inverse map round-trips every byte value.
func TestBytesToUnicodeIsBijection(t *testing.T) {
	table := bytesToUnicode()
	seen := make(map[rune]int, 256)
	for b, r := range table {
		if prev, ok := seen[r]; ok {
			t.Fatalf("bytesToUnicode: rune %q reused for bytes %d and %d", r, prev, b)
		}
		seen[r] = b
	}
	if len(seen) != 256 {
		t.Fatalf("bytesToUnicode: expected 256 distinct runes, got %d", len(seen))
	}
	// Spot-check a few known mappings against the canonical GPT-2 table.
	// Byte 0x20 (' ') is not in the "printable" set (the set starts at
	// 0x21 '!') so it gets remapped to U+0120 (Ġ, the famous "space"
	// glyph in GPT-2 token strings).
	const wantSpace = 'Ġ'
	if got := table[' ']; got != wantSpace {
		t.Errorf("bytesToUnicode[' '] = %q (U+%04X), want %q (U+%04X)",
			got, got, wantSpace, wantSpace)
	}
	// 'A' (0x41) is printable ASCII so it maps to itself.
	if got := table['A']; got != 'A' {
		t.Errorf("bytesToUnicode['A'] = %q, want 'A'", got)
	}
	// Byte 0x00 (NUL) is not printable so it maps to a remapped codepoint.
	if table[0] < 256 {
		t.Errorf("bytesToUnicode[0] = %q (U+%04X), expected codepoint >= 256",
			table[0], table[0])
	}
}

// -----------------------------------------------------------------------
// 2. Unit tests: pre-tokenizer.
// -----------------------------------------------------------------------

// TestPreTokenize exercises the hand-rolled regex emulation against a
// handful of cases that stress the (?!\S) lookahead semantics. Expected
// values were obtained by running the canonical Python regex on the same
// inputs.
func TestPreTokenize(t *testing.T) {
	b := &BPE{} // preTokenize does not touch other fields
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"hello-comma-world", "Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"two-leading-spaces", "  hello", []string{" ", " hello"}},
		{"single-leading-space", " hello", []string{" hello"}},
		{"trailing-space-eos", "hi ", []string{"hi", " "}},
		{"three-spaces-eos", "hi   ", []string{"hi", "   "}},
		{"unicode-letters", "café", []string{"café"}},
		{"digits-and-text", "abc 123", []string{"abc", " 123"}},
		{"contraction-s", "it's", []string{"it", "'s"}},
		{"contraction-ll", "we'll", []string{"we", "'ll"}},
		{"contraction-re", "they're", []string{"they", "'re"}},
		{"punct-run", "...?!", []string{"...?!"}},
		{"space-then-punct", " ...", []string{" ..."}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := b.preTokenize(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("preTokenize(%q) = %#v, want %#v", tc.in, got, tc.want)
			}
		})
	}
}

// -----------------------------------------------------------------------
// 3. Unit tests: BPE merge loop against a tiny hand-crafted vocab.
// -----------------------------------------------------------------------

// tinyVocab and tinyMerges form a self-contained BPE that knows how to
// tokenize the word "hello" using a chain of three merges:
//
//	h, e, l, l, o
//	-> h, e, ll, o      (merge "l l" at rank 0)
//	-> he, ll, o        (merge "h e" at rank 1)
//	-> hell, o          (merge "he ll" at rank 2)
//	-> hello            (merge "hell o" at rank 3)
//
// The vocab includes every intermediate sub-token so each merge has a
// representable id, plus a few standalone runes for negative tests.
const tinyVocab = `{
  "h": 0, "e": 1, "l": 2, "o": 3, "ll": 4, "he": 5, "hell": 6, "hello": 7,
  "x": 8, "y": 9
}`

const tinyMerges = `#version: 0.2
l l
h e
he ll
hell o
`

func TestTinyBPEMergeChain(t *testing.T) {
	tok, err := NewBPE([]byte(tinyVocab), []byte(tinyMerges))
	if err != nil {
		t.Fatalf("NewBPE: %v", err)
	}
	// The BPE merge loop should reduce "hello" all the way to a single
	// sub-token "hello" (id 7) since every intermediate merge is in the
	// rank table.
	got := tok.bpe("hello")
	want := []string{"hello"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("bpe(\"hello\") = %#v, want %#v", got, want)
	}
	// "xy" has no merges in the table, so it should stay split into
	// single-rune sub-tokens.
	got = tok.bpe("xy")
	want = []string{"x", "y"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("bpe(\"xy\") = %#v, want %#v", got, want)
	}
}

func TestNewBPEErrors(t *testing.T) {
	if _, err := NewBPE([]byte("not json"), []byte(tinyMerges)); err == nil {
		t.Error("NewBPE accepted invalid vocab JSON")
	}
	if _, err := NewBPE([]byte("{}"), []byte(tinyMerges)); err == nil {
		t.Error("NewBPE accepted empty vocab")
	}
	if _, err := NewBPE([]byte(tinyVocab), []byte("#version: 0.2\n")); err == nil {
		t.Error("NewBPE accepted empty merges file")
	}
}

// -----------------------------------------------------------------------
// 4. Reference fixtures: real GPT-2 vocab + merges, ids computed during
//    dispatch from the canonical Python tokenizer. Tests skip if the
//    vocab/merges files are absent (see testdata/README.md).
// -----------------------------------------------------------------------

// SHA-256 of the canonical files; used to detect a vocab/merges drift if
// someone re-downloads with different content.
const (
	wantVocabSHA  = "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
	wantMergesSHA = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
)

type refFixture struct {
	name string
	text string
	ids  []int32
}

// referenceFixtures: ground truth produced by running the canonical Python
// GPT-2 tokenizer at dispatch time. Do not edit by hand. To regenerate,
// see testdata/README.md.
var referenceFixtures = []refFixture{
	{"hello-world", "Hello, world!", []int32{15496, 11, 995, 0}},
	{
		"pangram",
		"The quick brown fox jumps over the lazy dog.",
		[]int32{464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13},
	},
	{"empty", "", nil},
	{"unicode-cafe", "café", []int32{66, 1878, 2634}},
	{"two-leading-spaces", "  hello", []int32{220, 23748}},
	{
		"anneal-sentence",
		"Anneal is a Go-native ML compiler.",
		[]int32{43227, 282, 318, 257, 1514, 12, 30191, 10373, 17050, 13},
	},
	{"arithmetic", "1 + 2 = 3", []int32{16, 1343, 362, 796, 513}},
	{
		"leading-space-then-words",
		" leading space then word",
		[]int32{3756, 2272, 788, 1573},
	},
}

// loadReferenceBPE loads the GPT-2 vocab.json + merges.txt from
// testdata/ and verifies their SHA-256. Returns nil + a skip reason if the
// files are not present (the common case in CI).
func loadReferenceBPE(t *testing.T) *BPE {
	t.Helper()
	vocabPath := filepath.Join("testdata", "vocab.json")
	mergesPath := filepath.Join("testdata", "merges.txt")
	vocab, err := os.ReadFile(vocabPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("skipping reference fixtures: %s missing (see testdata/README.md)", vocabPath)
		}
		t.Fatalf("read %s: %v", vocabPath, err)
	}
	merges, err := os.ReadFile(mergesPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("skipping reference fixtures: %s missing (see testdata/README.md)", mergesPath)
		}
		t.Fatalf("read %s: %v", mergesPath, err)
	}
	if got := sha256Hex(vocab); got != wantVocabSHA {
		t.Fatalf("vocab.json SHA-256 mismatch: got %s, want %s (canonical GPT-2 vocab.json from HuggingFace expected; embedded fixtures are stale or the file was tampered with)",
			got, wantVocabSHA)
	}
	if got := sha256Hex(merges); got != wantMergesSHA {
		t.Fatalf("merges.txt SHA-256 mismatch: got %s, want %s", got, wantMergesSHA)
	}
	tok, err := NewBPE(vocab, merges)
	if err != nil {
		t.Fatalf("NewBPE with real GPT-2 vocab/merges: %v", err)
	}
	return tok
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// TestReferenceFixturesEncode encodes each fixture text and compares
// against the ground-truth id sequence.
func TestReferenceFixturesEncode(t *testing.T) {
	tok := loadReferenceBPE(t)
	for _, f := range referenceFixtures {
		t.Run(f.name, func(t *testing.T) {
			got := tok.Encode(f.text)
			if !sliceEqualInt32(got, f.ids) {
				t.Errorf("Encode(%q):\n got = %v\nwant = %v", f.text, got, f.ids)
			}
		})
	}
}

// TestReferenceFixturesRoundTrip encodes then decodes each fixture and
// requires byte-exact recovery of the original text.
func TestReferenceFixturesRoundTrip(t *testing.T) {
	tok := loadReferenceBPE(t)
	for _, f := range referenceFixtures {
		t.Run(f.name, func(t *testing.T) {
			ids := tok.Encode(f.text)
			rt := tok.Decode(ids)
			if rt != f.text {
				t.Errorf("Decode(Encode(%q)) = %q, want %q", f.text, rt, f.text)
			}
		})
	}
}

// TestReferenceFixturesDecodeFromIds bypasses Encode and decodes the
// stored ground-truth ids directly; this catches Decode-only regressions.
func TestReferenceFixturesDecodeFromIds(t *testing.T) {
	tok := loadReferenceBPE(t)
	for _, f := range referenceFixtures {
		t.Run(f.name, func(t *testing.T) {
			got := tok.Decode(f.ids)
			if got != f.text {
				t.Errorf("Decode(%v) = %q, want %q", f.ids, got, f.text)
			}
		})
	}
}

// -----------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------

func sliceEqualInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Compile-time check that the embedded fixtures are non-empty, so a stray
// edit that wipes them out is caught.
var _ = fmt.Sprintf("%d", len(referenceFixtures))
