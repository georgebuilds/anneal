package gpt2

import (
	"reflect"
	"strings"
	"testing"
)

// ── parseMerges branch coverage ─────────────────────────────────────────────

// TestParseMergesNoHeader covers the "first line is not a #version header"
// branch: the first line is treated as the first merge rule rather than
// skipped.
func TestParseMergesNoHeader(t *testing.T) {
	ranks, err := parseMerges([]byte("a b\nc d\n"))
	if err != nil {
		t.Fatalf("parseMerges: %v", err)
	}
	if got, ok := ranks[[2]string{"a", "b"}]; !ok || got != 0 {
		t.Errorf("merge (a,b) rank = %d, ok=%v; want rank 0", got, ok)
	}
	if got, ok := ranks[[2]string{"c", "d"}]; !ok || got != 1 {
		t.Errorf("merge (c,d) rank = %d, ok=%v; want rank 1", got, ok)
	}
}

// TestParseMergesBlankLinesSkipped covers the `line == ""` continue branch:
// blank lines between merges are ignored without advancing the rank.
func TestParseMergesBlankLinesSkipped(t *testing.T) {
	ranks, err := parseMerges([]byte("#version: 0.2\na b\n\nc d\n"))
	if err != nil {
		t.Fatalf("parseMerges: %v", err)
	}
	if got := ranks[[2]string{"c", "d"}]; got != 1 {
		t.Errorf("merge (c,d) rank = %d, want 1 (blank line must not bump rank)", got)
	}
}

// TestParseMergesBadLine covers the `len(parts) != 2` error branch: a merge
// line with no space is malformed.
func TestParseMergesBadLine(t *testing.T) {
	_, err := parseMerges([]byte("#version: 0.2\nsingletoken\n"))
	if err == nil {
		t.Fatal("parseMerges accepted a merge line with no space")
	}
	if !strings.Contains(err.Error(), "bad merge line") {
		t.Errorf("error = %v, want bad-merge-line message", err)
	}
}

// TestParseMergesScannerError covers the `sc.Err() != nil` branch: a single
// line longer than the 1 MiB scanner buffer makes bufio.Scanner fail with
// ErrTooLong, which parseMerges surfaces as a read error.
func TestParseMergesScannerError(t *testing.T) {
	// 2 MiB of non-newline bytes on the first line (no header) overflows the
	// scanner's max-token buffer (1 MiB).
	huge := make([]byte, 2*1024*1024)
	for i := range huge {
		huge[i] = 'a'
	}
	_, err := parseMerges(huge)
	if err == nil || !strings.Contains(err.Error(), "read merges.txt") {
		t.Errorf("oversized merge line: got %v, want read-merges error", err)
	}
}

// ── matchContraction branch coverage ────────────────────────────────────────

// TestMatchContraction exercises every return path: 3-char hit, 2-char hit,
// apostrophe followed by a non-contraction (falls through to ""), and a
// non-apostrophe start.
func TestMatchContraction(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		i       int
		wantTok string
		wantN   int
	}{
		{"3char-re", "'re", 0, "'re", 3},
		{"3char-ve", "'ve", 0, "'ve", 3},
		{"3char-ll", "'ll", 0, "'ll", 3},
		{"2char-s", "'s", 0, "'s", 2},
		{"2char-t", "'t", 0, "'t", 2},
		{"2char-m", "'m", 0, "'m", 2},
		{"2char-d", "'d", 0, "'d", 2},
		// apostrophe but the following chars are not a contraction (3-char
		// "'xy" misses the s3 switch, then 2-char "'x" misses the s2 switch).
		{"apostrophe-nonmatch", "'xy", 0, "", 0},
		// apostrophe at the very end of input: rem==1, neither block fires.
		{"apostrophe-eos", "'", 0, "", 0},
		// not an apostrophe at i.
		{"not-apostrophe", "abc", 0, "", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tok, n := matchContraction([]rune(tc.in), tc.i)
			if tok != tc.wantTok || n != tc.wantN {
				t.Errorf("matchContraction(%q, %d) = (%q, %d), want (%q, %d)",
					tc.in, tc.i, tok, n, tc.wantTok, tc.wantN)
			}
		})
	}
}

// ── matchWhitespace branch coverage ─────────────────────────────────────────

// TestMatchWhitespace covers all four exits of matchWhitespace:
//   - non-space start -> ("", 0)
//   - whitespace run to end of input -> whole run
//   - single whitespace before non-space -> ("", 0)
//   - multi whitespace before non-space -> run minus one
func TestMatchWhitespace(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		i       int
		wantTok string
		wantN   int
	}{
		{"non-space", "abc", 0, "", 0},
		{"run-to-eos", "  ", 0, "  ", 2},
		{"single-space-then-letter", " a", 0, "", 0},
		{"multi-space-then-letter", "   a", 0, "  ", 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tok, n := matchWhitespace([]rune(tc.in), tc.i)
			if tok != tc.wantTok || n != tc.wantN {
				t.Errorf("matchWhitespace(%q, %d) = (%q, %d), want (%q, %d)",
					tc.in, tc.i, tok, n, tc.wantTok, tc.wantN)
			}
		})
	}
}

// TestPreTokenizeWhitespaceOnly drives preTokenize through a whitespace-only
// input so the matchWhitespace end-of-input branch is reached via the public
// path (defends against the helper and caller drifting apart).
func TestPreTokenizeWhitespaceOnly(t *testing.T) {
	b := &BPE{}
	got := b.preTokenize("   ")
	want := []string{"   "}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("preTokenize(%q) = %#v, want %#v", "   ", got, want)
	}
}
