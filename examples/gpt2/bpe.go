// Package gpt2 contains a pure-Go, byte-level BPE encoder/decoder that
// matches the canonical GPT-2 tokenizer (the one shipped in
// huggingface/transformers' tokenization_gpt2.py).
//
// The tokenizer has three stages:
//
//  1. bytes_to_unicode: a fixed bijection from each of the 256 byte values
//     to a printable Unicode codepoint. This lets the merge process work in
//     a string domain without worrying about unprintable control bytes.
//
//  2. Pre-tokenization: a regular expression splits the input UTF-8 text
//     into a sequence of pre-tokens. The canonical GPT-2 pattern is
//     "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".
//     Go's regexp engine (RE2) does not support the (?!\S) lookahead, so the
//     pre-tokenizer here is hand-rolled and matches the canonical behavior
//     exactly. See preTokenize for details.
//
//  3. BPE merge loop: each pre-token (after byte_encoder substitution) is
//     run through the rank-ordered pair-merge loop until no merge applies,
//     and the resulting sub-tokens are mapped to ids via vocab.json.
//
// Decode reverses these stages: id -> token string -> bytes (via the inverse
// bijection) -> UTF-8 text.
package gpt2

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode"
)

// BPE is a GPT-2-compatible byte-level BPE tokenizer.
type BPE struct {
	encoder     map[string]int32 // token string -> id
	decoder     map[int32]string // id -> token string
	bpeRanks    map[[2]string]int
	byteEncoder [256]rune // byte -> unicode codepoint (bytes_to_unicode)
	byteDecoder map[rune]byte
}

// NewBPE constructs a tokenizer from in-memory vocab.json bytes and merges.txt
// bytes. The expected formats are:
//
//   - vocabJSON: a single JSON object mapping token strings to integer ids.
//     This is exactly the file GPT-2 ships as vocab.json.
//
//   - mergesTxt: a UTF-8 text file whose first line is a "#version: ..."
//     header, followed by one merge per line. Each merge line is two
//     space-separated tokens. The line index (after the header) is the
//     merge's rank; lower rank = applied earlier.
//
// Passing bytes instead of paths keeps the constructor decoupled from the
// asset-loading layer and trivially testable without filesystem access.
func NewBPE(vocabJSON, mergesTxt []byte) (*BPE, error) {
	var encoder map[string]int32
	if err := json.Unmarshal(vocabJSON, &encoder); err != nil {
		return nil, fmt.Errorf("gpt2: parse vocab.json: %w", err)
	}
	if len(encoder) == 0 {
		return nil, fmt.Errorf("gpt2: vocab.json is empty")
	}
	decoder := make(map[int32]string, len(encoder))
	for tok, id := range encoder {
		decoder[id] = tok
	}

	ranks, err := parseMerges(mergesTxt)
	if err != nil {
		return nil, err
	}

	be := bytesToUnicode()
	bd := make(map[rune]byte, 256)
	for b, r := range be {
		bd[r] = byte(b)
	}

	return &BPE{
		encoder:     encoder,
		decoder:     decoder,
		bpeRanks:    ranks,
		byteEncoder: be,
		byteDecoder: bd,
	}, nil
}

// parseMerges parses a GPT-2 merges.txt blob. The first line is a version
// header which we skip. Subsequent non-empty lines each contain one merge
// rule formatted as "A B" (two tokens separated by a single space).
func parseMerges(mergesTxt []byte) (map[[2]string]int, error) {
	ranks := make(map[[2]string]int)
	sc := bufio.NewScanner(bytes.NewReader(mergesTxt))
	// merges.txt can have very long lines for unusual tokens; bump the
	// scanner buffer so we don't hit bufio.ErrTooLong on pathological inputs.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	rank := 0
	first := true
	for sc.Scan() {
		line := sc.Text()
		if first {
			first = false
			if strings.HasPrefix(line, "#") {
				continue
			}
			// No header line; treat this line as the first merge.
		}
		if line == "" {
			continue
		}
		// Each merge is exactly two tokens separated by a single space.
		// SplitN(..., 2) is intentional: tokens themselves never contain a
		// space (the byte_encoder bijection maps 0x20 to a non-space rune).
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("gpt2: bad merge line %d: %q", rank, line)
		}
		ranks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("gpt2: read merges.txt: %w", err)
	}
	if len(ranks) == 0 {
		return nil, fmt.Errorf("gpt2: merges.txt has no merge rules")
	}
	return ranks, nil
}

// bytesToUnicode returns the canonical GPT-2 byte-to-rune bijection: a
// permutation of the 256 byte values onto 256 distinct printable Unicode
// codepoints. The construction is exactly the one in tokenization_gpt2.py:
// printable ASCII and Latin-1 codepoints map to themselves; the remaining
// byte values are assigned to codepoints starting at 256, in order.
func bytesToUnicode() [256]rune {
	// Step 1: gather the "already printable" byte ranges. These bytes map to
	// themselves as runes.
	var printable []int
	add := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			printable = append(printable, b)
		}
	}
	add('!', '~') // 0x21..0x7E
	add('¡', '¬') // 0xA1..0xAC
	add('®', 'ÿ') // 0xAE..0xFF

	inPrintable := make(map[int]bool, len(printable))
	for _, b := range printable {
		inPrintable[b] = true
	}

	// Step 2: every remaining byte value gets mapped to a fresh codepoint
	// starting at U+0100 (256), in ascending byte order.
	var table [256]rune
	for _, b := range printable {
		table[b] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if inPrintable[b] {
			continue
		}
		table[b] = rune(256 + n)
		n++
	}
	return table
}

// preTokenize implements the canonical GPT-2 regex
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// as a hand-rolled scanner. Go's RE2 lacks lookahead, so we cannot just feed
// this to regexp.MustCompile; the (?!\S) branch matters for trailing-
// whitespace handling at end of input and at word boundaries.
//
// At each position we try the alternatives in order and take the first
// match. The contractions (top of the alternation) are case-sensitive and
// only fire at apostrophes.
func (b *BPE) preTokenize(text string) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var out []string
	i := 0
	for i < len(runes) {
		if tok, n := matchContraction(runes, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := matchOptSpaceClass(runes, i, isLetter); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := matchOptSpaceClass(runes, i, isNumber); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := matchOptSpaceClass(runes, i, isOther); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		if tok, n := matchWhitespace(runes, i); n > 0 {
			out = append(out, tok)
			i += n
			continue
		}
		// Defensive: a single rune that matched nothing above should not
		// happen for well-formed UTF-8 input, but skip it rather than loop.
		out = append(out, string(runes[i]))
		i++
	}
	return out
}

// matchContraction matches one of the canonical English contractions
// ('s, 't, 're, 've, 'm, 'll, 'd) starting at position i. The contractions
// are tried longest-first so 're beats 'r-... matches.
func matchContraction(r []rune, i int) (string, int) {
	if r[i] != '\'' {
		return "", 0
	}
	rem := len(r) - i
	// Order matters only for the 2-char vs 3-char distinction; check
	// 3-char first so 're/'ve/'ll win over their 2-char prefixes (which
	// don't exist here, but the principle still applies).
	if rem >= 3 {
		s3 := string(r[i : i+3])
		switch s3 {
		case "'re", "'ve", "'ll":
			return s3, 3
		}
	}
	if rem >= 2 {
		s2 := string(r[i : i+2])
		switch s2 {
		case "'s", "'t", "'m", "'d":
			return s2, 2
		}
	}
	return "", 0
}

// matchOptSpaceClass matches " ?CLASS+" where CLASS is one of the unicode
// categories. The leading space is optional and only consumed if a CLASS
// rune follows it.
func matchOptSpaceClass(r []rune, i int, cls func(rune) bool) (string, int) {
	start := i
	if r[i] == ' ' {
		// Try to consume the space, but only if there is at least one CLASS
		// rune immediately after it.
		if i+1 < len(r) && cls(r[i+1]) {
			i++
		}
	}
	if i >= len(r) || !cls(r[i]) {
		return "", 0
	}
	j := i
	for j < len(r) && cls(r[j]) {
		j++
	}
	return string(r[start:j]), j - start
}

// matchWhitespace implements "\s+(?!\S)|\s+". With greedy "\s+", the
// (?!\S) form only ever matches when the whitespace run extends to end of
// input (since after greedy consumption there is nothing else to look at,
// and any non-whitespace rune would have to follow). The canonical regex
// engine handles this with backtracking: when followed by a non-whitespace
// rune, the first alternative trims one whitespace off the end so that the
// last whitespace can be picked up by the next " ?CLASS+" branch.
//
// Concretely: a whitespace run of length L
//   - at end of input: emit one token of length L.
//   - otherwise: emit one token of length L-1 (the "prefix" run), letting
//     the next iteration consume the trailing single whitespace as the
//     " ?" prefix of a CLASS token. If L == 1, the loop emits no token here
//     and the next iteration's " ?CLASS+" consumes the single space.
func matchWhitespace(r []rune, i int) (string, int) {
	if !isSpace(r[i]) {
		return "", 0
	}
	j := i
	for j < len(r) && isSpace(r[j]) {
		j++
	}
	runLen := j - i
	if j == len(r) {
		// End of input: take the whole run.
		return string(r[i:j]), runLen
	}
	// Followed by a non-whitespace rune; leave the last whitespace to be
	// consumed as the optional leading space of the next CLASS token.
	if runLen == 1 {
		// Single whitespace before a non-whitespace rune; produce no
		// whitespace token, letting the next branch (e.g. " ?\p{L}+") pick
		// up the space as its leading " ?".
		return "", 0
	}
	return string(r[i : j-1]), runLen - 1
}

// isLetter / isNumber / isOther / isSpace mirror the regex character
// classes \p{L}, \p{N}, [^\s\p{L}\p{N}], and \s respectively.
func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }
func isSpace(r rune) bool  { return unicode.IsSpace(r) }
func isOther(r rune) bool {
	return !isSpace(r) && !isLetter(r) && !isNumber(r)
}

// Encode tokenizes text into a sequence of GPT-2 BPE ids. The pipeline is
// pre-tokenize -> byte_encoder substitution -> BPE merge -> vocab lookup.
// Any pre-token piece that is not present in vocab.json (which should be
// impossible for a well-formed GPT-2 vocab) is silently skipped; this
// matches the behavior of feeding unknown sub-tokens to encoder.get with
// no fallback in Python (an unknown token would KeyError there, here we
// drop it to keep the API total).
func (b *BPE) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	preTokens := b.preTokenize(text)
	var ids []int32
	for _, pt := range preTokens {
		// Apply the byte-to-unicode bijection over the UTF-8 bytes of the
		// pre-token. The result is a string of runes drawn from the
		// 256-rune image of the bijection.
		var sb strings.Builder
		sb.Grow(len(pt))
		for i := 0; i < len(pt); i++ {
			sb.WriteRune(b.byteEncoder[pt[i]])
		}
		mapped := sb.String()
		for _, sub := range b.bpe(mapped) {
			if id, ok := b.encoder[sub]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// bpe runs the rank-ordered pair-merge loop on a single byte-encoded token,
// returning the final sub-token sequence. The algorithm:
//
//   - Start with the token split into individual runes.
//   - Repeatedly find the bigram (adjacent pair) with the lowest rank in
//     bpeRanks; if none of the current bigrams are in the table, stop.
//   - Merge ALL non-overlapping occurrences of that bigram in a single pass
//     (this matches the canonical implementation's outer-loop semantics).
//   - Continue until the token is a single sub-token or no merges apply.
//
// The output is the slice of sub-token strings that will be looked up in
// the vocab.
func (b *BPE) bpe(token string) []string {
	if token == "" {
		return nil
	}
	// Split into per-rune sub-tokens. After byte_encoder substitution every
	// rune is one codepoint from the bijection image, so this matches the
	// Python "word = tuple(token)" step.
	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}
	if len(word) == 1 {
		return word
	}

	for {
		// Find the lowest-rank bigram present in the current word.
		bestRank := math.MaxInt
		bestIdx := -1
		for k := 0; k < len(word)-1; k++ {
			if r, ok := b.bpeRanks[[2]string{word[k], word[k+1]}]; ok && r < bestRank {
				bestRank = r
				bestIdx = k
			}
		}
		if bestIdx < 0 {
			break
		}
		first := word[bestIdx]
		second := word[bestIdx+1]

		// Build a new word by merging every non-overlapping occurrence of
		// (first, second) into first+second.
		newWord := make([]string, 0, len(word))
		k := 0
		for k < len(word) {
			// Find the next index >= k where word[idx] == first.
			idx := -1
			for kk := k; kk < len(word); kk++ {
				if word[kk] == first {
					idx = kk
					break
				}
			}
			if idx < 0 {
				newWord = append(newWord, word[k:]...)
				break
			}
			newWord = append(newWord, word[k:idx]...)
			k = idx
			if k < len(word)-1 && word[k] == first && word[k+1] == second {
				newWord = append(newWord, first+second)
				k += 2
			} else {
				newWord = append(newWord, word[k])
				k++
			}
		}
		word = newWord
		if len(word) == 1 {
			break
		}
	}
	return word
}

// Decode maps a sequence of ids back to text. Per-id token strings are
// concatenated, then each rune is mapped through the inverse byte_encoder
// bijection to recover the original UTF-8 byte sequence. Invalid UTF-8 in
// the recovered bytes is preserved verbatim; we leave the caller to decide
// how to render it.
func (b *BPE) Decode(ids []int32) string {
	if len(ids) == 0 {
		return ""
	}
	var concat strings.Builder
	for _, id := range ids {
		if tok, ok := b.decoder[id]; ok {
			concat.WriteString(tok)
		}
	}
	s := concat.String()
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if bt, ok := b.byteDecoder[r]; ok {
			out = append(out, bt)
		}
		// Runes outside the bijection image are silently dropped; this
		// only happens if the caller hands us synthetic ids that didn't
		// come from Encode.
	}
	return string(out)
}
