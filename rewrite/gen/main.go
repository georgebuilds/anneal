// Command gen parses a .upat file and emits a Go source file with per-op match
// functions, replacing the v0 runtime UPat struct traversal with inlined Go code.
//
// Usage (from the directory containing the .upat file):
//
//	go run ../gen/main.go -pkg PKG -in FILE.upat -out FILE_gen.go
//
// The generated file implements a *genMatcher type with a Rewrite method that
// dispatches via a fixed-size per-op array — no reflection, no map lookup in
// the hot path.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"os"
	"strconv"
	"strings"
	"unicode"
)

func main() {
	pkg := flag.String("pkg", "rules", "package name for generated file")
	in := flag.String("in", "", "input .upat file")
	out := flag.String("out", "", "output .go file")
	flag.Parse()

	if *in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: gen -pkg PKG -in FILE.upat -out FILE_gen.go")
		os.Exit(1)
	}

	rules, err := parseUPat(*in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse %s: %v\n", *in, err)
		os.Exit(1)
	}

	src, err := generate(*pkg, *in, rules)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*out, src, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", *out, err)
		os.Exit(1)
	}
}

// ── AST ───────────────────────────────────────────────────────────────────────

// pattern represents one node in a .upat pattern tree.
type pattern struct {
	ops     []string // op names; empty = wildcard
	name    string   // capture name; "" = no capture
	dtype   string   // dtype constraint name; "" = any
	hasArg  bool
	arg     argVal
	srcs    []pattern // nil = srcWild (no src constraint); [] = strict empty
	hasSrcs bool      // distinguishes nil srcs (srcWild) from []
}

type argVal struct {
	kind string // "int64" | "float64" | "bool"
	raw  string // the literal text inside parens
}

// rule is one parsed rewrite rule: a top-level pattern and its handler.
type rule struct {
	pat     pattern
	handler string
	lineNum int
}

// ── Parser ────────────────────────────────────────────────────────────────────

func parseUPat(path string) ([]rule, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var rules []rule
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Split on "=>"
		idx := strings.Index(line, "=>")
		if idx < 0 {
			return nil, fmt.Errorf("line %d: missing '=>'", lineNum)
		}
		patStr := strings.TrimSpace(line[:idx])
		handler := strings.TrimSpace(line[idx+2:])
		if handler == "" {
			return nil, fmt.Errorf("line %d: missing handler name", lineNum)
		}
		p, err := parsePattern(patStr)
		if err != nil {
			return nil, fmt.Errorf("line %d: %v", lineNum, err)
		}
		rules = append(rules, rule{pat: p, handler: handler, lineNum: lineNum})
	}
	return rules, scanner.Err()
}

// parsePattern parses one pattern expression.
// Accepts:
//   - "(Op[:name][@dtype][!arg] srcs...)" — op pattern
//   - "*[:name][@dtype]"                   — wildcard
func parsePattern(s string) (pattern, error) {
	s = strings.TrimSpace(s)
	if s == "*" || strings.HasPrefix(s, "*:") || strings.HasPrefix(s, "*@") || strings.HasPrefix(s, "* ") || s == "* " {
		return parseWild(s)
	}
	if !strings.HasPrefix(s, "(") {
		// bare wildcard: "*"
		if s == "*" {
			return pattern{}, nil
		}
		return parseWild(s)
	}
	return parseOpPat(s)
}

func parseWild(s string) (pattern, error) {
	// s is "*" possibly followed by ":name" and/or "@dtype"
	s = strings.TrimPrefix(s, "*")
	s = strings.TrimSpace(s)
	p := pattern{}
	if err := parseAttrs(s, &p); err != nil {
		return p, err
	}
	return p, nil
}

// parseOpPat parses "(Op[:name][@dtype][!arg] srcs...)" or "(* attrs...)" (wildcard in parens).
func parseOpPat(s string) (pattern, error) {
	if !strings.HasPrefix(s, "(") || !strings.HasSuffix(s, ")") {
		return pattern{}, fmt.Errorf("expected parenthesised pattern, got %q", s)
	}
	// Find the matching closing paren (not just last char, in case of nesting).
	closeIdx, err := matchParen(s)
	if err != nil {
		return pattern{}, err
	}
	inner := s[1:closeIdx] // strip outer parens
	inner = strings.TrimSpace(inner)

	// Parenthesized wildcard: "(* attrs...)"
	if inner == "*" || strings.HasPrefix(inner, "* ") || strings.HasPrefix(inner, "*:") || strings.HasPrefix(inner, "*@") {
		return parseWild(inner)
	}

	p := pattern{}

	// The first token is the ops expression (Op1|Op2|...) possibly with attrs.
	// Attrs start at the first ':' or '@' or '!' that's not inside nested parens.
	// We need to find the end of the ops+attrs prefix.
	opsAndAttrs, rest, err := splitOpsAttrsFromSrcs(inner)
	if err != nil {
		return p, err
	}

	if err := parseOpsAndAttrs(opsAndAttrs, &p); err != nil {
		return p, err
	}

	// Parse srcs from rest (may be empty → srcWild).
	rest = strings.TrimSpace(rest)
	if rest == "" {
		// No src list — srcWild.
		p.hasSrcs = false
	} else {
		p.hasSrcs = true
		srcs, err := parseSrcList(rest)
		if err != nil {
			return p, err
		}
		p.srcs = srcs
	}

	return p, nil
}

// splitOpsAttrsFromSrcs splits the inner content of "(OPS_ATTRS srcs...)" into
// the ops+attrs prefix and the rest (srcs). The ops+attrs prefix is everything
// before the first nested pattern (i.e., before the first '(' or '*' that starts
// a src).
func splitOpsAttrsFromSrcs(inner string) (opsAttrs, rest string, err error) {
	i := 0
	depth := 0
	for i < len(inner) {
		ch := inner[i]
		if ch == '(' && depth == 0 {
			// Start of first src pattern — everything before is opsAttrs.
			break
		}
		if ch == '(' {
			depth++
		} else if ch == ')' {
			depth--
		}
		i++
	}
	// Also check for '*' at depth 0 as start of wildcard src.
	// Find the first unambiguous src start: '(' or a space followed by '*'.
	// Re-do: scan forward and find the first token that looks like a src.
	// A src starts with '(' or with "* " or "*:" or "*@" or lone "*".
	// The opsAttrs part ends at the first whitespace-separated token that is a src.

	// Simple approach: find the last position where the ops+attrs prefix ends.
	// Ops+attrs can contain: letters, digits, '|', ':', '@', '!', '(', ')', '-', '.'.
	// Srcs start at the first '(' or '*' after whitespace, when depth=0.

	i = 0
	for i < len(inner) {
		ch := rune(inner[i])
		if unicode.IsSpace(ch) {
			// Peek ahead: is the next non-space a src starter?
			j := i + 1
			for j < len(inner) && unicode.IsSpace(rune(inner[j])) {
				j++
			}
			if j >= len(inner) {
				break
			}
			next := inner[j]
			if next == '(' || next == '*' {
				opsAttrs = strings.TrimSpace(inner[:i])
				rest = strings.TrimSpace(inner[j:])
				return opsAttrs, rest, nil
			}
		}
		i++
	}
	// No src found.
	return strings.TrimSpace(inner), "", nil
}

// parseOpsAndAttrs parses "Op1|Op2[:name][@dtype][!arg]" into the pattern.
func parseOpsAndAttrs(s string, p *pattern) error {
	// Ops are separated by '|' up to first ':' or '@' or '!'.
	// Find where the ops end.
	opEnd := len(s)
	for i, ch := range s {
		if ch == ':' || ch == '@' || ch == '!' {
			opEnd = i
			break
		}
	}
	opsStr := s[:opEnd]
	attrsStr := s[opEnd:]

	// Parse ops.
	if opsStr != "" {
		for _, op := range strings.Split(opsStr, "|") {
			op = strings.TrimSpace(op)
			if op != "" {
				p.ops = append(p.ops, op)
			}
		}
	}

	// Parse attrs: can appear in any order.
	return parseAttrs(attrsStr, p)
}

// parseAttrs parses optional attrs: [:name][@dtype][!arg] in any order.
func parseAttrs(s string, p *pattern) error {
	s = strings.TrimSpace(s)
	for s != "" {
		switch {
		case strings.HasPrefix(s, ":"):
			s = s[1:]
			end := strings.IndexFunc(s, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
			})
			if end < 0 {
				end = len(s)
			}
			p.name = s[:end]
			s = strings.TrimSpace(s[end:])
		case strings.HasPrefix(s, "@"):
			s = s[1:]
			end := strings.IndexFunc(s, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r)
			})
			if end < 0 {
				end = len(s)
			}
			p.dtype = s[:end]
			s = strings.TrimSpace(s[end:])
		case strings.HasPrefix(s, "!"):
			s = s[1:]
			// arg literal: "int64(N)", "float64(F)", "bool(B)"
			av, rest, err := parseArgLiteral(s)
			if err != nil {
				return err
			}
			p.hasArg = true
			p.arg = av
			s = strings.TrimSpace(rest)
		default:
			if strings.TrimSpace(s) == "" {
				return nil
			}
			return fmt.Errorf("unexpected attr char in %q", s)
		}
	}
	return nil
}

func parseArgLiteral(s string) (argVal, string, error) {
	for _, kind := range []string{"int64", "float64", "bool"} {
		if strings.HasPrefix(s, kind+"(") {
			rest := s[len(kind)+1:]
			end := strings.Index(rest, ")")
			if end < 0 {
				return argVal{}, "", fmt.Errorf("unclosed arg literal %q", s)
			}
			raw := rest[:end]
			return argVal{kind: kind, raw: raw}, rest[end+1:], nil
		}
	}
	return argVal{}, "", fmt.Errorf("unknown arg literal type in %q", s)
}

// parseSrcList parses a space-separated list of src patterns.
// Each src is either "(*...)" or "(...)" — possibly nested.
func parseSrcList(s string) ([]pattern, error) {
	var srcs []pattern
	s = strings.TrimSpace(s)
	for s != "" {
		pat, rest, err := parseSrcToken(s)
		if err != nil {
			return nil, err
		}
		srcs = append(srcs, pat)
		s = strings.TrimSpace(rest)
	}
	return srcs, nil
}

// parseSrcToken extracts the next src pattern from s and returns (pattern, remainder).
func parseSrcToken(s string) (pattern, string, error) {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "(") {
		// Find matching ')'.
		end, err := matchParen(s)
		if err != nil {
			return pattern{}, "", err
		}
		tok := s[:end+1]
		rest := s[end+1:]
		// Check for attrs after closing paren (e.g. "(Mod ...) :base").
		rest = strings.TrimSpace(rest)
		attrStr := ""
		if len(rest) > 0 && (rest[0] == ':' || rest[0] == '@') {
			// Consume attrs from rest.
			end2 := 0
			for end2 < len(rest) {
				if rest[end2] == '(' || rest[end2] == '*' {
					break
				}
				if unicode.IsSpace(rune(rest[end2])) {
					// Check if next non-space is src starter.
					j := end2 + 1
					for j < len(rest) && unicode.IsSpace(rune(rest[j])) {
						j++
					}
					if j >= len(rest) || rest[j] == '(' || rest[j] == '*' {
						attrStr = strings.TrimSpace(rest[:end2])
						rest = strings.TrimSpace(rest[j:])
						break
					}
				}
				end2++
			}
			if attrStr == "" {
				attrStr = strings.TrimSpace(rest)
				rest = ""
			}
		}
		p, err := parseOpPat(tok)
		if err != nil {
			return p, "", err
		}
		if attrStr != "" {
			if err := parseAttrs(attrStr, &p); err != nil {
				return p, "", err
			}
		}
		return p, rest, nil
	}
	// Wildcard: starts with '*'.
	if strings.HasPrefix(s, "*") {
		// Consume until next src starter (space followed by '(' or '*') or end.
		end := 1
		for end < len(s) {
			if unicode.IsSpace(rune(s[end])) {
				j := end + 1
				for j < len(s) && unicode.IsSpace(rune(s[j])) {
					j++
				}
				if j >= len(s) || s[j] == '(' || s[j] == '*' {
					break
				}
				// Peek: is what follows another attr? (: or @)
				if s[j] == ':' || s[j] == '@' {
					end = j
					continue
				}
				break
			}
			end++
		}
		p, err := parseWild(s[:end])
		return p, s[end:], err
	}
	return pattern{}, "", fmt.Errorf("expected src pattern, got %q", s)
}

// matchParen finds the index of the closing ')' that matches the opening '(' at s[0].
func matchParen(s string) (int, error) {
	depth := 0
	for i, ch := range s {
		switch ch {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i, nil
			}
		}
	}
	return -1, fmt.Errorf("unmatched '(' in %q", s)
}

// ── Code generator ────────────────────────────────────────────────────────────

// opRules maps op name → ordered list of rules for that op.
type opRules struct {
	op    string
	rules []rule
}

func generate(pkg, inputFile string, rules []rule) ([]byte, error) {
	// Group rules by op, preserving original order within each op.
	opOrder := []string{}
	opMap := map[string][]rule{}
	seen := map[string]bool{}

	for _, r := range rules {
		for _, op := range r.pat.ops {
			if !seen[op] {
				seen[op] = true
				opOrder = append(opOrder, op)
			}
			opMap[op] = append(opMap[op], r)
		}
		// A rule with no ops (wildcard root) is unusual; skip for now.
	}

	var buf bytes.Buffer
	w := &buf

	// File header.
	fmt.Fprintf(w, "// Code generated from %s by rewrite/gen. DO NOT EDIT.\n\n", inputFile)
	fmt.Fprintf(w, "package %s\n\n", pkg)
	fmt.Fprintf(w, "import (\n")
	fmt.Fprintf(w, "\t\"math\"\n")
	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "\t\"github.com/georgebuilds/anneal/rewrite\"\n")
	fmt.Fprintf(w, "\t\"github.com/georgebuilds/anneal/uop\"\n")
	fmt.Fprintf(w, ")\n\n")

	// genMatcher type and constructor.
	fmt.Fprintf(w, "// symbolicGenerated is the generated matcher, produced from symbolic.upat.\n")
	fmt.Fprintf(w, "// It replaces the v0 UPat struct traversal with inlined per-op match functions.\n")
	fmt.Fprintf(w, "// No reflection; captures are local variables built only on full match.\n")
	fmt.Fprintf(w, "var symbolicGenerated rewrite.Matcher = newSymbolicGenerated()\n\n")

	fmt.Fprintf(w, "// genMatcher dispatches rewrite attempts via a fixed array indexed by op.\n")
	fmt.Fprintf(w, "// Array indexing is O(1) with no map allocation.\n")
	fmt.Fprintf(w, "type genMatcher struct {\n")
	fmt.Fprintf(w, "\tfns [uop.OpCount]func(uop.UOp, any) (uop.UOp, bool)\n")
	fmt.Fprintf(w, "}\n\n")

	fmt.Fprintf(w, "func (gm *genMatcher) Rewrite(u uop.UOp, ctx any) (uop.UOp, bool) {\n")
	fmt.Fprintf(w, "\tif fn := gm.fns[u.Op()]; fn != nil {\n")
	fmt.Fprintf(w, "\t\treturn fn(u, ctx)\n")
	fmt.Fprintf(w, "\t}\n")
	fmt.Fprintf(w, "\treturn uop.UOp{}, false\n")
	fmt.Fprintf(w, "}\n\n")

	fmt.Fprintf(w, "func newSymbolicGenerated() *genMatcher {\n")
	fmt.Fprintf(w, "\tgm := &genMatcher{}\n")
	for _, op := range opOrder {
		fmt.Fprintf(w, "\tgm.fns[uop.Op%s] = genMatch%s\n", op, op)
	}
	fmt.Fprintf(w, "\treturn gm\n")
	fmt.Fprintf(w, "}\n\n")

	// genArgEq — inline arg equality helper.
	fmt.Fprintf(w, "// genArgEq reports whether a == b for arg matching purposes.\n")
	fmt.Fprintf(w, "// NaN float64 values compare equal when bits match (same constant → same node).\n")
	fmt.Fprintf(w, "func genArgEq(a, b any) bool {\n")
	fmt.Fprintf(w, "\tif a == nil && b == nil { return true }\n")
	fmt.Fprintf(w, "\tif a == nil || b == nil { return false }\n")
	fmt.Fprintf(w, "\tswitch av := a.(type) {\n")
	fmt.Fprintf(w, "\tcase int64:\n")
	fmt.Fprintf(w, "\t\tbv, ok := b.(int64)\n")
	fmt.Fprintf(w, "\t\treturn ok && av == bv\n")
	fmt.Fprintf(w, "\tcase float64:\n")
	fmt.Fprintf(w, "\t\tbv, ok := b.(float64)\n")
	fmt.Fprintf(w, "\t\treturn ok && math.Float64bits(av) == math.Float64bits(bv)\n")
	fmt.Fprintf(w, "\tcase bool:\n")
	fmt.Fprintf(w, "\t\tbv, ok := b.(bool)\n")
	fmt.Fprintf(w, "\t\treturn ok && av == bv\n")
	fmt.Fprintf(w, "\t}\n")
	fmt.Fprintf(w, "\treturn false\n")
	fmt.Fprintf(w, "}\n\n")

	// Per-op match functions.
	fmt.Fprintf(w, "// ── per-op match functions ─────────────────────────────────────────────────────\n\n")
	for _, op := range opOrder {
		if err := emitOpFunc(w, op, opMap[op]); err != nil {
			return nil, fmt.Errorf("op %s: %v", op, err)
		}
	}

	// gofmt the output.
	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		// Return unformatted with error context for debugging.
		return buf.Bytes(), fmt.Errorf("gofmt: %v\n\n--- raw output ---\n%s", err, buf.String())
	}
	return formatted, nil
}

// emitOpFunc emits one per-op match function.
func emitOpFunc(w *bytes.Buffer, op string, rules []rule) error {
	fmt.Fprintf(w, "func genMatch%s(u uop.UOp, ctx any) (uop.UOp, bool) {\n", op)
	for _, r := range rules {
		if err := emitRule(w, r); err != nil {
			return fmt.Errorf("rule line %d: %v", r.lineNum, err)
		}
	}
	fmt.Fprintf(w, "\treturn uop.UOp{}, false\n")
	fmt.Fprintf(w, "}\n\n")
	return nil
}

// emitRule emits one rule as nested if blocks inside an op match function.
// The root op is already guaranteed by dispatch so only srcs/attrs need checking.
func emitRule(w *bytes.Buffer, r rule) error {
	fmt.Fprintf(w, "\t// Rule line %d: => %s\n", r.lineNum, r.handler)
	// caps maps capture name → Go variable holding the captured UOp.
	caps := map[string]string{}
	if r.pat.name != "" {
		caps[r.pat.name] = "u"
	}
	return emitSrcsMatch(w, r.pat, "u", caps, r.handler, "\t")
}

// emitSrcsMatch emits the nested match block for pat's src list, then calls handler.
// nodeVar is the Go variable for the current node. caps accumulates captures.
func emitSrcsMatch(w *bytes.Buffer, pat pattern, nodeVar string, caps map[string]string, handler, indent string) error {
	if !pat.hasSrcs {
		// srcWild — no src constraint; emit handler call directly.
		return emitHandlerCall(w, handler, caps, indent)
	}
	// NSrc check.
	fmt.Fprintf(w, "%sif %s.NSrc() == %d {\n", indent, nodeVar, len(pat.srcs))
	err := emitSrcChain(w, pat.srcs, nodeVar, 0, caps, handler, indent+"\t")
	fmt.Fprintf(w, "%s}\n", indent)
	return err
}

// srcVarNeeded reports whether a src pattern requires a local variable to be declared.
// Pure wildcards (no op/dtype/arg/name/srcs) need no variable — they match unconditionally.
func srcVarNeeded(sp pattern, caps map[string]string) bool {
	if len(sp.ops) > 0 || sp.dtype != "" || sp.hasArg || sp.hasSrcs {
		return true
	}
	if sp.name != "" {
		return true // either new capture or equality check against existing cap
	}
	return false
}

// emitSrcChain emits the match chain for srcs[i:].
// For each src it: (1) builds any non-equality conditions, (2) emits inline checks or if blocks,
// (3) captures the node, (4) recurses into nested srcs, then moves to srcs[i+1].
func emitSrcChain(w *bytes.Buffer, srcs []pattern, parentVar string, i int, caps map[string]string, handler, indent string) error {
	if i >= len(srcs) {
		return emitHandlerCall(w, handler, caps, indent)
	}
	sp := srcs[i]
	if !srcVarNeeded(sp, caps) {
		// Pure wildcard — matches unconditionally; skip variable declaration.
		return emitSrcChain(w, srcs, parentVar, i+1, caps, handler, indent)
	}
	srcVar := fmt.Sprintf("_s%d%s", i, varSuffix(parentVar))
	fmt.Fprintf(w, "%s%s := %s.Src(%d)\n", indent, srcVar, parentVar, i)

	// Collect non-equality conditions for this src.
	var conds []string
	if len(sp.ops) == 1 {
		conds = append(conds, fmt.Sprintf("%s.Op() == uop.Op%s", srcVar, sp.ops[0]))
	} else if len(sp.ops) > 1 {
		parts := make([]string, len(sp.ops))
		for j, op := range sp.ops {
			parts[j] = fmt.Sprintf("%s.Op() == uop.Op%s", srcVar, op)
		}
		conds = append(conds, "("+strings.Join(parts, " || ")+")")
	}
	if sp.dtype != "" {
		conds = append(conds, fmt.Sprintf("%s.DType() == uop.Dtypes.%s", srcVar, sp.dtype))
	}
	if sp.hasArg {
		argLit, err := goArgLiteral(sp.arg)
		if err != nil {
			return err
		}
		conds = append(conds, fmt.Sprintf("genArgEq(%s.Arg(), %s)", srcVar, argLit))
	}
	// Equality constraint (same capture name used before).
	if sp.name != "" {
		if existing, ok := caps[sp.name]; ok {
			conds = append(conds, fmt.Sprintf("%s == %s", srcVar, existing))
		}
	}

	// Nested NSrc check goes inside the if block.
	needsNested := sp.hasSrcs
	hasNestedCaps := sp.name != "" && caps[sp.name] == ""
	needsBlock := len(conds) > 0 || needsNested

	if needsBlock {
		fmt.Fprintf(w, "%sif %s {\n", indent, strings.Join(conds, " && "))
		indent += "\t"
	}

	// New capture for this src (if any).
	if sp.name != "" {
		if _, ok := caps[sp.name]; !ok {
			capVar := fmt.Sprintf("_cap_%s", sp.name)
			fmt.Fprintf(w, "%s%s := %s\n", indent, capVar, srcVar)
			caps[sp.name] = capVar
		}
	}
	_ = hasNestedCaps

	// Recurse into nested srcs of this src, then continue to srcs[i+1].
	var err error
	if needsNested {
		fmt.Fprintf(w, "%sif %s.NSrc() == %d {\n", indent, srcVar, len(sp.srcs))
		subCaps := copyCaps(caps)
		// The nested src's name (if any) was set to srcVar above; subCaps inherits it.
		err = emitNestedSrcChain(w, sp, srcVar, subCaps, func(caps2 map[string]string) error {
			// After nested srcs match, continue with next top-level src.
			return emitSrcChain(w, srcs, parentVar, i+1, caps2, handler, indent+"\t")
		}, indent+"\t")
		fmt.Fprintf(w, "%s}\n", indent)
	} else {
		// No nested srcs — continue to next top-level src.
		err = emitSrcChain(w, srcs, parentVar, i+1, caps, handler, indent)
	}

	if needsBlock {
		indent = indent[:len(indent)-1]
		fmt.Fprintf(w, "%s}\n", indent)
	}

	return err
}

// emitNestedSrcChain emits match code for the src list of sp (a nested pattern).
// After all nested srcs match, calls cont with the accumulated caps.
func emitNestedSrcChain(w *bytes.Buffer, sp pattern, nodeVar string, caps map[string]string, cont func(map[string]string) error, indent string) error {
	return emitSrcChainWithCont(w, sp.srcs, nodeVar, 0, caps, cont, indent)
}

func emitSrcChainWithCont(w *bytes.Buffer, srcs []pattern, parentVar string, i int, caps map[string]string, cont func(map[string]string) error, indent string) error {
	if i >= len(srcs) {
		return cont(caps)
	}
	sp := srcs[i]
	if !srcVarNeeded(sp, caps) {
		return emitSrcChainWithCont(w, srcs, parentVar, i+1, caps, cont, indent)
	}
	srcVar := fmt.Sprintf("_s%d%s", i, varSuffix(parentVar))
	fmt.Fprintf(w, "%s%s := %s.Src(%d)\n", indent, srcVar, parentVar, i)

	var conds []string
	if len(sp.ops) == 1 {
		conds = append(conds, fmt.Sprintf("%s.Op() == uop.Op%s", srcVar, sp.ops[0]))
	} else if len(sp.ops) > 1 {
		parts := make([]string, len(sp.ops))
		for j, op := range sp.ops {
			parts[j] = fmt.Sprintf("%s.Op() == uop.Op%s", srcVar, op)
		}
		conds = append(conds, "("+strings.Join(parts, " || ")+")")
	}
	if sp.dtype != "" {
		conds = append(conds, fmt.Sprintf("%s.DType() == uop.Dtypes.%s", srcVar, sp.dtype))
	}
	if sp.hasArg {
		argLit, err := goArgLiteral(sp.arg)
		if err != nil {
			return err
		}
		conds = append(conds, fmt.Sprintf("genArgEq(%s.Arg(), %s)", srcVar, argLit))
	}
	if sp.name != "" {
		if existing, ok := caps[sp.name]; ok {
			conds = append(conds, fmt.Sprintf("%s == %s", srcVar, existing))
		}
	}

	needsBlock := len(conds) > 0 || sp.hasSrcs
	if needsBlock {
		fmt.Fprintf(w, "%sif %s {\n", indent, strings.Join(conds, " && "))
		indent += "\t"
	}

	if sp.name != "" {
		if _, ok := caps[sp.name]; !ok {
			capVar := fmt.Sprintf("_cap_%s", sp.name)
			fmt.Fprintf(w, "%s%s := %s\n", indent, capVar, srcVar)
			caps[sp.name] = capVar
		}
	}

	var err error
	if sp.hasSrcs {
		fmt.Fprintf(w, "%sif %s.NSrc() == %d {\n", indent, srcVar, len(sp.srcs))
		err = emitSrcChainWithCont(w, sp.srcs, srcVar, 0, copyCaps(caps), func(caps2 map[string]string) error {
			return emitSrcChainWithCont(w, srcs, parentVar, i+1, caps2, cont, indent+"\t")
		}, indent+"\t")
		fmt.Fprintf(w, "%s}\n", indent)
	} else {
		err = emitSrcChainWithCont(w, srcs, parentVar, i+1, caps, cont, indent)
	}

	if needsBlock {
		indent = indent[:len(indent)-1]
		fmt.Fprintf(w, "%s}\n", indent)
	}
	return err
}

// emitHandlerCall emits the caps map construction and handler call.
func emitHandlerCall(w *bytes.Buffer, handler string, caps map[string]string, indent string) error {
	fmt.Fprintf(w, "%s_caps := map[string]uop.UOp{", indent)
	first := true
	for _, name := range sortedKeys(caps) {
		if !first {
			fmt.Fprintf(w, ", ")
		}
		fmt.Fprintf(w, "%q: %s", name, caps[name])
		first = false
	}
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "%sif _r, _ok := %s(_caps, ctx); _ok {\n", indent, handler)
	fmt.Fprintf(w, "%s\treturn _r, true\n", indent)
	fmt.Fprintf(w, "%s}\n", indent)
	return nil
}

// copyCaps makes a shallow copy of the captures map.
func copyCaps(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// varSuffix creates a safe variable name suffix from a parent variable name.
func varSuffix(parent string) string {
	// "u" → "", "_s0u" → "s0u", etc. Strip leading "_" for brevity.
	s := safeVar(parent)
	if s == "u" {
		return ""
	}
	return "_" + s
}

// goArgLiteral converts an argVal to a Go literal expression.
func goArgLiteral(av argVal) (string, error) {
	switch av.kind {
	case "int64":
		_, err := strconv.ParseInt(av.raw, 10, 64)
		if err != nil {
			return "", fmt.Errorf("invalid int64 literal %q: %v", av.raw, err)
		}
		return fmt.Sprintf("int64(%s)", av.raw), nil
	case "float64":
		_, err := strconv.ParseFloat(av.raw, 64)
		if err != nil {
			return "", fmt.Errorf("invalid float64 literal %q: %v", av.raw, err)
		}
		return fmt.Sprintf("float64(%s)", av.raw), nil
	case "bool":
		if av.raw != "true" && av.raw != "false" {
			return "", fmt.Errorf("invalid bool literal %q", av.raw)
		}
		return av.raw, nil
	}
	return "", fmt.Errorf("unknown arg kind %q", av.kind)
}

// safeVar makes a string safe for use as a Go variable name suffix.
func safeVar(s string) string {
	var b strings.Builder
	for _, ch := range s {
		if unicode.IsLetter(ch) || unicode.IsDigit(ch) {
			b.WriteRune(ch)
		} else {
			b.WriteRune('_')
		}
	}
	return b.String()
}

// sortedKeys returns the keys of a map in lexicographic order (for determinism).
func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Simple insertion sort (small maps).
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}
