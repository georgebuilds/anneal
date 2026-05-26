package rewrite

import "github.com/georgebuilds/anneal/uop"

// Matcher is the interface implemented by PatternMatcher and generated matchers.
// GraphRewrite accepts any Matcher as its pm or bpm argument.
type Matcher interface {
	Rewrite(u uop.UOp, ctx any) (uop.UOp, bool)
}

// MatchFn is the handler called when a pattern fully matches a UOp.
// captures maps each named capture to the matched UOp.
// ctx is the optional context value passed through from GraphRewrite.
// Return (replacement, true) to accept the match; (zero, false) to skip and try the next rule.
type MatchFn func(captures map[string]uop.UOp, ctx any) (uop.UOp, bool)

// Rule pairs a UPat with its handler.
type Rule struct {
	Pat *UPat
	Fn  MatchFn
}

type pdictEntry struct {
	pat         *UPat
	fn          MatchFn
	earlyReject map[uop.Op]struct{}
}

// PatternMatcher is an op-indexed dispatch table of (pattern, handler) rules.
//
// Rule order is significant — earlier rules win. Internally, pdict maps each op to
// the list of rules that may match nodes of that op, enabling O(rules-for-this-op)
// dispatch rather than a linear scan over all rules.
//
// Top-level UPat nodes must specify at least one op. Use AnyPat to cover multiple ops
// with one rule.
//
// Composition: use Add to concatenate two matchers; earlier rules win.
type PatternMatcher struct {
	rules []Rule
	pdict map[uop.Op][]pdictEntry
}

// NewPatternMatcher builds a PatternMatcher from the given rules.
// Panics if a top-level pattern has no op constraint.
func NewPatternMatcher(rules []Rule) *PatternMatcher {
	pm := &PatternMatcher{
		rules: append([]Rule(nil), rules...),
		pdict: make(map[uop.Op][]pdictEntry),
	}
	for _, r := range rules {
		if r.Pat.isAny {
			for _, alt := range r.Pat.anyAlts {
				if len(alt.ops) == 0 {
					panic("rewrite: AnyPat alternative must specify at least one op")
				}
				pm.register(alt.ops, r.Pat, r.Fn)
			}
		} else {
			if len(r.Pat.ops) == 0 {
				panic("rewrite: top-level UPat must specify at least one op")
			}
			pm.register(r.Pat.ops, r.Pat, r.Fn)
		}
	}
	return pm
}

func (pm *PatternMatcher) register(ops []uop.Op, pat *UPat, fn MatchFn) {
	e := pdictEntry{pat: pat, fn: fn, earlyReject: pat.earlyReject}
	for _, op := range ops {
		pm.pdict[op] = append(pm.pdict[op], e)
	}
}

// Rewrite attempts to rewrite u by trying each applicable rule in order.
// Returns (replacement, true) on the first successful match; (zero, false) if none fires.
func (pm *PatternMatcher) Rewrite(u uop.UOp, ctx any) (uop.UOp, bool) {
	entries := pm.pdict[u.Op()]
	if len(entries) == 0 {
		return uop.UOp{}, false
	}
	srcOps := srcOpSet(u)
	for _, e := range entries {
		if !isSubset(e.earlyReject, srcOps) {
			continue
		}
		for _, captures := range e.pat.Match(u, map[string]uop.UOp{}) {
			if result, ok := e.fn(captures, ctx); ok {
				return result, true
			}
		}
	}
	return uop.UOp{}, false
}

// Add returns a new PatternMatcher combining pm's rules followed by more's rules.
// Earlier rules win; pm's rules take priority.
func (pm *PatternMatcher) Add(more *PatternMatcher) *PatternMatcher {
	combined := make([]Rule, len(pm.rules)+len(more.rules))
	copy(combined, pm.rules)
	copy(combined[len(pm.rules):], more.rules)
	return NewPatternMatcher(combined)
}

// ── helpers ───────────────────────────────────────────────────────────────────

// srcOpSet returns the set of ops among the direct children of u.
// Used for early-reject filtering: if a rule requires certain child ops and they
// are absent, the rule cannot possibly match.
func srcOpSet(u uop.UOp) map[uop.Op]struct{} {
	s := make(map[uop.Op]struct{}, u.NSrc())
	for i := 0; i < u.NSrc(); i++ {
		s[u.Src(i).Op()] = struct{}{}
	}
	return s
}

// isSubset reports whether every op in required appears in available.
func isSubset(required, available map[uop.Op]struct{}) bool {
	for op := range required {
		if _, ok := available[op]; !ok {
			return false
		}
	}
	return true
}
