package main

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// parseUpatPairs parses a symbolic.upat file and returns every (op, handler)
// pair as a set of "op:handler" strings.  Op names are lowercased; alternation
// "Add|Mul" is expanded to one entry per op.
func parseUpatPairs(path string) (map[string]bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	pairs := make(map[string]bool)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=>")
		if idx < 0 {
			continue
		}
		patternSide := strings.TrimSpace(line[:idx])
		handler := strings.TrimSpace(line[idx+2:])
		if handler == "" || !strings.HasPrefix(patternSide, "(") {
			continue
		}
		// Extract the OPS_EXPR: everything between "(" and the first " ", ":", or ")".
		inner := patternSide[1:]
		end := strings.IndexAny(inner, " :\t)")
		if end < 0 {
			end = len(inner)
		}
		opsExpr := inner[:end]
		if opsExpr == "" || opsExpr == "*" {
			continue
		}
		for _, op := range strings.Split(opsExpr, "|") {
			op = strings.ToLower(op)
			if op != "" {
				pairs[op+":"+handler] = true
			}
		}
	}
	return pairs, scanner.Err()
}

// TestUpatDriftCheck verifies that the curated allRules symbolic entries
// exactly match the (op, handler) pairs defined in symbolic.upat for every
// op the curated table covers.
//
// Drift detection: adding or removing a .upat rule for a covered op without
// updating allRules in cmd_explain.go will cause this test to fail, naming
// the specific (op, handler) pair that is out of sync.
func TestUpatDriftCheck(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller: cannot determine test file path")
	}
	upatPath := filepath.Join(filepath.Dir(thisFile), "..", "..", "rewrite", "rules", "symbolic.upat")

	upatPairs, err := parseUpatPairs(upatPath)
	if err != nil {
		t.Fatalf("parseUpatPairs(%s): %v", upatPath, err)
	}
	if len(upatPairs) == 0 {
		t.Fatalf("parseUpatPairs returned 0 pairs — file missing or empty?")
	}

	// Build the curated (op, handler) set and the covered-ops set from allRules.
	curatedPairs := make(map[string]bool)
	coveredOps := make(map[string]bool)
	for _, r := range allRules {
		if r.kind != "symbolic" {
			continue
		}
		if r.handler == "" {
			t.Errorf("symbolic ruleEntry missing handler: ops=%v pattern=%q — add handler field in cmd_explain.go", r.ops, r.pattern)
			continue
		}
		for _, op := range r.ops {
			curatedPairs[op+":"+r.handler] = true
			coveredOps[op] = true
		}
	}

	// relevantUpatPairs = .upat pairs restricted to ops the curated table covers.
	relevantUpatPairs := make(map[string]bool)
	for pair := range upatPairs {
		op := strings.SplitN(pair, ":", 2)[0]
		if coveredOps[op] {
			relevantUpatPairs[pair] = true
		}
	}

	// Direction 1: every relevant .upat pair must be in the curated table.
	for pair := range relevantUpatPairs {
		if !curatedPairs[pair] {
			parts := strings.SplitN(pair, ":", 2)
			t.Errorf(".upat defines rule (%s, %s) but allRules has no matching symbolic entry — update cmd_explain.go", parts[0], parts[1])
		}
	}

	// Direction 2: every curated pair must exist in .upat.
	for pair := range curatedPairs {
		if !upatPairs[pair] {
			parts := strings.SplitN(pair, ":", 2)
			t.Errorf("allRules has symbolic entry (%s, %s) but symbolic.upat has no matching rule — stale entry in cmd_explain.go?", parts[0], parts[1])
		}
	}
}
