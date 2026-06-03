package onnx

// Phase 4 ONNX conformance harness.
//
// Walks the curated subset of the ONNX 1.17.0 backend node-test corpus
// committed under onnx/testdata/node/ (regenerated via
// notes/scripts/copy_node_corpus.py), runs each model through the
// importer, and compares the realised output against the
// onnx-supplied golden output_*.pb (Strategy B fixtures — no
// onnxruntime or Python at test time).
//
// Per plan §7: the value oracle reports max-abs-diff and
// count-over-tolerance per case, not pass-counts. The skip list in
// conformance_skip.go is the documented exclusion contract; anything
// not skipped and not within tolerance is a real bug.
//
// Tolerance: 1e-3 (the conformance bar per plan §8 gates).

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
	"google.golang.org/protobuf/proto"
)

// conformanceTol is the per-case max-abs-diff allowed against the
// golden output. Matches the Phase 1/2/3 gates.
const conformanceTol = float32(1e-3)

// summaryPath is where we drop the grep-able run summary.
const conformanceSummaryPath = "../test_output_phase4_conformance.txt"

// conformanceResult is one row of the per-case summary.
type conformanceResult struct {
	Name         string
	Skipped      bool
	SkipReason   string
	Failed       bool
	FailReason   string
	MaxAbsDiff   float32
	CountOverTol int
	TotalElems   int
}

// conformanceCase is a single test directory under
// onnx/testdata/node/.
type conformanceCase struct {
	Name      string // e.g. "test_relu"
	Dir       string // absolute path
	ModelPath string
	DataSet0  string // absolute path of test_data_set_0
}

// conformanceCorpusDir returns the abs path of onnx/testdata/node/.
func conformanceCorpusDir(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs("testdata/node")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	return abs
}

// loadNodeCorpus enumerates every test_* directory under the corpus
// root. The result is sorted lexicographically so that subtest
// ordering is deterministic.
func loadNodeCorpus(t *testing.T) []conformanceCase {
	t.Helper()
	root := conformanceCorpusDir(t)
	st, err := os.Stat(root)
	if err != nil || !st.IsDir() {
		t.Skipf("conformance corpus missing at %s: run notes/scripts/copy_node_corpus.py to populate", root)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read %s: %v", root, err)
	}
	out := make([]conformanceCase, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, "test_") {
			continue
		}
		dir := filepath.Join(root, name)
		model := filepath.Join(dir, "model.onnx")
		tds0 := filepath.Join(dir, "test_data_set_0")
		if _, err := os.Stat(model); err != nil {
			continue
		}
		if st, err := os.Stat(tds0); err != nil || !st.IsDir() {
			continue
		}
		out = append(out, conformanceCase{
			Name:      name,
			Dir:       dir,
			ModelPath: model,
			DataSet0:  tds0,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// loadTensorProtoFile parses a *.pb TensorProto from disk.
func loadTensorProtoFile(path string) (*onnxpb.TensorProto, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}
	tp := &onnxpb.TensorProto{}
	if err := proto.Unmarshal(data, tp); err != nil {
		return nil, fmt.Errorf("unmarshal %q: %w", path, err)
	}
	return tp, nil
}

// listDataSetFiles returns (input_N.pb, output_N.pb) paths sorted by N.
func listDataSetFiles(dir string) (inputs, outputs []string, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	for _, e := range entries {
		n := e.Name()
		switch {
		case strings.HasPrefix(n, "input_") && strings.HasSuffix(n, ".pb"):
			inputs = append(inputs, filepath.Join(dir, n))
		case strings.HasPrefix(n, "output_") && strings.HasSuffix(n, ".pb"):
			outputs = append(outputs, filepath.Join(dir, n))
		}
	}
	sort.Strings(inputs)
	sort.Strings(outputs)
	return inputs, outputs, nil
}

// runConformanceCase loads the model + inputs, runs through the
// importer, and compares against the golden outputs. Recovers from
// importer panics so one buggy case doesn't kill the whole sweep.
func runConformanceCase(t *testing.T, c conformanceCase) (res conformanceResult) {
	t.Helper()
	res = conformanceResult{Name: c.Name}
	defer func() {
		if r := recover(); r != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("panic during run: %v", r)
		}
	}()

	arena := uop.NewArena(65536)

	// Import the model.
	r, err := ImportFile(c.ModelPath, arena, "test")
	if err != nil {
		res.Failed = true
		res.FailReason = fmt.Sprintf("ImportFile: %v", err)
		return res
	}

	inPaths, outPaths, err := listDataSetFiles(c.DataSet0)
	if err != nil {
		res.Failed = true
		res.FailReason = fmt.Sprintf("listDataSetFiles: %v", err)
		return res
	}
	if len(outPaths) == 0 {
		res.Failed = true
		res.FailReason = "no output_*.pb fixtures"
		return res
	}

	// Bind inputs to graph inputs by index. The ONNX backend node-test
	// convention is input_N.pb corresponds to graph.Input[N] (which is
	// the union of model inputs + initializers; non-initializer
	// inputs come first in test_data_set_0).
	inputs := r.Inputs()
	if len(inPaths) > len(inputs) {
		// Some tests stash initializer values in input_*.pb as well;
		// trim to graph-input count.
		inPaths = inPaths[:len(inputs)]
	}
	named := make(map[string]*tensor.Tensor, len(inPaths))
	for i, p := range inPaths {
		tp, err := loadTensorProtoFile(p)
		if err != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("decode input %d: %v", i, err)
			return res
		}
		// Find the graph input matching by name first, then fall back
		// to positional binding.
		name := tp.GetName()
		matched := false
		for _, in := range inputs {
			if in.Name == name {
				matched = true
				break
			}
		}
		if !matched {
			name = inputs[i].Name
		}
		tn, err := tensorFromProto(arena, tp, "test")
		if err != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("input %d (%s): %v", i, name, err)
			return res
		}
		named[name] = tn
	}

	out, err := r.Run(named)
	if err != nil {
		res.Failed = true
		res.FailReason = fmt.Sprintf("Run: %v", err)
		return res
	}

	// Compare each output to the golden. We sum max-abs-diff across
	// outputs (typically there is only one); count_over_tol is the
	// per-element tally across all outputs.
	outs := r.Outputs()
	if len(outPaths) > len(outs) {
		outPaths = outPaths[:len(outs)]
	}
	for i, p := range outPaths {
		tp, err := loadTensorProtoFile(p)
		if err != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("decode output %d: %v", i, err)
			return res
		}
		// Match output by name when possible.
		name := tp.GetName()
		matched := false
		for _, o := range outs {
			if o.Name == name {
				matched = true
				break
			}
		}
		if !matched {
			name = outs[i].Name
		}
		gotT, ok := out[name]
		if !ok {
			res.Failed = true
			res.FailReason = fmt.Sprintf("output %q missing", name)
			return res
		}
		got, _, err := cpuEval(gotT)
		if err != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("cpuEval output %d: %v", i, err)
			return res
		}
		// Decode golden via tensorFromProto so the same dtype-cast
		// logic applies (DOUBLE->f32, INT64 trap, ...).
		wantLeaf, err := tensorFromProto(arena, tp, "test")
		if err != nil {
			res.Failed = true
			res.FailReason = fmt.Sprintf("decode golden output %d: %v", i, err)
			return res
		}
		want := wantLeaf.Data()
		if len(got) != len(want) {
			res.Failed = true
			res.FailReason = fmt.Sprintf("output %d length mismatch: got=%d want=%d", i, len(got), len(want))
			return res
		}
		m, cnt := maxAbsDiffAndCount(got, want, conformanceTol)
		if m > res.MaxAbsDiff {
			res.MaxAbsDiff = m
		}
		res.CountOverTol += cnt
		res.TotalElems += len(got)
	}
	return res
}

// maxAbsDiffAndCount returns (max abs diff, count of elements with
// abs diff > tol).
func maxAbsDiffAndCount(a, b []float32, tol float32) (float32, int) {
	var m float32
	var cnt int
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
		if d > tol {
			cnt++
		}
	}
	return m, cnt
}

// conformanceState collects per-case results across the run.
type conformanceState struct {
	results []conformanceResult
}

var globalConformanceState conformanceState

// TestConformance_NodeCorpus walks every test in the curated corpus,
// applies the skip list, runs the importer, and asserts max-abs-diff
// is within tolerance for non-skipped cases. Each case logs its
// max-abs-diff and count-over-tolerance so the value oracle is
// per-case, not aggregate.
func TestConformance_NodeCorpus(t *testing.T) {
	cases := loadNodeCorpus(t)
	if len(cases) == 0 {
		t.Skip("conformance corpus empty")
	}
	t.Logf("conformance: %d cases discovered under testdata/node/", len(cases))

	// Reset state for re-runs.
	globalConformanceState.results = make([]conformanceResult, 0, len(cases))

	for _, c := range cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			if reason, skipped := matchSkip(c.Name); skipped {
				globalConformanceState.results = append(globalConformanceState.results, conformanceResult{
					Name: c.Name, Skipped: true, SkipReason: reason,
				})
				t.Skipf("skip: %s", reason)
				return
			}
			res := runConformanceCase(t, c)
			globalConformanceState.results = append(globalConformanceState.results, res)
			if res.Failed {
				t.Errorf("FAIL: %s", res.FailReason)
				return
			}
			t.Logf("max-abs-diff=%.3e count_over_tol=%d/%d",
				res.MaxAbsDiff, res.CountOverTol, res.TotalElems)
			if res.MaxAbsDiff > conformanceTol {
				t.Errorf("max-abs-diff %.3e exceeds tol %.3e (count_over_tol=%d/%d)",
					res.MaxAbsDiff, conformanceTol, res.CountOverTol, res.TotalElems)
			}
		})
	}
}

// TestConformance_SummaryReport summarises the run: total, skipped,
// passed, failed, plus the per-reason histogram of skips. Writes the
// summary to test_output_phase4_conformance.txt for grep-ability.
//
// Asserts (a) skip list has explicit entries, (b) failure count = 0,
// (c) core families pass (test_relu, test_add, test_matmul_2d).
//
// IMPORTANT: this test depends on TestConformance_NodeCorpus having
// run first (it consumes globalConformanceState.results). Go's
// in-package alphabetical test ordering puts NodeCorpus before
// SummaryReport, which is the order we want.
func TestConformance_SummaryReport(t *testing.T) {
	results := globalConformanceState.results
	if len(results) == 0 {
		t.Skip("no conformance results captured (run TestConformance_NodeCorpus first)")
	}

	var total, skipped, passed, failed int
	skipReasons := make(map[string]int)
	bestDiff := float32(-1)
	worstDiff := float32(-1)
	var diffs []float32
	var failNames []string
	for _, r := range results {
		total++
		if r.Skipped {
			skipped++
			skipReasons[r.SkipReason]++
			continue
		}
		if r.Failed || r.MaxAbsDiff > conformanceTol {
			failed++
			failNames = append(failNames, r.Name+": "+r.FailReason)
			continue
		}
		passed++
		diffs = append(diffs, r.MaxAbsDiff)
		if bestDiff < 0 || r.MaxAbsDiff < bestDiff {
			bestDiff = r.MaxAbsDiff
		}
		if r.MaxAbsDiff > worstDiff {
			worstDiff = r.MaxAbsDiff
		}
	}
	sort.Slice(diffs, func(i, j int) bool { return diffs[i] < diffs[j] })
	var median float32
	if len(diffs) > 0 {
		median = diffs[len(diffs)/2]
	}

	// Skip-reason histogram.
	type rcount struct {
		reason string
		count  int
	}
	var hist []rcount
	for r, c := range skipReasons {
		hist = append(hist, rcount{r, c})
	}
	sort.Slice(hist, func(i, j int) bool {
		if hist[i].count != hist[j].count {
			return hist[i].count > hist[j].count
		}
		return hist[i].reason < hist[j].reason
	})

	// Build the report.
	var sb strings.Builder
	fmt.Fprintf(&sb, "Phase 4 ONNX conformance summary\n")
	fmt.Fprintf(&sb, "================================\n")
	fmt.Fprintf(&sb, "total:   %d\n", total)
	fmt.Fprintf(&sb, "passed:  %d\n", passed)
	fmt.Fprintf(&sb, "skipped: %d\n", skipped)
	fmt.Fprintf(&sb, "failed:  %d\n", failed)
	fmt.Fprintf(&sb, "\nMax-abs-diff range across passing cases:\n")
	fmt.Fprintf(&sb, "  best:   %.3e\n", bestDiff)
	fmt.Fprintf(&sb, "  median: %.3e\n", median)
	fmt.Fprintf(&sb, "  worst:  %.3e\n", worstDiff)
	fmt.Fprintf(&sb, "  tolerance: %.3e\n", conformanceTol)
	fmt.Fprintf(&sb, "\nSkip-reason histogram (count, reason):\n")
	for _, h := range hist {
		fmt.Fprintf(&sb, "  %4d  %s\n", h.count, h.reason)
	}
	if failed > 0 {
		fmt.Fprintf(&sb, "\nFAILURES:\n")
		sort.Strings(failNames)
		for _, n := range failNames {
			fmt.Fprintf(&sb, "  %s\n", n)
		}
	}

	t.Logf("\n%s", sb.String())

	if err := os.WriteFile(conformanceSummaryPath, []byte(sb.String()), 0o644); err != nil {
		t.Logf("warning: could not write summary to %s: %v", conformanceSummaryPath, err)
	}

	// Assertions (per Phase 4 dispatch spec):
	// (a) skip list is non-trivial — we have explicit deferrals.
	if skipped == 0 {
		t.Errorf("expected skip list to fire at least once (skipped=%d); the documented exclusion contract is empty", skipped)
	}
	// (b) failure count = 0.
	if failed > 0 {
		t.Errorf("conformance failures: %d cases failed. See test_output_phase4_conformance.txt", failed)
	}
	// (c) at least some op families pass.
	requirePassing := []string{"test_relu", "test_add", "test_matmul_2d"}
	pass := make(map[string]bool)
	for _, r := range results {
		if !r.Skipped && !r.Failed && r.MaxAbsDiff <= conformanceTol {
			pass[r.Name] = true
		}
	}
	for _, n := range requirePassing {
		if !pass[n] {
			t.Errorf("required core case %q did not pass", n)
		}
	}
}
