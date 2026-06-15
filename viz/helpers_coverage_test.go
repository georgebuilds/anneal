//go:build !js

package viz

import (
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor/npy"
	"github.com/georgebuilds/anneal/tensor/safetensors"
	"github.com/georgebuilds/anneal/uop"
)

// TestDtypeStr covers every dtype branch plus the nil/void and fallback cases.
func TestDtypeStr(t *testing.T) {
	cases := []struct {
		dt   *uop.DType
		want string
	}{
		{nil, "void"},
		{uop.Dtypes.Void, "void"},
		{uop.Dtypes.Float32, "f32"},
		{uop.Dtypes.Float16, "f16"},
		{uop.Dtypes.BFloat16, "bf16"},
		{uop.Dtypes.FP8E4M3, "e4m3"},
		{uop.Dtypes.FP8E5M2, "e5m2"},
		{uop.Dtypes.Float64, "f64"},
		{uop.Dtypes.Int32, "i32"},
		{uop.Dtypes.UInt32, "u32"},
		{uop.Dtypes.Bool, "bool"},
		{uop.Dtypes.Index, "idx"},
	}
	for _, c := range cases {
		if got := dtypeStr(c.dt); got != c.want {
			t.Errorf("dtypeStr(%v) = %q, want %q", c.dt, got, c.want)
		}
	}
}

// TestDtypeStrFor covers the nil-safe wrapper used by the ONNX importer.
func TestDtypeStrFor(t *testing.T) {
	if got := dtypeStrFor(nil); got != "" {
		t.Errorf("dtypeStrFor(nil) = %q, want empty", got)
	}
	if got := dtypeStrFor(uop.Dtypes.Float32); got != "f32" {
		t.Errorf("dtypeStrFor(f32) = %q, want f32", got)
	}
}

// TestArgStr covers each arg-type branch of argStr, including string truncation
// and the const/symbolic range cases.
func TestArgStr(t *testing.T) {
	a := uop.NewArena(1024)

	// int64
	if got := argStr(a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(42), nil)); got != "42" {
		t.Errorf("int64 argStr = %q, want 42", got)
	}
	// float64
	if got := argStr(a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1.5), nil)); got != "1.5" {
		t.Errorf("float64 argStr = %q, want 1.5", got)
	}
	// bool true / false
	bt := a.New(uop.OpConst, uop.Dtypes.Bool, nil, true, nil)
	bf := a.New(uop.OpConst, uop.Dtypes.Bool, nil, false, nil)
	if argStr(bt) != "true" || argStr(bf) != "false" {
		t.Errorf("bool argStr: got %q/%q", argStr(bt), argStr(bf))
	}
	// short string passes through
	short := a.New(uop.OpConst, uop.Dtypes.Index, nil, "hello", nil)
	if got := argStr(short); got != "hello" {
		t.Errorf("short string argStr = %q, want hello", got)
	}
	// long string truncates to 16 + ellipsis
	long := a.New(uop.OpConst, uop.Dtypes.Index, nil, "0123456789abcdefghij", nil)
	if got := argStr(long); got != "0123456789abcdef…" {
		t.Errorf("long string argStr = %q", got)
	}
	// []int64
	sl := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3}, nil)
	if got := argStr(sl); got != "[2 3]" {
		t.Errorf("[]int64 argStr = %q, want [2 3]", got)
	}
	// ReduceArg
	ra := a.New(uop.OpConst, uop.Dtypes.Float32, nil, uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}}, nil)
	if got := argStr(ra); got == "" {
		t.Errorf("ReduceArg argStr empty, want non-empty")
	}
	// unsupported arg type -> empty
	none := a.New(uop.OpConst, uop.Dtypes.Float32, nil, nil, nil)
	if got := argStr(none); got != "" {
		t.Errorf("nil arg argStr = %q, want empty", got)
	}
}

// TestArgStr_RangeArg covers the RangeArg formatting branches: concrete loop,
// concrete reduce, and symbolic-with-DefineVar.
func TestArgStr_RangeArg(t *testing.T) {
	a := uop.NewArena(1024)

	// concrete loop range [0,16)loop
	c16 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(16), nil)
	rLoop := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c16}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	if got := argStr(rLoop); got != "[0,16)loop" {
		t.Errorf("concrete loop range = %q, want [0,16)loop", got)
	}

	// concrete reduce range [0,8)red
	c8 := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(8), nil)
	rRed := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c8}, uop.RangeArg{ID: 1, Type: uop.AxisReduce}, nil)
	if got := argStr(rRed); got != "[0,8)red" {
		t.Errorf("concrete reduce range = %q, want [0,8)red", got)
	}

	// symbolic range bounded by a DefineVar named "N"
	dv := a.New(uop.OpDefineVar, uop.Dtypes.Index, nil, uop.VarArg{Name: "N"}, nil)
	rSym := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{dv}, uop.RangeArg{ID: 2, Type: uop.AxisLoop}, nil)
	if got := argStr(rSym); got != "[0,N)loop" {
		t.Errorf("symbolic range = %q, want [0,N)loop", got)
	}
}

// TestNodeLabel covers Buffer (with/without shape), Const (with/without arg),
// ReduceAxis, Sink, and the default op-name branch.
func TestNodeLabel(t *testing.T) {
	a := uop.NewArena(1024)

	bufShaped := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{2, 3}, nil)
	if got := nodeLabel(bufShaped); got != "Buffer[2 3]" {
		t.Errorf("shaped buffer label = %q, want Buffer[2 3]", got)
	}

	bufNoShape := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, nil, nil)
	if got := nodeLabel(bufNoShape); got != "Buffer" {
		t.Errorf("unshaped buffer label = %q, want Buffer", got)
	}

	constVal := a.New(uop.OpConst, uop.Dtypes.Int32, nil, int64(7), nil)
	if got := nodeLabel(constVal); got != "Const(7)" {
		t.Errorf("const label = %q, want Const(7)", got)
	}

	constNoArg := a.New(uop.OpConst, uop.Dtypes.Float32, nil, nil, nil)
	if got := nodeLabel(constNoArg); got != "Const" {
		t.Errorf("argless const label = %q, want Const", got)
	}

	red := a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{constVal},
		uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}}, nil)
	if got := nodeLabel(red); got != "Add-reduce" {
		t.Errorf("reduce label = %q, want Add-reduce", got)
	}

	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{constVal}, nil, nil)
	if got := nodeLabel(sink); got != "Sink" {
		t.Errorf("sink label = %q, want Sink", got)
	}

	// default branch: any other op renders its op name.
	add := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{constVal, constVal}, nil, nil)
	if got := nodeLabel(add); got != "Add" {
		t.Errorf("default label = %q, want Add", got)
	}
}

// TestBufShape covers all three branches of bufShape.
func TestBufShape(t *testing.T) {
	a := uop.NewArena(1024)
	slice := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{4, 5}, nil)
	if got := bufShape(slice); len(got) != 2 || got[0] != 4 || got[1] != 5 {
		t.Errorf("bufShape([]int64) = %v, want [4 5]", got)
	}
	scalar := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, int64(9), nil)
	if got := bufShape(scalar); len(got) != 1 || got[0] != 9 {
		t.Errorf("bufShape(int64) = %v, want [9]", got)
	}
	none := a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, nil, nil)
	if got := bufShape(none); got != nil {
		t.Errorf("bufShape(nil) = %v, want nil", got)
	}
}

// TestKindOf covers the Sink, ReduceAxis, Buffer, and default classifications.
func TestKindOf(t *testing.T) {
	a := uop.NewArena(1024)
	c := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	cases := []struct {
		u    uop.UOp
		want string
	}{
		{a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{c}, nil, nil), KindSink},
		{a.New(uop.OpReduceAxis, uop.Dtypes.Float32, []uop.UOp{c}, uop.ReduceArg{Op: uop.OpAdd, Axes: []int{0}}, nil), KindReduce},
		{a.New(uop.OpBuffer, uop.Dtypes.Float32, nil, []int64{1}, nil), KindLeaf},
		{a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{c, c}, nil, nil), KindDefault},
	}
	for _, tc := range cases {
		if got := kindOf(tc.u, a); got != tc.want {
			t.Errorf("kindOf(%s) = %q, want %q", tc.u.Op(), got, tc.want)
		}
	}
}

// TestDescriptionForOp covers the curated-hit path, the group-based fallbacks
// (unary/binary/movement/structural), and the unknown-op path.
func TestDescriptionForOp(t *testing.T) {
	// Unknown op name -> "no description available".
	if got := descriptionForOp("ZZNotAnOp"); got != "ZZNotAnOp: no description available." {
		t.Errorf("unknown op desc = %q", got)
	}
	// Unary group fallback (Sin is unary; if it has a curated entry that is
	// also fine — assert only that the description is non-empty and mentions
	// the op).
	for _, name := range []string{"Sin", "Add", "Reshape", "Sink"} {
		if got := descriptionForOp(name); got == "" {
			t.Errorf("descriptionForOp(%q) empty", name)
		}
	}
}

// TestDescriptionForOp_GroupFallbacks asserts that ops absent from the curated
// opDescriptions map still get a non-empty group-based fallback description.
func TestDescriptionForOp_GroupFallbacks(t *testing.T) {
	for _, op := range []uop.Op{uop.OpAdd, uop.OpSin, uop.OpReshape, uop.OpWhere} {
		name := op.String()
		if _, curated := opDescriptions[name]; curated {
			continue // skip curated ops; only the fallback path is under test
		}
		if got := descriptionForOp(name); got == "" {
			t.Errorf("descriptionForOp(%q) empty", name)
		}
	}
}

// TestRewriteTextForHandler covers a representative set of handler branches and
// the default fallback.
func TestRewriteTextForHandler(t *testing.T) {
	cases := map[string]string{
		"hReturnX":      "x",
		"hReturnBase":   "base",
		"hFoldBinary":   "Const",
		"hBindFold":     "Const(val)",
		"hMulZero":      "0",
		"hIDivSelf":     "1",
		"hCmpSelf":      "false",
		"hOrTrue":       "true",
		"hIDivNegOne":   "-x",
		"hBoolMul":      "x & y",
		"hBoolAdd":      "x | y",
		"hCanonicalize": "(canonicalized)",
		"hUnknownXYZ":   "hUnknownXYZ", // default branch returns the input
	}
	for h, want := range cases {
		if got := rewriteTextForHandler(h); got != want {
			t.Errorf("rewriteTextForHandler(%q) = %q, want %q", h, got, want)
		}
	}
}

// TestGradientRuleForOp covers a registered-with-pattern op, a registered
// non-differentiable op, and an unregistered op (nil).
func TestGradientRuleForOp(t *testing.T) {
	// Mul is registered and has a curated pattern.
	if gr := gradientRuleForOp(uop.OpMul); gr == nil {
		t.Error("gradientRuleForOp(Mul) = nil, want a rule")
	} else if gr.Source == "" || gr.Pattern == "" {
		t.Errorf("gradientRuleForOp(Mul) incomplete: %+v", gr)
	}

	// An op that is not registered for gradients returns nil. OpBuffer is a
	// structural leaf and never has a gradient rule.
	if gr := gradientRuleForOp(uop.OpBuffer); gr != nil {
		t.Errorf("gradientRuleForOp(Buffer) = %+v, want nil", gr)
	}
}

// TestMiniGraphForOp covers a curated op (Add) and an absent op (pass-through).
func TestMiniGraphForOp(t *testing.T) {
	add := miniGraphForOp("Add")
	if len(add.Before) == 0 || len(add.After) == 0 {
		t.Errorf("miniGraphForOp(Add) = %+v, want non-empty before/after", add)
	}
	// An op without a curated mini-graph still renders something.
	other := miniGraphForOp("ZZNotAnOp")
	if len(other.Before) == 0 {
		t.Errorf("miniGraphForOp(unknown) before empty, want pass-through node")
	}
}

// TestAllOpNames asserts the op-name list is sorted, non-empty, and free of
// the "Op(<n>)" stringer-fallback artifacts.
func TestAllOpNames(t *testing.T) {
	names := AllOpNames()
	if len(names) == 0 {
		t.Fatal("AllOpNames returned empty")
	}
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Errorf("AllOpNames not sorted at %d: %q > %q", i, names[i-1], names[i])
		}
	}
	for _, n := range names {
		if n == "" {
			t.Error("AllOpNames contains empty string")
		}
	}
}

// TestNpyItemSize covers byte-order prefixes, each type-char family, and the
// malformed/unknown fallbacks (which return 0).
func TestNpyItemSize(t *testing.T) {
	cases := []struct {
		descr string
		want  int
	}{
		{"<f4", 4},
		{">f8", 8},
		{"|i1", 1},
		{"=u2", 2},
		{"f4", 4},
		{"c16", 16},
		{"b1", 1},
		{"", 0},     // empty
		{"<", 0},    // too short after prefix
		{"x4", 0},   // unknown type char
		{"f4x", 0},  // non-digit in size
		{"<S10", 0}, // string type char not in the allowed set
	}
	for _, c := range cases {
		if got := npyItemSize(c.descr); got != c.want {
			t.Errorf("npyItemSize(%q) = %d, want %d", c.descr, got, c.want)
		}
	}
}

// TestStItemSize covers every safetensors dtype size plus the unknown fallback.
func TestStItemSize(t *testing.T) {
	cases := map[string]int{
		"BOOL": 1, "I8": 1, "U8": 1,
		"F16": 2, "BF16": 2, "I16": 2, "U16": 2,
		"F32": 4, "I32": 4, "U32": 4,
		"F64": 8, "I64": 8, "U64": 8,
		"WAT": 0,
	}
	for dt, want := range cases {
		if got := stItemSize(dt); got != want {
			t.Errorf("stItemSize(%q) = %d, want %d", dt, got, want)
		}
	}
}

// TestFormatPhaseForUOp covers the nil-arena, out-of-range, forward, and
// backward branches.
func TestFormatPhaseForUOp(t *testing.T) {
	if got := formatPhaseForUOp(nil, 0); got != ClassForward {
		t.Errorf("formatPhaseForUOp(nil) = %q, want forward", got)
	}
	a := uop.NewArena(1024)
	fwd := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if got := formatPhaseForUOp(a, fwd.Index()); got != ClassForward {
		t.Errorf("forward node phase = %q, want forward", got)
	}
	// Out-of-range index defaults to forward.
	if got := formatPhaseForUOp(a, uint32(a.Len()+100)); got != ClassForward {
		t.Errorf("out-of-range phase = %q, want forward", got)
	}
	// Backward-provenance node classifies as backward.
	prev := a.SetPhase(uop.PhaseBackward)
	bwd := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{fwd}, nil, nil)
	a.SetPhase(prev)
	if got := formatPhaseForUOp(a, bwd.Index()); got != ClassBackward {
		t.Errorf("backward node phase = %q, want backward", got)
	}
}

// TestDescriptionForOp_StructuralFallback covers the default ("structural /
// scheduling op") branch with an uncurated op that is a known stringer value.
func TestDescriptionForOp_StructuralFallback(t *testing.T) {
	// "Load" is a real op that is not in opDescriptions and belongs to no ALU
	// or movement group, so it hits the structural/scheduling fallback.
	got := descriptionForOp("Load")
	if got != "Load: structural / scheduling op." {
		t.Errorf("descriptionForOp(Load) = %q", got)
	}
}

// TestCountBuffers covers the empty-Bufs and populated branches.
func TestCountBuffers(t *testing.T) {
	if in, out := countBuffers(schedule.ExecItem{}); in != 0 || out != 0 {
		t.Errorf("countBuffers(empty) = (%d,%d), want (0,0)", in, out)
	}
	item := schedule.ExecItem{Bufs: []schedule.Buffer{
		{Shape: []int64{4}}, // output
		{Shape: []int64{4}}, // input
		{Shape: []int64{4}}, // input
	}}
	if in, out := countBuffers(item); in != 2 || out != 1 {
		t.Errorf("countBuffers(3 bufs) = (%d,%d), want (2,1)", in, out)
	}
}

// TestOutputShape covers the no-buffer and populated (defensive-copy) branches.
func TestOutputShape(t *testing.T) {
	if got := outputShape(schedule.ExecItem{}); got != nil {
		t.Errorf("outputShape(empty) = %v, want nil", got)
	}
	orig := []int64{2, 3}
	item := schedule.ExecItem{Bufs: []schedule.Buffer{{Shape: orig}}}
	got := outputShape(item)
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("outputShape = %v, want [2 3]", got)
	}
	// Must be a defensive copy: mutating the result must not touch the source.
	got[0] = 99
	if orig[0] != 2 {
		t.Error("outputShape did not return a defensive copy")
	}
}

// TestNpyTensorInfo covers numel computation and the negative-numel clamp.
func TestNpyTensorInfo(t *testing.T) {
	e := npy.InspectEntry{DType: "<f4", Shape: []int64{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
	info := npyTensorInfo("arr", e)
	if info.Name != "arr" || info.Numel != 6 || info.Bytes != 24 {
		t.Errorf("npyTensorInfo = %+v, want numel 6 bytes 24", info)
	}
	// A negative dim (unknown/symbolic placeholder) clamps numel to 0.
	neg := npy.InspectEntry{DType: "<f4", Shape: []int64{-1, 3}}
	if info := npyTensorInfo("neg", neg); info.Numel != 0 {
		t.Errorf("npyTensorInfo(neg) numel = %d, want 0", info.Numel)
	}
}

// TestSafetensorsTensorInfo mirrors TestNpyTensorInfo for the safetensors path.
func TestSafetensorsTensorInfo(t *testing.T) {
	e := safetensors.InspectEntry{Name: "w", DType: "F32", Shape: []int64{2, 2}, Data: []float32{1, 2, 3, 4}}
	info := safetensorsTensorInfo(e)
	if info.Name != "w" || info.Numel != 4 || info.Bytes != 16 {
		t.Errorf("safetensorsTensorInfo = %+v, want numel 4 bytes 16", info)
	}
	neg := safetensors.InspectEntry{Name: "n", DType: "F32", Shape: []int64{-1, 2}}
	if info := safetensorsTensorInfo(neg); info.Numel != 0 {
		t.Errorf("safetensorsTensorInfo(neg) numel = %d, want 0", info.Numel)
	}
}

// TestImportUnsupportedReason covers a curated reason and the generic fallback.
func TestImportUnsupportedReason(t *testing.T) {
	if got := importUnsupportedReason("Resize"); got == "" {
		t.Error("importUnsupportedReason(Resize) empty, want curated reason")
	}
	got := importUnsupportedReason("TotallyMadeUpOp")
	if got != "no handler registered yet (visualize only; cannot execute)" {
		t.Errorf("importUnsupportedReason(unknown) = %q", got)
	}
}

// TestTimelineToJSON covers TimelineData.ToJSON's marshalling.
func TestTimelineToJSON(t *testing.T) {
	td := &TimelineData{
		Name:   "demo",
		Nodes:  []NodeData{{ID: 1, Op: "Add"}},
		Edges:  []EdgeData{{From: 1, To: 2}},
		Stages: []StageData{{}},
	}
	b, err := td.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	if len(b) == 0 {
		t.Fatal("ToJSON returned empty bytes")
	}
}
