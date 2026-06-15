package codegen

import (
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

// buildSinkWithRanges builds SINK→END(body, range...) with the given AxisLoop
// range sizes (all concrete). Returns the sink and the arena.
func buildSinkWithRanges(sizes ...int64) (uop.UOp, *uop.Arena) {
	a := uop.NewArena(64)
	body := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	srcs := []uop.UOp{body}
	for i, sz := range sizes {
		c := a.New(uop.OpConst, uop.Dtypes.Index, nil, sz, nil)
		r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{c}, uop.RangeArg{ID: i, Type: uop.AxisLoop}, nil)
		srcs = append(srcs, r)
	}
	end := a.New(uop.OpEnd, uop.Dtypes.Void, srcs, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, nil, nil)
	return sink, a
}

// buildSinkWithSymRange builds SINK→END with a single symbolic AxisLoop range.
func buildSinkWithSymRange() (uop.UOp, *uop.Arena) {
	a := uop.NewArena(64)
	body := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	n := a.DefineVar("n", 1, 16)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{n}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{body, r}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, nil, nil)
	return sink, a
}

func notSink(a *uop.Arena) uop.UOp {
	return a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(1), nil)
}

func sinkNoEnd(a *uop.Arena) uop.UOp {
	inner := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	return a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{inner}, nil, nil)
}

// ── applyLocal guards ────────────────────────────────────────────────────────

func TestApplyLocalGuards(t *testing.T) {
	a := uop.NewArena(8)
	if got := applyLocal(notSink(a), 0, 8); got.Op() != uop.OpConst {
		t.Error("applyLocal on non-Sink must passthrough")
	}
	if got := applyLocal(sinkNoEnd(a), 0, 8); got.Op() != uop.OpSink {
		t.Error("applyLocal on Sink->non-End must passthrough")
	}
	// Axis beyond available loop ranges → target invalid → passthrough.
	sink, _ := buildSinkWithRanges(16)
	if got := applyLocal(sink, 5, 8); got.Index() != sink.Index() {
		t.Error("applyLocal with out-of-range axis must passthrough")
	}
}

// ── applyTile guards ─────────────────────────────────────────────────────────

func TestApplyTileGuards(t *testing.T) {
	a := uop.NewArena(8)
	// applyTile refuses on a non-matmul sink (tilableMatmulReduce false).
	sink, _ := buildSinkWithRanges(16)
	if got := applyTile(sink, 0, 16); got.Index() != sink.Index() {
		t.Error("applyTile on non-matmul sink must passthrough")
	}
	if got := applyTile(notSink(a), 0, 16); got.Op() != uop.OpConst {
		t.Error("applyTile on non-Sink must passthrough")
	}
}

// ── applyUpcast guards ───────────────────────────────────────────────────────

func TestApplyUpcastGuards(t *testing.T) {
	a := uop.NewArena(8)
	if got := applyUpcast(notSink(a), 0, 4); got.Op() != uop.OpConst {
		t.Error("applyUpcast on non-Sink must passthrough")
	}
	// factor <= 1 → passthrough.
	sink, _ := buildSinkWithRanges(16)
	if got := applyUpcast(sink, 0, 1); got.Index() != sink.Index() {
		t.Error("applyUpcast factor<=1 must passthrough")
	}
	if got := applyUpcast(sinkNoEnd(a), 0, 4); got.Op() != uop.OpSink {
		t.Error("applyUpcast on Sink->non-End must passthrough")
	}
	// Out-of-range axis → target invalid → passthrough.
	if got := applyUpcast(sink, 9, 4); got.Index() != sink.Index() {
		t.Error("applyUpcast out-of-range axis must passthrough")
	}
	// Symbolic range → refuse.
	symSink, _ := buildSinkWithSymRange()
	if got := applyUpcast(symSink, 0, 4); got.Index() != symSink.Index() {
		t.Error("applyUpcast on symbolic range must passthrough")
	}
}

// ── applyVectorize guards ────────────────────────────────────────────────────

func TestApplyVectorizeGuards(t *testing.T) {
	a := uop.NewArena(8)
	if got := applyVectorize(notSink(a), 0, 4); got.Op() != uop.OpConst {
		t.Error("applyVectorize on non-Sink must passthrough")
	}
	sink, _ := buildSinkWithRanges(16)
	if got := applyVectorize(sink, 0, 1); got.Index() != sink.Index() {
		t.Error("applyVectorize width<=1 must passthrough")
	}
	if got := applyVectorize(sinkNoEnd(a), 0, 4); got.Op() != uop.OpSink {
		t.Error("applyVectorize on Sink->non-End must passthrough")
	}
	if got := applyVectorize(sink, 9, 4); got.Index() != sink.Index() {
		t.Error("applyVectorize out-of-range axis must passthrough")
	}
	symSink, _ := buildSinkWithSymRange()
	if got := applyVectorize(symSink, 0, 4); got.Index() != symSink.Index() {
		t.Error("applyVectorize on symbolic range must passthrough")
	}
}

// ── containsGatherIdx ────────────────────────────────────────────────────────

func TestContainsGatherIdxInvalidNode(t *testing.T) {
	var zero uop.UOp
	if containsGatherIdx(zero) {
		t.Error("containsGatherIdx(invalid) must be false")
	}
}

func TestContainsGatherIdxPositiveAndNegative(t *testing.T) {
	a := uop.NewArena(16)
	plain := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	if containsGatherIdx(plain) {
		t.Error("plain const must not contain GatherIdx")
	}
	idx := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	g := a.New(uop.OpGatherIdx, uop.Dtypes.Float32, []uop.UOp{plain, idx}, nil, nil)
	wrap := a.New(uop.OpNeg, uop.Dtypes.Float32, []uop.UOp{g}, nil, nil)
	if !containsGatherIdx(wrap) {
		t.Error("node tree containing GatherIdx must report true")
	}
}
