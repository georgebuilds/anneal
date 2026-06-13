package codegen

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// ── spreadWorkgroupCount ─────────────────────────────────────────────────────

func TestSpreadWorkgroupCount(t *testing.T) {
	cases := []struct {
		name string
		in   [3]int
		want [3]int
	}{
		{"no_spread_small", [3]int{10, 1, 1}, [3]int{10, 1, 1}},
		{"no_spread_y_used", [3]int{100000, 2, 1}, [3]int{100000, 2, 1}},
		{"spread_into_y", [3]int{100000, 1, 1}, [3]int{65535, 2, 1}},
		// 65535*65535+1 workgroups overflow Y too and cascade into Z.
		{"spread_into_z", [3]int{65535*65535 + 1, 1, 1}, [3]int{65535, 65535, 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := &lowerer{workgroupCount: tc.in}
			l.spreadWorkgroupCount()
			if l.workgroupCount != tc.want {
				t.Errorf("spreadWorkgroupCount(%v) = %v, want %v", tc.in, l.workgroupCount, tc.want)
			}
		})
	}
}

// ── lowerImageSlot defensive panics ──────────────────────────────────────────

// newImageSlotLowerer builds the minimal lowerer + loopRange fixtures for
// driving lowerImageSlot directly: one AxisLoop range of size 4 and an
// image-typed output buffer.
func newImageSlotLowerer() (*lowerer, uop.UOp, uop.UOp) {
	a := uop.NewArena(64)
	bound := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	r := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{bound}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	body := a.New(uop.OpConst, uop.Dtypes.Float32, nil, float64(1), nil)
	l := &lowerer{
		item:   schedule.ExecItem{Bufs: []schedule.Buffer{{DType: uop.Dtypes.ImageFloat32}}},
		exprOf: make(map[uint32]string),
	}
	return l, r, body
}

// TestLowerImageSlot_SymbolicStridePanics pins the fail-loud guard: a
// symbolic stride leaking into the image slot path (which the lowerSink
// branch excludes by construction) must panic, not render garbage indices.
func TestLowerImageSlot_SymbolicStridePanics(t *testing.T) {
	l, r, body := newImageSlotLowerer()
	defer func() {
		if rec := recover(); rec == nil {
			t.Errorf("lowerImageSlot with symbolic stride did not panic")
		}
	}()
	l.lowerImageSlot(body, []uop.UOp{r}, []strideAcc{{constPart: 1, symPart: "params_n.n0"}}, 4)
}

// TestLowerImageSlot_ZeroStridePanics pins the sentinel-leak guard: a
// concrete stride of 0 is the legacy zero-default sentinel and must panic.
func TestLowerImageSlot_ZeroStridePanics(t *testing.T) {
	l, r, body := newImageSlotLowerer()
	defer func() {
		if rec := recover(); rec == nil {
			t.Errorf("lowerImageSlot with zero stride did not panic")
		}
	}()
	l.lowerImageSlot(body, []uop.UOp{r}, []strideAcc{{constPart: 0}}, 4)
}

// TestLowerImageSlot_ConstPlaceholderRange pins the size-1-dim placeholder
// handling: an OpConst loopRange entry (freshRanges collapses size-1 dims to
// const 0) registers "0u" and contributes no per-lane let.
func TestLowerImageSlot_ConstPlaceholderRange(t *testing.T) {
	l, r, body := newImageSlotLowerer()
	cst := r.Arena().New(uop.OpConst, uop.Dtypes.Index, nil, int64(0), nil)
	instrs := l.lowerImageSlot(body, []uop.UOp{cst, r},
		[]strideAcc{{constPart: 4}, {constPart: 1}}, 4)
	if got := l.exprOf[cst.Index()]; got != "0u" {
		t.Errorf("const placeholder exprOf = %q, want %q", got, "0u")
	}
	var lets int
	for _, ins := range instrs {
		if ins.Kind == InstrLet && strings.HasPrefix(ins.Name, "r") {
			lets++
		}
	}
	if lets != 1 {
		t.Errorf("expected exactly 1 per-lane range let (const placeholder skipped), got %d", lets)
	}
}

// ── sinkOutputIsImage structural early-outs ──────────────────────────────────

func TestSinkOutputIsImage_Structure(t *testing.T) {
	a := uop.NewArena(128)
	f32 := uop.Dtypes.Float32
	img := uop.Dtypes.ImageFloat32

	mkChain := func(paramDType *uop.DType) uop.UOp {
		param := a.New(uop.OpParam, paramDType, nil, int64(0), nil)
		idx := a.New(uop.OpIndex, uop.Dtypes.Void, []uop.UOp{param}, nil, nil)
		body := a.New(uop.OpConst, f32, nil, float64(1), nil)
		store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{idx, body}, nil, nil)
		end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store}, nil, nil)
		return a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, nil, nil)
	}

	cst := a.New(uop.OpConst, f32, nil, float64(0), nil)
	sinkOverConst := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{cst}, nil, nil)
	endOverConst := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{cst}, nil, nil)
	sinkBadStore := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{endOverConst}, nil, nil)

	cases := []struct {
		name string
		sink uop.UOp
		want bool
	}{
		{"not_a_sink", cst, false},
		{"sink_without_end", sinkOverConst, false},
		{"end_without_store", sinkBadStore, false},
		{"scalar_output", mkChain(f32), false},
		{"image_output", mkChain(img), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sinkOutputIsImage(tc.sink); got != tc.want {
				t.Errorf("sinkOutputIsImage(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// ── legacy image InstrStore cascade (symbolic-only path) ─────────────────────

// TestRenderInstrs_LegacyImageStoreCascade pins the legacy per-lane storage
// cascade that SYMBOLIC image kernels still lower through (static image
// kernels take the InstrImgLane* slot dispatch instead). The cascade's
// static-swizzle form is what keeps naga from degrading the component write;
// its aligned-row-stride constraint is documented in LIMITATIONS.md.
func TestRenderInstrs_LegacyImageStoreCascade(t *testing.T) {
	a := uop.NewArena(64)
	ast := a.New(uop.OpSink, uop.Dtypes.Void, nil, uop.KernelInfo{NumParams: 1}, nil)
	item := schedule.ExecItem{
		Ast:  ast,
		Bufs: []schedule.Buffer{{DType: uop.Dtypes.ImageFloat32}},
	}
	instrs := []Instr{{Kind: InstrStore, DType: uop.Dtypes.ImageFloat32, IndexExpr: "gid_x", Expr: "1.0"}}
	wgsl := renderInstrs(instrs, item, [3]int{64, 1, 1}, [3]int{1, 1, 1})
	for _, want := range []string{
		"let _img_slot = u32(gid_x) / 4u;",
		"let _img_lane = u32(gid_x) % 4u;",
		"if (_img_lane == 0u) { data0[_img_slot].x = _img_val; }",
		"else { data0[_img_slot].w = _img_val; }",
	} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("legacy image store cascade missing %q\nfull shader:\n%s", want, wgsl)
		}
	}
}
