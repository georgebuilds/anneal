package webgpu_test

// Slice 7d - emitIndex multi-dim stride math under sym non-outermost.
//
// The Slice 7b report flagged a latent in emitIndex's OpIndex multi-dim
// lowering: strides were computed from Buffer.Shape []int64 (with 0-sentinels
// for sym dims), so strides left of a sym dim silently became 0 - same shape
// as STOP-2 but on the input-load path instead of the final-store path.
//
// The Slice 7b C2/C3/C4 tests sidestepped this by hand-flattening the input
// index expression so emitIndex's nDims==1 branch ran instead of the
// multi-dim branch. The kernels here drive the multi-dim branch directly:
// OpIndex(buf, dim0, dim1, ...) with one dim source per axis. Without the
// Slice 7d fix the row-discriminating pattern would collapse (all rows
// would equal row 0) and max-abs-diff would be non-zero.
//
// Shapes covered:
//   - [4, n]      sym non-outermost (the original STOP-2 shape)
//   - [n, m]      two distinct syms
//   - [4, n, 4]   sym in the middle (stride for dim 0 carries `n * 4`)

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// buildEmitIndex2D builds c[i,j] = a[i,j] + b[i,j] with shape [shape0, n],
// where shape0 is concrete (or 0 for sym), n is symbolic. Each OpIndex carries
// the dim ranges as separate sources - driving emitIndex's multi-dim path.
func buildEmitIndex2D(a *uop.Arena, dim0 int64, varN string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	r0Bound := a.New(uop.OpConst, uop.Dtypes.Index, nil, dim0, nil)
	r0 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{r0Bound}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	r1 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	// Multi-dim OpIndex: paramA, dim0, dim1 - exercises emitIndex's multi-dim
	// branch (the path the Slice 7b report flagged).
	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, r0, r1}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, r0, r1}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r0, r1}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	bufs := []schedule.Buffer{
		mixedSymElemBuf(paramOut.Index(), []int64{dim0, 0}, varN, 1),
		mixedSymElemBuf(paramA.Index(), []int64{dim0, 0}, varN, 1),
		mixedSymElemBuf(paramB.Index(), []int64{dim0, 0}, varN, 1),
	}
	return schedule.ExecItem{
		Ast:     sink,
		Bufs:    bufs,
		SymVars: []string{varN},
	}
}

// TestEmitIndexSymInner_4xN drives emitIndex with shape [4, n]. Without the
// Slice 7d fix the stride for dim 0 would resolve to shape[1]==0, every row
// would collapse to row 0, and the row-discriminating pattern would diverge.
func TestEmitIndexSymInner_4xN(t *testing.T) {
	dev := requireDevice(t)

	cases := []int64{3, 5, 7}
	type result struct {
		n      int64
		maxErr float64
		load   string
	}
	var results []result

	for _, n := range cases {
		a := uop.NewArena(128)
		item := buildEmitIndex2D(a, 4, "en")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("[4,n] n=%d CompileSymKernel: %v", n, err)
		}
		// Row-discriminating values so row-collapse is loud.
		inA := make([]float32, 4*n)
		inB := make([]float32, 4*n)
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				inA[i*n+j] = float32(i*1000 + j)
				inB[i*n+j] = float32(i*7 + j*13)
			}
		}
		binding := map[string]int64{"en": n}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, 4*n, [][]float32{inA, inB})
		handle.Release()
		if err != nil {
			t.Fatalf("[4,n] n=%d dispatch: %v", n, err)
		}
		maxErr := 0.0
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				want := inA[i*n+j] + inB[i*n+j]
				got := out[i*n+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}
		// WGSL load spot-check: find a `data1[…]` load line.
		wgsl := codegen.RenderWGSL(item).WGSL
		loadLine := ""
		for _, ln := range splitLines(wgsl) {
			if strings.Contains(ln, "data1[") {
				loadLine = strings.TrimSpace(ln)
				break
			}
		}
		t.Logf("[4,n] n=%d  max-abs-diff=%.3e  load=%s", n, maxErr, loadLine)
		if maxErr != 0 {
			t.Errorf("[4,n] n=%d max-abs-diff=%g (expect 0)", n, maxErr)
			for i := int64(0); i < min64(2, 4); i++ {
				t.Logf("  row %d got: %v", i, out[i*n:(i+1)*n])
			}
		}
		results = append(results, result{n, maxErr, loadLine})
	}

	t.Logf("=== SLICE 7d emitIndex [4,n] PROOF ===")
	for _, r := range results {
		t.Logf("[n=%d] max-abs-diff=%.3e  load=%s", r.n, r.maxErr, r.load)
	}
}

// buildEmitIndex2DTwoSym builds the both-sym shape [n, m] variant.
func buildEmitIndex2DTwoSym(a *uop.Arena, varN, varM string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	defM := a.DefineVar(varM, 1, 100)
	r0 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	r1 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defM}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, r0, r1}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, r0, r1}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r0, r1}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	syms := []string{varN, varM}
	if syms[0] > syms[1] {
		syms[0], syms[1] = syms[1], syms[0]
	}
	return schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			twoSymElemBuf(paramOut.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
			twoSymElemBuf(paramA.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
			twoSymElemBuf(paramB.Index(), [2]string{varN, varM}, [2]int64{1, 1}),
		},
		SymVars: syms,
	}
}

func TestEmitIndexSym_NxM(t *testing.T) {
	dev := requireDevice(t)

	cases := []struct{ n, m int64 }{{3, 5}, {5, 3}, {7, 11}}
	type result struct {
		n, m   int64
		maxErr float64
		load   string
	}
	var results []result

	for _, c := range cases {
		a := uop.NewArena(128)
		item := buildEmitIndex2DTwoSym(a, "een", "eem")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("[n,m] n=%d m=%d CompileSymKernel: %v", c.n, c.m, err)
		}
		inA := make([]float32, c.n*c.m)
		inB := make([]float32, c.n*c.m)
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				inA[i*c.m+j] = float32(i*100 + j)
				inB[i*c.m+j] = float32(i*3 + j*7)
			}
		}
		binding := map[string]int64{"een": c.n, "eem": c.m}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, c.n*c.m, [][]float32{inA, inB})
		handle.Release()
		if err != nil {
			t.Fatalf("[n,m] n=%d m=%d dispatch: %v", c.n, c.m, err)
		}
		maxErr := 0.0
		for i := int64(0); i < c.n; i++ {
			for j := int64(0); j < c.m; j++ {
				want := inA[i*c.m+j] + inB[i*c.m+j]
				got := out[i*c.m+j]
				if e := math.Abs(float64(got - want)); e > maxErr {
					maxErr = e
				}
			}
		}
		wgsl := codegen.RenderWGSL(item).WGSL
		loadLine := ""
		for _, ln := range splitLines(wgsl) {
			if strings.Contains(ln, "data1[") {
				loadLine = strings.TrimSpace(ln)
				break
			}
		}
		t.Logf("[n,m] n=%d m=%d  max-abs-diff=%.3e  load=%s", c.n, c.m, maxErr, loadLine)
		if maxErr != 0 {
			t.Errorf("[n,m] n=%d m=%d max-abs-diff=%g (expect 0)", c.n, c.m, maxErr)
		}
		results = append(results, result{c.n, c.m, maxErr, loadLine})
	}

	t.Logf("=== SLICE 7d emitIndex [n,m] PROOF ===")
	for _, r := range results {
		t.Logf("[n=%d m=%d] max-abs-diff=%.3e  load=%s", r.n, r.m, r.maxErr, r.load)
	}
}

// buildEmitIndex3DSymMid builds c[i,j,k] = a[i,j,k] + b[i,j,k] with shape
// [4, n, 4] - sym in the middle. emitIndex must produce stride[0] = n*4 and
// stride[1] = 4. Without the fix stride[0] would resolve to shape[1]*shape[2]
// = 0*4 = 0 and all i-rows would collapse.
func buildEmitIndex3DSymMid(a *uop.Arena, varN string) schedule.ExecItem {
	defN := a.DefineVar(varN, 1, 100)
	r0Bound := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	r2Bound := a.New(uop.OpConst, uop.Dtypes.Index, nil, int64(4), nil)
	r0 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{r0Bound}, uop.RangeArg{ID: 0, Type: uop.AxisLoop}, nil)
	r1 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{defN}, uop.RangeArg{ID: 1, Type: uop.AxisLoop}, nil)
	r2 := a.New(uop.OpRange, uop.Dtypes.Index, []uop.UOp{r2Bound}, uop.RangeArg{ID: 2, Type: uop.AxisLoop}, nil)

	paramOut := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(0), nil)
	paramA := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(1), nil)
	paramB := a.New(uop.OpParam, uop.Dtypes.Float32, nil, int64(2), nil)

	indexA := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramA, r0, r1, r2}, nil, nil)
	indexB := a.New(uop.OpIndex, uop.Dtypes.Float32, []uop.UOp{paramB, r0, r1, r2}, nil, nil)
	sum := a.New(uop.OpAdd, uop.Dtypes.Float32, []uop.UOp{indexA, indexB}, nil, nil)

	store := a.New(uop.OpStore, uop.Dtypes.Void, []uop.UOp{paramOut, sum}, nil, nil)
	end := a.New(uop.OpEnd, uop.Dtypes.Void, []uop.UOp{store, r0, r1, r2}, nil, nil)
	sink := a.New(uop.OpSink, uop.Dtypes.Void, []uop.UOp{end}, uop.KernelInfo{NumParams: 3}, nil)

	return schedule.ExecItem{
		Ast: sink,
		Bufs: []schedule.Buffer{
			mixedSymElemBuf(paramOut.Index(), []int64{4, 0, 4}, varN, 1),
			mixedSymElemBuf(paramA.Index(), []int64{4, 0, 4}, varN, 1),
			mixedSymElemBuf(paramB.Index(), []int64{4, 0, 4}, varN, 1),
		},
		SymVars: []string{varN},
	}
}

func TestEmitIndexSymMid_4xNx4(t *testing.T) {
	dev := requireDevice(t)

	cases := []int64{3, 5}
	type result struct {
		n      int64
		maxErr float64
		load   string
	}
	var results []result

	for _, n := range cases {
		a := uop.NewArena(128)
		item := buildEmitIndex3DSymMid(a, "emn")

		handle, err := dev.CompileSymKernel(item)
		if err != nil {
			t.Fatalf("[4,n,4] n=%d CompileSymKernel: %v", n, err)
		}
		total := 4 * n * 4
		inA := make([]float32, total)
		inB := make([]float32, total)
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				for k := int64(0); k < 4; k++ {
					idx := i*n*4 + j*4 + k
					inA[idx] = float32(i*10000 + j*100 + k)
					inB[idx] = float32(i*3 + j*7 + k*11)
				}
			}
		}
		binding := map[string]int64{"emn": n}
		out, _, err := dev.DispatchSymKernelWithBinding(handle, item.SymVars, binding, total, [][]float32{inA, inB})
		handle.Release()
		if err != nil {
			t.Fatalf("[4,n,4] n=%d dispatch: %v", n, err)
		}
		maxErr := 0.0
		for i := int64(0); i < 4; i++ {
			for j := int64(0); j < n; j++ {
				for k := int64(0); k < 4; k++ {
					idx := i*n*4 + j*4 + k
					want := inA[idx] + inB[idx]
					got := out[idx]
					if e := math.Abs(float64(got - want)); e > maxErr {
						maxErr = e
					}
				}
			}
		}
		wgsl := codegen.RenderWGSL(item).WGSL
		loadLine := ""
		for _, ln := range splitLines(wgsl) {
			if strings.Contains(ln, "data1[") {
				loadLine = strings.TrimSpace(ln)
				break
			}
		}
		t.Logf("[4,n,4] n=%d  max-abs-diff=%.3e  load=%s", n, maxErr, loadLine)
		if maxErr != 0 {
			t.Errorf("[4,n,4] n=%d max-abs-diff=%g (expect 0)", n, maxErr)
		}
		results = append(results, result{n, maxErr, loadLine})
	}

	t.Logf("=== SLICE 7d emitIndex [4,n,4] PROOF ===")
	for _, r := range results {
		t.Logf("[n=%d] max-abs-diff=%.3e  load=%s", r.n, r.maxErr, r.load)
	}
}

// TestEmitIndexWGSLSpotCheck_4xN renders the kernel for [4, n] without
// requiring a GPU and asserts the data1 load expression uses params_n.n0
// as the stride (rather than the literal 0u that the pre-Slice-7d code path
// would have emitted). Runs even on CI without a real GPU.
func TestEmitIndexWGSLSpotCheck_4xN(t *testing.T) {
	a := uop.NewArena(128)
	item := buildEmitIndex2D(a, 4, "en")
	wgsl := codegen.RenderWGSL(item).WGSL
	var loadLine string
	for _, ln := range splitLines(wgsl) {
		if strings.Contains(ln, "data1[") {
			loadLine = strings.TrimSpace(ln)
			break
		}
	}
	if loadLine == "" {
		t.Fatalf("WGSL spot-check: no data1[ load line found\n%s", wgsl)
	}
	t.Logf("WGSL [4,n] data1 load: %s", loadLine)
	if !strings.Contains(loadLine, "params_n.n0") {
		t.Errorf("WGSL spot-check: load line missing params_n.n0 (sym-stride fix not applied)\n  got: %s", loadLine)
	}
	// Pre-fix would have a literal `* 0)` term in the load expression.
	if strings.Contains(loadLine, "* 0)") {
		t.Errorf("WGSL spot-check: load line still contains `* 0)` - sym-stride sentinel leaked\n  got: %s", loadLine)
	}
	// Surface the full WGSL on -v so the bound expression is auditable.
	t.Logf("--- full WGSL for [4, n] ---\n%s", indent(wgsl, "  "))
}

func indent(s, prefix string) string {
	lines := splitLines(s)
	return prefix + strings.Join(lines, "\n"+prefix)
}

var _ = fmt.Sprintf
