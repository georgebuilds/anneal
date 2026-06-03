package onnx

import (
	"strings"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// helper: int32 tensor proto for Constant-host tests.
func intConstTensor(name string, dims []int64, vals []int64) *onnxpb.TensorProto {
	out := make([]int32, len(vals))
	for i, v := range vals {
		out[i] = int32(v)
	}
	return &onnxpb.TensorProto{
		Name:      name,
		Dims:      dims,
		DataType:  int32(onnxpb.TensorProto_INT32),
		Int32Data: out,
	}
}

func TestHostShape_DeviceInput(t *testing.T) {
	arena := uop.NewArena(8)
	x := tensor.NewLeaf(arena, []int64{2, 3, 4}, uop.Dtypes.Float32, "test")
	v, err := hostShape(&Node{}, []Value{Device(x)}, NewHostState())
	if err != nil {
		t.Fatalf("hostShape err=%v", err)
	}
	if v.Kind != KindHostInts {
		t.Fatalf("kind=%d, want HostInts", v.Kind)
	}
	got := v.Ints()
	want := []int64{2, 3, 4}
	if !int64SliceEq(got, want) {
		t.Errorf("Shape=%v, want %v", got, want)
	}
}

func TestHostShape_SymbolicInput(t *testing.T) {
	arena := uop.NewArena(16)
	x := tensor.NewSymbolicBatchInput(arena, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "test")
	v, err := hostShape(&Node{}, []Value{Device(x)}, NewHostState())
	if err != nil {
		t.Fatalf("hostShape err=%v", err)
	}
	if v.Kind != KindHostSints {
		t.Fatalf("symbolic Shape kind=%d, want HostSints", v.Kind)
	}
	sh := v.Sints()
	if len(sh) != 2 {
		t.Fatalf("rank=%d, want 2", len(sh))
	}
	if _, ok := sh[0].ConstValue(); ok {
		t.Errorf("dim 0 should be symbolic")
	}
	if c, ok := sh[1].ConstValue(); !ok || c != 4 {
		t.Errorf("dim 1=%v ok=%v, want 4", c, ok)
	}
}

func TestHostSize_Concrete(t *testing.T) {
	arena := uop.NewArena(8)
	x := tensor.NewLeaf(arena, []int64{2, 3, 4}, uop.Dtypes.Float32, "test")
	v, err := hostSize(&Node{}, []Value{Device(x)}, NewHostState())
	if err != nil {
		t.Fatalf("hostSize err=%v", err)
	}
	if v.Int64() != 24 {
		t.Errorf("Size=%d, want 24", v.Int64())
	}
}

func TestHostSize_SymbolicErrors(t *testing.T) {
	arena := uop.NewArena(16)
	x := tensor.NewSymbolicBatchInput(arena, "n", 1, 32, []int64{4}, uop.Dtypes.Float32, "test")
	_, err := hostSize(&Node{}, []Value{Device(x)}, NewHostState())
	if err == nil {
		t.Fatalf("expected symbolic Size error, got nil")
	}
}

func TestHostConstant_IntScalar(t *testing.T) {
	tp := intConstTensor("c", []int64{}, []int64{42})
	node := &Node{OpType: "Constant", Attrs: map[string]Attr{
		"value": {Kind: AttrTensor, T: tp},
	}}
	v, err := hostConstant(node, nil, NewHostState())
	if err != nil {
		t.Fatalf("hostConstant err=%v", err)
	}
	if v.Kind != KindHostInt64 || v.Int64() != 42 {
		t.Errorf("Constant=%v, want HostInt64(42)", v)
	}
}

func TestHostConstant_IntVector(t *testing.T) {
	tp := intConstTensor("c", []int64{3}, []int64{1, 2, 3})
	node := &Node{OpType: "Constant", Attrs: map[string]Attr{
		"value": {Kind: AttrTensor, T: tp},
	}}
	v, err := hostConstant(node, nil, NewHostState())
	if err != nil {
		t.Fatalf("hostConstant err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3}) {
		t.Errorf("Constant=%v, want [1 2 3]", got)
	}
}

func TestHostConstant_FloatFallsThrough(t *testing.T) {
	tp := &onnxpb.TensorProto{
		Dims:      []int64{2},
		DataType:  int32(onnxpb.TensorProto_FLOAT),
		FloatData: []float32{1, 2},
	}
	node := &Node{OpType: "Constant", Attrs: map[string]Attr{
		"value": {Kind: AttrTensor, T: tp},
	}}
	_, err := hostConstant(node, nil, NewHostState())
	if err != ErrFallThroughToDevice {
		t.Fatalf("err=%v, want ErrFallThroughToDevice", err)
	}
}

func TestHostRange_ForwardStep(t *testing.T) {
	node := &Node{OpType: "Range"}
	inputs := []Value{HostInt64(0), HostInt64(5), HostInt64(1)}
	v, err := hostRange(node, inputs, NewHostState())
	if err != nil {
		t.Fatalf("hostRange err=%v", err)
	}
	got := v.Ints()
	if !int64SliceEq(got, []int64{0, 1, 2, 3, 4}) {
		t.Errorf("Range=%v, want [0..4]", got)
	}
}

func TestHostRange_BackwardStep(t *testing.T) {
	node := &Node{OpType: "Range"}
	inputs := []Value{HostInt64(5), HostInt64(0), HostInt64(-2)}
	v, err := hostRange(node, inputs, NewHostState())
	if err != nil {
		t.Fatalf("hostRange err=%v", err)
	}
	got := v.Ints()
	if !int64SliceEq(got, []int64{5, 3, 1}) {
		t.Errorf("Range=%v, want [5 3 1]", got)
	}
}

func TestHostRange_ZeroDeltaErrors(t *testing.T) {
	node := &Node{OpType: "Range"}
	inputs := []Value{HostInt64(0), HostInt64(5), HostInt64(0)}
	_, err := hostRange(node, inputs, NewHostState())
	if err == nil {
		t.Fatalf("expected delta=0 error")
	}
}

func TestHostAdd_ScalarScalar(t *testing.T) {
	v, err := hostAdd(&Node{}, []Value{HostInt64(3), HostInt64(4)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if v.Int64() != 7 {
		t.Errorf("Add=%d, want 7", v.Int64())
	}
}

func TestHostAdd_VectorScalar(t *testing.T) {
	v, err := hostAdd(&Node{}, []Value{HostInts([]int64{1, 2, 3}), HostInt64(10)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{11, 12, 13}) {
		t.Errorf("Add=%v, want [11 12 13]", got)
	}
}

func TestHostAdd_ScalarVector(t *testing.T) {
	v, err := hostAdd(&Node{}, []Value{HostInt64(10), HostInts([]int64{1, 2, 3})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{11, 12, 13}) {
		t.Errorf("Add=%v, want [11 12 13]", got)
	}
}

func TestHostAdd_VectorVector(t *testing.T) {
	v, err := hostAdd(&Node{}, []Value{HostInts([]int64{1, 2, 3}), HostInts([]int64{4, 5, 6})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{5, 7, 9}) {
		t.Errorf("Add=%v, want [5 7 9]", got)
	}
}

func TestHostSub(t *testing.T) {
	v, err := hostSub(&Node{}, []Value{HostInt64(10), HostInt64(3)}, NewHostState())
	if err != nil || v.Int64() != 7 {
		t.Fatalf("Sub=%d err=%v, want 7 nil", v.Int64(), err)
	}
}

func TestHostMul(t *testing.T) {
	v, err := hostMul(&Node{}, []Value{HostInts([]int64{2, 3, 4}), HostInt64(5)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{10, 15, 20}) {
		t.Errorf("Mul=%v, want [10 15 20]", got)
	}
}

func TestHostDiv(t *testing.T) {
	v, err := hostDiv(&Node{}, []Value{HostInt64(20), HostInt64(3)}, NewHostState())
	if err != nil || v.Int64() != 6 {
		t.Fatalf("Div=%d err=%v, want 6 nil (truncating)", v.Int64(), err)
	}
}

func TestHostNeg(t *testing.T) {
	v, err := hostNeg(&Node{}, []Value{HostInts([]int64{1, -2, 3})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{-1, 2, -3}) {
		t.Errorf("Neg=%v, want [-1 2 -3]", got)
	}
}

func TestHostGather_Vector(t *testing.T) {
	node := &Node{OpType: "Gather", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	data := HostInts([]int64{10, 20, 30, 40, 50})
	idx := HostInts([]int64{1, 3, 0})
	v, err := hostGather(node, []Value{data, idx}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{20, 40, 10}) {
		t.Errorf("Gather=%v, want [20 40 10]", got)
	}
}

func TestHostGather_NegativeIndex(t *testing.T) {
	node := &Node{OpType: "Gather", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	data := HostInts([]int64{10, 20, 30})
	idx := HostInt64(-1)
	v, err := hostGather(node, []Value{data, idx}, NewHostState())
	if err != nil || v.Int64() != 30 {
		t.Fatalf("Gather=%v err=%v, want 30 nil", v, err)
	}
}

func TestHostGather_OutOfRange(t *testing.T) {
	node := &Node{OpType: "Gather", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	data := HostInts([]int64{10, 20, 30})
	idx := HostInt64(5)
	_, err := hostGather(node, []Value{data, idx}, NewHostState())
	if err == nil {
		t.Fatalf("expected out-of-range error")
	}
}

func TestHostGather_NonZeroAxisErrors(t *testing.T) {
	node := &Node{OpType: "Gather", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 1}}}
	data := HostInts([]int64{1, 2, 3})
	idx := HostInt64(0)
	_, err := hostGather(node, []Value{data, idx}, NewHostState())
	if err == nil || !strings.Contains(err.Error(), "axis") {
		t.Fatalf("expected axis error, got %v", err)
	}
}

func TestHostConcat_Vectors(t *testing.T) {
	node := &Node{OpType: "Concat", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	v, err := hostConcat(node, []Value{
		HostInts([]int64{1, 2}),
		HostInts([]int64{3, 4, 5}),
		HostInt64(6),
	}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3, 4, 5, 6}) {
		t.Errorf("Concat=%v, want [1..6]", got)
	}
}

func TestHostConcat_NonZeroAxisErrors(t *testing.T) {
	node := &Node{OpType: "Concat", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 1}}}
	_, err := hostConcat(node, []Value{HostInts([]int64{1, 2})}, NewHostState())
	if err == nil {
		t.Fatalf("expected axis error")
	}
}

func TestHostUnsqueeze(t *testing.T) {
	// axes attr (opset 12)
	node := &Node{OpType: "Unsqueeze", Attrs: map[string]Attr{
		"axes": {Kind: AttrInts, Is: []int64{0}},
	}}
	v, err := hostUnsqueeze(node, []Value{HostInts([]int64{1, 2, 3})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	// Host payload is a flat int vector — unsqueeze is a no-op on the data.
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3}) {
		t.Errorf("Unsqueeze=%v, want [1 2 3]", got)
	}
}

func TestHostUnsqueeze_OpSet13Input(t *testing.T) {
	node := &Node{OpType: "Unsqueeze"}
	v, err := hostUnsqueeze(node, []Value{
		HostInts([]int64{1, 2, 3}),
		HostInts([]int64{0}),
	}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3}) {
		t.Errorf("Unsqueeze=%v, want [1 2 3]", got)
	}
}

func TestHostSqueeze(t *testing.T) {
	v, err := hostSqueeze(&Node{}, []Value{HostInts([]int64{1, 2, 3})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3}) {
		t.Errorf("Squeeze=%v, want [1 2 3]", got)
	}
}

func TestHostIdentity_PassThrough(t *testing.T) {
	v, err := hostIdentity(&Node{}, []Value{HostInt64(42)}, NewHostState())
	if err != nil || v.Int64() != 42 {
		t.Fatalf("Identity=%v err=%v, want 42 nil", v, err)
	}
}

func TestHostCast_IntToInt(t *testing.T) {
	node := &Node{OpType: "Cast", Attrs: map[string]Attr{
		"to": {Kind: AttrInt, I: int64(onnxpb.TensorProto_INT32)},
	}}
	v, err := hostCast(node, []Value{HostInts([]int64{1, 2, 3})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{1, 2, 3}) {
		t.Errorf("Cast=%v, want [1 2 3]", got)
	}
}

func TestHostCast_FloatFallsThrough(t *testing.T) {
	node := &Node{OpType: "Cast", Attrs: map[string]Attr{
		"to": {Kind: AttrInt, I: int64(onnxpb.TensorProto_FLOAT)},
	}}
	_, err := hostCast(node, []Value{HostInts([]int64{1, 2, 3})}, NewHostState())
	if err != ErrFallThroughToDevice {
		t.Errorf("err=%v, want ErrFallThroughToDevice", err)
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func int64SliceEq(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
