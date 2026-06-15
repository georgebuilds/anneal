package onnx

import (
	"strings"
	"testing"

	onnxpb "github.com/georgebuilds/anneal/onnx/onnxpb"
	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/uop"
)

// ── host op arg-count / kind error branches ──────────────────────────────────

func TestHostShape_Errors(t *testing.T) {
	if _, err := hostShape(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	if _, err := hostShape(&Node{}, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

func TestHostShape_HostInputKinds(t *testing.T) {
	// Shape of a host int vector → 1-D shape [len].
	v, err := hostShape(&Node{}, []Value{HostInts([]int64{4, 5, 6})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{3}) {
		t.Errorf("HostInts shape=%v, want [3]", got)
	}
	// Shape of a host Sints vector → 1-D shape [len].
	sv, err := hostShape(&Node{}, []Value{HostSints([]shape.Sint{shape.Const(2), shape.Const(3)})}, NewHostState())
	if err != nil {
		t.Fatalf("sints err=%v", err)
	}
	if got := sv.Ints(); !int64SliceEq(got, []int64{2}) {
		t.Errorf("HostSints shape=%v, want [2]", got)
	}
}

func TestHostSize_Errors(t *testing.T) {
	if _, err := hostSize(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	if _, err := hostSize(&Node{}, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

func TestHostSize_HostKinds(t *testing.T) {
	v, err := hostSize(&Node{}, []Value{HostInts([]int64{1, 2, 3, 4})}, NewHostState())
	if err != nil || v.Int64() != 4 {
		t.Fatalf("HostInts size=%v err=%v, want 4", v, err)
	}
}

func TestHostConstant_MissingValue(t *testing.T) {
	node := &Node{OpType: "Constant", Attrs: map[string]Attr{}}
	if _, err := hostConstant(node, nil, NewHostState()); err == nil {
		t.Fatal("want error for missing value attr")
	}
}

func TestHostConstant_BadIntRawData(t *testing.T) {
	// INT64 raw_data of wrong length triggers decodeIntTensor error.
	tp := &onnxpb.TensorProto{
		Dims:     []int64{2},
		DataType: int32(onnxpb.TensorProto_INT64),
		RawData:  []byte{1, 2, 3}, // not 2*8
	}
	node := &Node{OpType: "Constant", Attrs: map[string]Attr{"value": {Kind: AttrTensor, T: tp}}}
	if _, err := hostConstant(node, nil, NewHostState()); err == nil {
		t.Fatal("want decode error for malformed raw_data")
	}
}

func TestHostRange_Errors(t *testing.T) {
	node := &Node{OpType: "Range"}
	if _, err := hostRange(node, []Value{HostInt64(0)}, NewHostState()); err == nil {
		t.Fatal("want error for too few inputs")
	}
	// non-scalar start
	bad := []Value{HostInts([]int64{1, 2}), HostInt64(5), HostInt64(1)}
	if _, err := hostRange(node, bad, NewHostState()); err == nil {
		t.Fatal("want error for non-scalar start")
	}
}

func TestHostRange_EmptyResult(t *testing.T) {
	node := &Node{OpType: "Range"}
	// start >= limit with positive delta → empty.
	v, err := hostRange(node, []Value{HostInt64(5), HostInt64(5), HostInt64(1)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); len(got) != 0 {
		t.Errorf("Range=%v, want empty", got)
	}
}

func TestHostNeg_Errors(t *testing.T) {
	if _, err := hostNeg(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	if _, err := hostNeg(&Node{}, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

func TestHostNeg_Scalar(t *testing.T) {
	v, err := hostNeg(&Node{}, []Value{HostInt64(7)}, NewHostState())
	if err != nil || v.Int64() != -7 {
		t.Fatalf("Neg=%v err=%v, want -7", v, err)
	}
}

func TestHostBinop_Errors(t *testing.T) {
	// wrong arg count
	if _, err := hostAdd(&Node{}, []Value{HostInt64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for one input")
	}
	// vector length mismatch
	in := []Value{HostInts([]int64{1, 2}), HostInts([]int64{1, 2, 3})}
	if _, err := hostAdd(&Node{}, in, NewHostState()); err == nil {
		t.Fatal("want error for length mismatch")
	}
	// unsupported kinds
	in2 := []Value{HostFloat64(1), HostFloat64(2)}
	if _, err := hostAdd(&Node{}, in2, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kinds")
	}
}

func TestHostGather_Errors(t *testing.T) {
	node := &Node{OpType: "Gather", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	// too few inputs
	if _, err := hostGather(node, []Value{HostInts([]int64{1})}, NewHostState()); err == nil {
		t.Fatal("want error for one input")
	}
	// unsupported data kind
	bad := []Value{HostFloat64(1), HostInt64(0)}
	if _, err := hostGather(node, bad, NewHostState()); err == nil {
		t.Fatal("want error for bad data kind")
	}
	// unsupported index kind
	bad2 := []Value{HostInts([]int64{1, 2}), HostFloat64(0)}
	if _, err := hostGather(node, bad2, NewHostState()); err == nil {
		t.Fatal("want error for bad index kind")
	}
	// scalar-data gather
	v, err := hostGather(node, []Value{HostInt64(99), HostInt64(0)}, NewHostState())
	if err != nil || v.Int64() != 99 {
		t.Fatalf("scalar-data gather=%v err=%v, want 99", v, err)
	}
}

func TestHostConcat_Errors(t *testing.T) {
	node := &Node{OpType: "Concat", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	// zero inputs
	if _, err := hostConcat(node, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	// unsupported kind
	if _, err := hostConcat(node, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

func TestHostConcat_Empty(t *testing.T) {
	node := &Node{OpType: "Concat", Attrs: map[string]Attr{"axis": {Kind: AttrInt, I: 0}}}
	// concat of an empty vector → empty (non-nil) result.
	v, err := hostConcat(node, []Value{HostInts([]int64{})}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); got == nil || len(got) != 0 {
		t.Errorf("Concat=%v, want empty non-nil", got)
	}
}

func TestHostUnsqueeze_Errors(t *testing.T) {
	if _, err := hostUnsqueeze(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	// unsupported input kind
	node := &Node{OpType: "Unsqueeze", Attrs: map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{0}}}}
	if _, err := hostUnsqueeze(node, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported input kind")
	}
}

func TestHostUnsqueeze_ScalarInput(t *testing.T) {
	node := &Node{OpType: "Unsqueeze", Attrs: map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{0}}}}
	v, err := hostUnsqueeze(node, []Value{HostInt64(42)}, NewHostState())
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if got := v.Ints(); !int64SliceEq(got, []int64{42}) {
		t.Errorf("Unsqueeze=%v, want [42]", got)
	}
}

func TestHostSqueeze_Errors(t *testing.T) {
	if _, err := hostSqueeze(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	if _, err := hostSqueeze(&Node{}, []Value{HostFloat64(1)}, NewHostState()); err == nil {
		t.Fatal("want error for unsupported kind")
	}
}

func TestHostSqueeze_Scalar(t *testing.T) {
	v, err := hostSqueeze(&Node{}, []Value{HostInt64(8)}, NewHostState())
	if err != nil || v.Int64() != 8 {
		t.Fatalf("Squeeze=%v err=%v, want 8", v, err)
	}
}

func TestHostIdentity_Errors(t *testing.T) {
	if _, err := hostIdentity(&Node{}, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
}

func TestHostCast_Errors(t *testing.T) {
	node := &Node{OpType: "Cast", Attrs: map[string]Attr{"to": {Kind: AttrInt, I: int64(onnxpb.TensorProto_INT32)}}}
	if _, err := hostCast(node, nil, NewHostState()); err == nil {
		t.Fatal("want error for zero inputs")
	}
	// int target but unsupported value kind → falls through to device.
	if _, err := hostCast(node, []Value{HostFloat64(1)}, NewHostState()); err != ErrFallThroughToDevice {
		t.Fatalf("err=%v, want ErrFallThroughToDevice", err)
	}
}

func TestHostCast_ScalarIntToInt(t *testing.T) {
	node := &Node{OpType: "Cast", Attrs: map[string]Attr{"to": {Kind: AttrInt, I: int64(onnxpb.TensorProto_INT64)}}}
	v, err := hostCast(node, []Value{HostInt64(11)}, NewHostState())
	if err != nil || v.Int64() != 11 {
		t.Fatalf("Cast=%v err=%v, want 11", v, err)
	}
}

// ── asHostScalar ─────────────────────────────────────────────────────────────

func TestAsHostScalar(t *testing.T) {
	if v, err := asHostScalar(HostInt64(3)); err != nil || v != 3 {
		t.Fatalf("scalar=%d err=%v", v, err)
	}
	if v, err := asHostScalar(HostInts([]int64{7})); err != nil || v != 7 {
		t.Fatalf("len-1 vec=%d err=%v", v, err)
	}
	if _, err := asHostScalar(HostInts([]int64{1, 2})); err == nil {
		t.Fatal("want error for multi-element vec")
	}
	if _, err := asHostScalar(HostFloat64(1)); err == nil {
		t.Fatal("want error for non-int kind")
	}
}

// ── decodeIntTensor branches ─────────────────────────────────────────────────

func TestDecodeIntTensor_Int32RawData(t *testing.T) {
	// INT32 via raw_data: 2 elems = 8 bytes LE.
	raw := []byte{1, 0, 0, 0, 255, 255, 255, 255} // 1, -1
	tp := &onnxpb.TensorProto{Dims: []int64{2}, DataType: int32(onnxpb.TensorProto_INT32), RawData: raw}
	out, err := decodeIntTensor(tp)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if !int64SliceEq(out, []int64{1, -1}) {
		t.Errorf("decode=%v, want [1 -1]", out)
	}
}

func TestDecodeIntTensor_Int32RawData_BadLen(t *testing.T) {
	tp := &onnxpb.TensorProto{Dims: []int64{2}, DataType: int32(onnxpb.TensorProto_INT32), RawData: []byte{1, 2, 3}}
	if _, err := decodeIntTensor(tp); err == nil {
		t.Fatal("want length-mismatch error")
	}
}

func TestDecodeIntTensor_Int64RawData(t *testing.T) {
	raw := make([]byte, 16)
	raw[0] = 5                // elem 0 = 5
	raw[8] = 0xff             // low byte of elem 1
	for i := 9; i < 16; i++ { // elem 1 = -1 (all 0xff)
		raw[i] = 0xff
	}
	tp := &onnxpb.TensorProto{Dims: []int64{2}, DataType: int32(onnxpb.TensorProto_INT64), RawData: raw}
	out, err := decodeIntTensor(tp)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if !int64SliceEq(out, []int64{5, -1}) {
		t.Errorf("decode=%v, want [5 -1]", out)
	}
}

func TestDecodeIntTensor_Int64RawData_BadLen(t *testing.T) {
	tp := &onnxpb.TensorProto{Dims: []int64{2}, DataType: int32(onnxpb.TensorProto_INT64), RawData: []byte{1, 2, 3}}
	if _, err := decodeIntTensor(tp); err == nil {
		t.Fatal("want length-mismatch error")
	}
}

func TestDecodeIntTensor_UnsupportedDtype(t *testing.T) {
	tp := &onnxpb.TensorProto{Dims: []int64{1}, DataType: int32(onnxpb.TensorProto_FLOAT)}
	if _, err := decodeIntTensor(tp); err == nil {
		t.Fatal("want error for unsupported dtype")
	}
}

// ── readAxesAttrOrInput branches ─────────────────────────────────────────────

func TestReadAxesAttrOrInput(t *testing.T) {
	// from input[1] HostInts
	node := &Node{}
	axes, err := readAxesAttrOrInput(node, []Value{HostInt64(0), HostInts([]int64{1, 2})}, 1)
	if err != nil || !int64SliceEq(axes, []int64{1, 2}) {
		t.Fatalf("from-input axes=%v err=%v", axes, err)
	}
	// from input[1] HostInt64 scalar
	axes, err = readAxesAttrOrInput(node, []Value{HostInt64(0), HostInt64(3)}, 1)
	if err != nil || !int64SliceEq(axes, []int64{3}) {
		t.Fatalf("scalar-input axes=%v err=%v", axes, err)
	}
	// unset input → falls back to attr
	node2 := &Node{Attrs: map[string]Attr{"axes": {Kind: AttrInts, Is: []int64{5}}}}
	axes, err = readAxesAttrOrInput(node2, []Value{HostInt64(0), {}}, 1)
	if err != nil || !int64SliceEq(axes, []int64{5}) {
		t.Fatalf("attr-fallback axes=%v err=%v", axes, err)
	}
	// unsupported input kind
	if _, err := readAxesAttrOrInput(node, []Value{HostInt64(0), HostFloat64(1)}, 1); err == nil {
		t.Fatal("want error for unsupported axes kind")
	}
	// no input, no attr → nil
	axes, err = readAxesAttrOrInput(&Node{}, []Value{HostInt64(0)}, 1)
	if err != nil || axes != nil {
		t.Fatalf("no-axes: axes=%v err=%v, want nil nil", axes, err)
	}
}

// ── runner.HasHandler ────────────────────────────────────────────────────────

func TestRunner_HasHandler(t *testing.T) {
	// Build a minimal valid model so Import returns a configured Runner.
	b := &singleNodeBuilder{
		opType:  "Relu",
		inputs:  []nameInfo{{Name: "x", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
		outputs: []nameInfo{{Name: "y", DType: onnxpb.TensorProto_FLOAT, Dims: []int64{2}}},
	}
	arena := uop.NewArena(32)
	r, err := Import(mustMarshalProto(t, b.build(t)), arena, "test")
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	if !r.HasHandler("Add") {
		t.Error("Add should have a handler")
	}
	if r.HasHandler("NoSuchOpXYZ") {
		t.Error("unknown op should not have a handler")
	}
}

// ── conformance SkipCount ────────────────────────────────────────────────────

func TestSkipCount(t *testing.T) {
	if SkipCount() < 0 {
		t.Fatal("skip count must be non-negative")
	}
	// SkipCount must agree with what matchSkip can resolve for a known glob.
	for pat := range conformanceSkipList {
		probe := pat
		if strings.HasSuffix(pat, "*") {
			probe = strings.TrimSuffix(pat, "*") + "anything"
		}
		if _, ok := matchSkip(probe); !ok {
			t.Errorf("matchSkip(%q) miss for listed entry %q", probe, pat)
		}
	}
}
