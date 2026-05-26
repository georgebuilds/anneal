package uop_test

import (
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/uop"
)

func TestOpString(t *testing.T) {
	tests := []struct {
		op   uop.Op
		want string
	}{
		{uop.OpAdd, "Add"},
		{uop.OpMul, "Mul"},
		{uop.OpConst, "Const"},
		{uop.OpSink, "Sink"},
		{uop.OpReshape, "Reshape"},
		{uop.OpReduceAxis, "ReduceAxis"},
		{uop.OpContiguousBackward, "ContiguousBackward"},
	}
	for _, tc := range tests {
		if got := tc.op.String(); got != tc.want {
			t.Errorf("Op(%d).String() = %q, want %q", int(tc.op), got, tc.want)
		}
	}
}

func TestOpStringUnknown(t *testing.T) {
	// An Op value beyond the known range should not panic and should be
	// recognisable (contains the numeric value).
	s := uop.Op(9999).String()
	if !strings.Contains(s, "9999") {
		t.Errorf("unknown op String() = %q, want to contain numeric value", s)
	}
}

func TestGroupMembership(t *testing.T) {
	tests := []struct {
		name  string
		op    uop.Op
		group uop.OpSet
		want  bool
	}{
		// Unary
		{"Exp2∈Unary", uop.OpExp2, uop.GroupUnary, true},
		{"Trunc∈Unary", uop.OpTrunc, uop.GroupUnary, true},
		{"Add∉Unary", uop.OpAdd, uop.GroupUnary, false},

		// Binary
		{"Add∈Binary", uop.OpAdd, uop.GroupBinary, true},
		{"FDiv∈Binary", uop.OpFDiv, uop.GroupBinary, true},
		{"Where∉Binary", uop.OpWhere, uop.GroupBinary, false},

		// Ternary
		{"Where∈Ternary", uop.OpWhere, uop.GroupTernary, true},
		{"MulAcc∈Ternary", uop.OpMulAcc, uop.GroupTernary, true},
		{"Add∉Ternary", uop.OpAdd, uop.GroupTernary, false},

		// ALU = union(Unary, Binary, Ternary)
		{"Exp2∈ALU", uop.OpExp2, uop.GroupALU, true},
		{"Add∈ALU", uop.OpAdd, uop.GroupALU, true},
		{"Where∈ALU", uop.OpWhere, uop.GroupALU, true},
		{"Cast∉ALU", uop.OpCast, uop.GroupALU, false},
		{"Reshape∉ALU", uop.OpReshape, uop.GroupALU, false},

		// Elementwise = ALU ∪ {Cast, Bitcast}
		{"Cast∈Elementwise", uop.OpCast, uop.GroupElementwise, true},
		{"Bitcast∈Elementwise", uop.OpBitcast, uop.GroupElementwise, true},
		{"Add∈Elementwise", uop.OpAdd, uop.GroupElementwise, true},
		{"Load∉Elementwise", uop.OpLoad, uop.GroupElementwise, false},

		// Movement
		{"Reshape∈Movement", uop.OpReshape, uop.GroupMovement, true},
		{"Permute∈Movement", uop.OpPermute, uop.GroupMovement, true},
		{"Expand∈Movement", uop.OpExpand, uop.GroupMovement, true},
		{"Pad∈Movement", uop.OpPad, uop.GroupMovement, true},
		{"Shrink∈Movement", uop.OpShrink, uop.GroupMovement, true},
		{"Flip∈Movement", uop.OpFlip, uop.GroupMovement, true},
		{"Add∉Movement", uop.OpAdd, uop.GroupMovement, false},
		{"Multi∉Movement", uop.OpMulti, uop.GroupMovement, false},

		// Comparison
		{"CmpLt∈Comparison", uop.OpCmpLt, uop.GroupComparison, true},
		{"CmpNe∈Comparison", uop.OpCmpNe, uop.GroupComparison, true},
		{"CmpEq∈Comparison", uop.OpCmpEq, uop.GroupComparison, true},
		{"Add∉Comparison", uop.OpAdd, uop.GroupComparison, false},

		// Commutative
		{"Add∈Commutative", uop.OpAdd, uop.GroupCommutative, true},
		{"Mul∈Commutative", uop.OpMul, uop.GroupCommutative, true},
		{"Sub∉Commutative", uop.OpSub, uop.GroupCommutative, false},
		{"IDiv∉Commutative", uop.OpIDiv, uop.GroupCommutative, false},

		// Associative
		{"Add∈Associative", uop.OpAdd, uop.GroupAssociative, true},
		{"Sub∉Associative", uop.OpSub, uop.GroupAssociative, false},

		// Idempotent
		{"Or∈Idempotent", uop.OpOr, uop.GroupIdempotent, true},
		{"And∈Idempotent", uop.OpAnd, uop.GroupIdempotent, true},
		{"Max∈Idempotent", uop.OpMax, uop.GroupIdempotent, true},
		{"Add∉Idempotent", uop.OpAdd, uop.GroupIdempotent, false},

		// UnsafePad
		{"Reciprocal∈UnsafePad", uop.OpReciprocal, uop.GroupUnsafePad, true},
		{"Log2∈UnsafePad", uop.OpLog2, uop.GroupUnsafePad, true},
		{"Add∉UnsafePad", uop.OpAdd, uop.GroupUnsafePad, false},

		// All
		{"Add∈All", uop.OpAdd, uop.GroupAll, true},
		{"Sink∈All", uop.OpSink, uop.GroupAll, true},
		{"PtrCat∈All", uop.OpPtrCat, uop.GroupAll, true},

		// Irreducible
		{"Const∈Irreducible", uop.OpConst, uop.GroupIrreducible, true},
		{"Range∈Irreducible", uop.OpRange, uop.GroupIrreducible, true},
		{"Add∉Irreducible", uop.OpAdd, uop.GroupIrreducible, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.group.Has(tc.op); got != tc.want {
				t.Errorf("group.Has(%s) = %v, want %v", tc.op, got, tc.want)
			}
		})
	}
}

// TestGroupALUIsUnionOfParts verifies that GroupALU == GroupUnary ∪ GroupBinary ∪ GroupTernary.
func TestGroupALUIsUnionOfParts(t *testing.T) {
	parts := []uop.OpSet{uop.GroupUnary, uop.GroupBinary, uop.GroupTernary}
	for op := uop.Op(0); int(op) < 200; op++ {
		inAny := false
		for _, g := range parts {
			if g.Has(op) {
				inAny = true
				break
			}
		}
		inALU := uop.GroupALU.Has(op)
		if inAny != inALU {
			t.Errorf("GroupALU.Has(%s)=%v but union-of-parts=%v", op, inALU, inAny)
		}
	}
}

// TestGroupElementwiseContainsALU verifies Elementwise ⊇ ALU and adds Cast/Bitcast.
func TestGroupElementwiseContainsALU(t *testing.T) {
	for op := range uop.GroupALU {
		if !uop.GroupElementwise.Has(op) {
			t.Errorf("GroupElementwise missing ALU op %s", op)
		}
	}
	for _, op := range []uop.Op{uop.OpCast, uop.OpBitcast} {
		if !uop.GroupElementwise.Has(op) {
			t.Errorf("GroupElementwise missing %s", op)
		}
	}
}
