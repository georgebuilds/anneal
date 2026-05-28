package uop

import (
	"fmt"
	"strings"
)

// Op is a single opcode in the UOp IR.
// Numeric values control toposort ordering within the scheduler — do not reorder casually.
type Op int

const (
	// ── 1  defines / special ─────────────────────────────────────────────────

	OpDefineVar  Op = iota // pointer to an external (symbolic) variable
	OpBind                 // binds a DefineVar to a concrete value
	OpSpecial              // GPU dimension range (akin to a symbolic shape, but not exactly)
	OpDefineLocal          // threadgroup-local allocation
	OpDefineReg            // register-level allocation

	// ── 2  non-renderable bookkeeping ────────────────────────────────────────

	OpNoop
	OpRewriteError
	OpParam
	OpCall

	// ── 2b  program / kernel representation ──────────────────────────────────

	OpProgram // complete UOp list for a program
	OpLinear  // linearized instruction list
	OpSource  // human-readable device source (string arg)
	OpBinary  // compiled binary blob (bytes arg)

	// ── 2c  graph structure ───────────────────────────────────────────────────

	// OpSink serves three roles: tensor sink at realize(), kernel AST root
	// (carrying KernelInfo), and traversal gate between scheduler and codegen.
	OpSink
	OpAfter // passes src[0]; ensures any consumer of After runs after src[1:]
	OpGroup // NOOP that merges nodes for toposort purposes

	// ── 2d  vector construction / extraction ─────────────────────────────────

	OpGEP       // extract one lane from a vector
	OpVectorize // build a vector from scalar sources

	// ── 3  load / store ───────────────────────────────────────────────────────

	// OpIndex is the schedule-level n-dimensional tensor index used by the
	// rangeify scheduler. src[0] is the buffer or expression being indexed;
	// src[1:] are the per-dimension index expressions (one per output dim,
	// typically RANGE UOps or arithmetic over them). After rangeify propagation
	// all movement ops are dissolved, leaving INDEX only at leaf BUFFER accesses.
	// Codegen lowers INDEX(BUFFER, *indices) to a flat-offset LOAD.
	OpIndex
	OpLoad
	OpStore

	// ── 4  math ───────────────────────────────────────────────────────────────

	OpWMMA // tensor-core matrix multiply-accumulate; not elementwise

	// Unary ALU
	OpCast
	OpBitcast
	OpExp2
	OpLog2
	OpSin
	OpSqrt
	OpReciprocal
	OpNeg
	OpTrunc

	// Binary ALU
	OpAdd
	OpMul
	OpShl
	OpShr
	OpIDiv
	OpMax
	OpMod
	OpCmpLt
	OpCmpNe
	OpCmpEq
	OpXor
	OpOr
	OpAnd
	OpThreeFry
	OpSub
	OpFDiv
	OpPow

	// Ternary ALU
	OpWhere
	OpMulAcc

	// ── 5  control flow / consts / custom ────────────────────────────────────

	OpBarrier
	OpRange
	OpIf
	OpEnd
	OpEndIf

	OpVConst // vectorized constant
	OpConst  // scalar constant

	OpCustom  // emit a raw string into codegen
	OpCustomI // inline variant of Custom

	OpIns // machine instruction

	// ── 6  tensor-graph-only (not present in emitted programs) ───────────────

	OpUnique
	OpDevice
	OpAssign

	OpLUnique // per-bufferize local-unique UID

	// Scheduler behaviour modifiers
	OpContiguous
	OpContiguousBackward
	OpDetach

	// Buffer ops
	OpBufferize
	OpCopy
	OpBuffer
	OpBufferView
	OpMSelect
	OpMStack
	OpEncDec

	// The six movement ops — exist only in the tensor graph; never copied.
	OpReshape
	OpPermute
	OpExpand
	OpPad
	OpShrink
	OpFlip
	OpMulti // movement-like multi-device op

	// Reduce
	OpReduceAxis
	OpReduce
	OpAllReduce

	// Expander ops
	OpUnroll
	OpContract
	OpCat
	OpPtrCat

	opCount // sentinel — always last; equals the total number of Op values
)

// OpCount is the total number of Op values. Used for fixed-size dispatch arrays in generated matchers.
const OpCount = int(opCount)

// opNames maps each Op to its printable name.
// Entries not listed here default to "" and String() falls back to "Op(N)".
var opNames = [opCount]string{
	OpDefineVar:          "DefineVar",
	OpBind:               "Bind",
	OpSpecial:            "Special",
	OpDefineLocal:        "DefineLocal",
	OpDefineReg:          "DefineReg",
	OpNoop:               "Noop",
	OpRewriteError:       "RewriteError",
	OpParam:              "Param",
	OpCall:               "Call",
	OpProgram:            "Program",
	OpLinear:             "Linear",
	OpSource:             "Source",
	OpBinary:             "Binary",
	OpSink:               "Sink",
	OpAfter:              "After",
	OpGroup:              "Group",
	OpGEP:                "GEP",
	OpVectorize:          "Vectorize",
	OpIndex:              "Index",
	OpLoad:               "Load",
	OpStore:              "Store",
	OpWMMA:               "WMMA",
	OpCast:               "Cast",
	OpBitcast:            "Bitcast",
	OpExp2:               "Exp2",
	OpLog2:               "Log2",
	OpSin:                "Sin",
	OpSqrt:               "Sqrt",
	OpReciprocal:         "Reciprocal",
	OpNeg:                "Neg",
	OpTrunc:              "Trunc",
	OpAdd:                "Add",
	OpMul:                "Mul",
	OpShl:                "Shl",
	OpShr:                "Shr",
	OpIDiv:               "IDiv",
	OpMax:                "Max",
	OpMod:                "Mod",
	OpCmpLt:              "CmpLt",
	OpCmpNe:              "CmpNe",
	OpCmpEq:              "CmpEq",
	OpXor:                "Xor",
	OpOr:                 "Or",
	OpAnd:                "And",
	OpThreeFry:           "ThreeFry",
	OpSub:                "Sub",
	OpFDiv:               "FDiv",
	OpPow:                "Pow",
	OpWhere:              "Where",
	OpMulAcc:             "MulAcc",
	OpBarrier:            "Barrier",
	OpRange:              "Range",
	OpIf:                 "If",
	OpEnd:                "End",
	OpEndIf:              "EndIf",
	OpVConst:             "VConst",
	OpConst:              "Const",
	OpCustom:             "Custom",
	OpCustomI:            "CustomI",
	OpIns:                "Ins",
	OpUnique:             "Unique",
	OpDevice:             "Device",
	OpAssign:             "Assign",
	OpLUnique:            "LUnique",
	OpContiguous:         "Contiguous",
	OpContiguousBackward: "ContiguousBackward",
	OpDetach:             "Detach",
	OpBufferize:          "Bufferize",
	OpCopy:               "Copy",
	OpBuffer:             "Buffer",
	OpBufferView:         "BufferView",
	OpMSelect:            "MSelect",
	OpMStack:             "MStack",
	OpEncDec:             "EncDec",
	OpReshape:            "Reshape",
	OpPermute:            "Permute",
	OpExpand:             "Expand",
	OpPad:                "Pad",
	OpShrink:             "Shrink",
	OpFlip:               "Flip",
	OpMulti:              "Multi",
	OpReduceAxis:         "ReduceAxis",
	OpReduce:             "Reduce",
	OpAllReduce:          "AllReduce",
	OpUnroll:             "Unroll",
	OpContract:           "Contract",
	OpCat:                "Cat",
	OpPtrCat:             "PtrCat",
}

// String returns the human-readable name of op.
func (op Op) String() string {
	if op >= 0 && int(op) < len(opNames) {
		if s := opNames[op]; s != "" {
			return s
		}
	}
	return fmt.Sprintf("Op(%d)", int(op))
}

var opsByName map[string]Op

func init() {
	opsByName = make(map[string]Op, len(opNames))
	for i, name := range opNames {
		if name != "" {
			opsByName[name] = Op(i)
		}
	}
}

// OpFromString returns the Op with the given name (case-insensitive).
func OpFromString(s string) (Op, bool) {
	// Try exact match first
	if op, ok := opsByName[s]; ok {
		return op, true
	}
	// Try case-insensitive
	for name, op := range opsByName {
		if strings.EqualFold(name, s) {
			return op, true
		}
	}
	return 0, false
}

// ── op sets ───────────────────────────────────────────────────────────────────

// OpSet is an immutable set of Op values with O(1) membership tests.
// Use newOpSet or unionOpSets to construct; do not modify after construction.
type OpSet map[Op]struct{}

// Has reports whether op is a member of s.
func (s OpSet) Has(op Op) bool {
	_, ok := s[op]
	return ok
}

func newOpSet(ops ...Op) OpSet {
	s := make(OpSet, len(ops))
	for _, op := range ops {
		s[op] = struct{}{}
	}
	return s
}

func unionOpSets(sets ...OpSet) OpSet {
	total := 0
	for _, s := range sets {
		total += len(s)
	}
	out := make(OpSet, total)
	for _, s := range sets {
		for op := range s {
			out[op] = struct{}{}
		}
	}
	return out
}

// Group* variables mirror tinygrad's GroupOp sets.
// The rewrite engine dispatches rules by group membership; the scheduler uses
// them to enforce fusion boundaries and identify movement ops.
var (
	GroupUnary = newOpSet(
		OpExp2, OpLog2, OpSin, OpSqrt, OpReciprocal, OpNeg, OpTrunc,
	)
	GroupBinary = newOpSet(
		OpAdd, OpMul, OpShl, OpShr, OpIDiv, OpMax, OpMod,
		OpCmpLt, OpCmpNe, OpCmpEq,
		OpXor, OpOr, OpAnd, OpThreeFry, OpSub, OpFDiv, OpPow,
	)
	GroupTernary    = newOpSet(OpWhere, OpMulAcc)
	GroupALU        = unionOpSets(GroupUnary, GroupBinary, GroupTernary)
	GroupElementwise = unionOpSets(GroupALU, newOpSet(OpCast, OpBitcast))

	GroupDefines    = newOpSet(OpParam, OpDefineLocal, OpDefineReg)
	GroupIrreducible = newOpSet(OpConst, OpDefineVar, OpSpecial, OpRange)
	GroupMovement   = newOpSet(OpReshape, OpExpand, OpPermute, OpPad, OpShrink, OpFlip)
	GroupBuffer     = newOpSet(OpLoad, OpStore, OpConst, OpDefineVar)

	// Algebraic properties used by symbolic simplification rules.
	GroupCommutative = newOpSet(OpAdd, OpMul, OpMax, OpCmpNe, OpCmpEq, OpXor, OpAnd, OpOr)
	GroupAssociative = newOpSet(OpAdd, OpMul, OpAnd, OpOr, OpMax)
	GroupIdempotent  = newOpSet(OpOr, OpAnd, OpMax)
	GroupComparison  = newOpSet(OpCmpLt, OpCmpNe, OpCmpEq)

	// Ops where f(0) ≠ 0; padding with zero is unsafe across these.
	GroupUnsafePad = newOpSet(OpReciprocal, OpLog2, OpExp2, OpIDiv, OpPow)

	GroupAll = func() OpSet {
		s := make(OpSet, int(opCount))
		for i := Op(0); i < opCount; i++ {
			s[i] = struct{}{}
		}
		return s
	}()
)
