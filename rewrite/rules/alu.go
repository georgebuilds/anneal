package rules

import (
	"math"

	"github.com/georgebuilds/anneal/uop"
)

// asInt converts a constant arg to int64. Supports int64, bool.
func asInt(v any) (int64, bool) {
	switch v := v.(type) {
	case int64:
		return v, true
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// asFloat converts a constant arg to float64. Supports float64, int64, bool.
func asFloat(v any) (float64, bool) {
	switch v := v.(type) {
	case float64:
		return v, true
	case int64:
		return float64(v), true
	case bool:
		if v {
			return 1.0, true
		}
		return 0.0, true
	}
	return 0, false
}

// asBool converts a constant arg to bool.
func asBool(v any) (bool, bool) {
	switch v := v.(type) {
	case bool:
		return v, true
	case int64:
		return v != 0, true
	case float64:
		return v != 0, true
	}
	return false, false
}

// floorDiv computes Python-style floor division (rounds toward −∞).
func floorDiv(a, b int64) int64 {
	q := a / b
	// When signs differ and there is a remainder, floor is one below truncation.
	if (a^b) < 0 && q*b != a {
		q--
	}
	return q
}

// floorMod computes Python-style floor modulo (result has the sign of b).
func floorMod(a, b int64) int64 {
	r := a % b
	if r != 0 && (r^b) < 0 {
		r += b
	}
	return r
}

// truncateInt applies dtype-specific integer overflow/truncation semantics.
// Result is stored as int64 in two's-complement representation.
func truncateInt(v int64, dtype *uop.DType) int64 {
	switch dtype.BitSize() {
	case 1: // bool
		if v != 0 {
			return 1
		}
		return 0
	case 8:
		if dtype.IsUnsigned() {
			return int64(uint8(v))
		}
		return int64(int8(v))
	case 16:
		if dtype.IsUnsigned() {
			return int64(uint16(v))
		}
		return int64(int16(v))
	case 32:
		if dtype.IsUnsigned() {
			return int64(uint32(v))
		}
		return int64(int32(v))
	}
	return v // 64-bit: no truncation needed
}

// truncateFloat rounds to the precision of dtype (float32 → float64 roundtrip).
func truncateFloat(v float64, dtype *uop.DType) float64 {
	if dtype == uop.Dtypes.Float32 {
		return float64(float32(v))
	}
	return v
}

// castValue converts constant arg v from fromDtype to toDtype, returning the
// correctly typed value for storage in a Const UOp arg.
func castValue(v any, fromDtype, toDtype *uop.DType) any {
	switch {
	case toDtype.IsBool():
		if fromDtype.IsFloat() {
			f, _ := asFloat(v)
			return f != 0
		}
		i, _ := asInt(v)
		return i != 0
	case toDtype.IsFloat():
		if fromDtype.IsFloat() {
			f, _ := asFloat(v)
			return truncateFloat(f, toDtype)
		}
		if fromDtype.IsBool() {
			b, _ := asBool(v)
			if b {
				return truncateFloat(1.0, toDtype)
			}
			return truncateFloat(0.0, toDtype)
		}
		i, _ := asInt(v)
		return truncateFloat(float64(i), toDtype)
	case toDtype.IsInt():
		if fromDtype.IsFloat() {
			f, _ := asFloat(v)
			return truncateInt(int64(math.Trunc(f)), toDtype)
		}
		i, _ := asInt(v)
		return truncateInt(i, toDtype)
	}
	return v
}

// execALU evaluates op on concrete constant values, matching tinygrad's python_alu
// semantics. Returns (result, true) on success, (nil, false) on undefined (e.g. div/0)
// or unsupported ops.
//
// vals must have the correct arity for op.
// Results are untruncated; the caller applies truncateInt/truncateFloat as needed.
// Comparison results are bool; arithmetic results are int64 or float64.
func execALU(op uop.Op, dtype *uop.DType, vals []any) (any, bool) {
	// Determine whether to use float arithmetic.
	useFloat := false
	for _, v := range vals {
		if _, ok := v.(float64); ok {
			useFloat = true
			break
		}
	}
	if dtype != nil && dtype.IsFloat() {
		useFloat = true
	}

	if useFloat {
		return execALUFloat(op, vals)
	}
	return execALUInt(op, vals)
}

func execALUFloat(op uop.Op, vals []any) (any, bool) {
	get := func(i int) float64 { f, _ := asFloat(vals[i]); return f }

	switch op {
	// Unary
	case uop.OpNeg:
		return -get(0), true
	case uop.OpExp2:
		return math.Exp2(get(0)), true
	case uop.OpLog2:
		x := get(0)
		if x > 0 {
			return math.Log2(x), true
		}
		if x == 0 {
			return math.Inf(-1), true
		}
		return math.NaN(), true
	case uop.OpSqrt:
		x := get(0)
		if x >= 0 {
			return math.Sqrt(x), true
		}
		return math.NaN(), true
	case uop.OpReciprocal:
		x := get(0)
		if x != 0 {
			return 1.0 / x, true
		}
		return math.Copysign(math.Inf(1), x), true
	case uop.OpSin:
		x := get(0)
		if math.IsInf(x, 0) {
			return math.NaN(), true
		}
		return math.Sin(x), true
	case uop.OpTrunc:
		return math.Trunc(get(0)), true

	// Binary float
	case uop.OpAdd:
		return get(0) + get(1), true
	case uop.OpSub:
		return get(0) - get(1), true
	case uop.OpMul:
		return get(0) * get(1), true
	case uop.OpFDiv:
		return get(0) / get(1), true
	case uop.OpMax:
		return math.Max(get(0), get(1)), true
	case uop.OpCmpLt:
		return get(0) < get(1), true
	case uop.OpCmpNe:
		return get(0) != get(1), true
	case uop.OpCmpEq:
		return get(0) == get(1), true

	// Ternary
	case uop.OpWhere:
		cond, _ := asBool(vals[0])
		if cond {
			return get(1), true
		}
		return get(2), true
	case uop.OpMulAcc:
		return get(0)*get(1) + get(2), true
	}
	return nil, false
}

func execALUInt(op uop.Op, vals []any) (any, bool) {
	get := func(i int) int64 { n, _ := asInt(vals[i]); return n }
	getB := func(i int) bool { b, _ := asBool(vals[i]); return b }

	switch op {
	// Unary
	case uop.OpNeg:
		return -get(0), true
	case uop.OpTrunc:
		return get(0), true // integer Trunc is identity

	// Binary integer
	case uop.OpAdd:
		return get(0) + get(1), true
	case uop.OpSub:
		return get(0) - get(1), true
	case uop.OpMul:
		return get(0) * get(1), true
	case uop.OpIDiv:
		b := get(1)
		if b == 0 {
			return nil, false // undefined
		}
		return floorDiv(get(0), b), true
	case uop.OpMod:
		b := get(1)
		if b == 0 {
			return nil, false
		}
		return floorMod(get(0), b), true
	case uop.OpMax:
		a, b := get(0), get(1)
		if a > b {
			return a, true
		}
		return b, true
	case uop.OpShl:
		return get(0) << uint(get(1)), true
	case uop.OpShr:
		return get(0) >> uint(get(1)), true
	case uop.OpXor:
		return get(0) ^ get(1), true
	case uop.OpOr:
		return get(0) | get(1), true
	case uop.OpAnd:
		return get(0) & get(1), true
	case uop.OpCmpLt:
		return get(0) < get(1), true
	case uop.OpCmpNe:
		return get(0) != get(1), true
	case uop.OpCmpEq:
		return get(0) == get(1), true

	// Ternary
	case uop.OpWhere:
		if getB(0) {
			return get(1), true
		}
		return get(2), true
	case uop.OpMulAcc:
		return get(0)*get(1) + get(2), true
	}
	return nil, false
}

// foldConstALU folds an ALU node whose sources are all OpConst.
// Returns the replacement Const UOp, or (zero, false) if folding is not possible.
func foldConstALU(node uop.UOp) (uop.UOp, bool) {
	if node.NSrc() == 0 {
		return uop.UOp{}, false
	}
	vals := make([]any, node.NSrc())
	for i := 0; i < node.NSrc(); i++ {
		s := node.Src(i)
		if s.Op() != uop.OpConst {
			return uop.UOp{}, false
		}
		vals[i] = s.Arg()
	}

	raw, ok := execALU(node.Op(), node.DType(), vals)
	if !ok {
		return uop.UOp{}, false
	}

	dtype := node.DType()
	var result any
	switch {
	case dtype.IsBool():
		b, _ := asBool(raw)
		result = b
	case dtype.IsFloat():
		f, _ := asFloat(raw)
		result = truncateFloat(f, dtype)
	case dtype.IsInt():
		// Comparison ops produce bool but the result dtype is Bool, handled above.
		// For integer arithmetic, raw is int64.
		i, _ := asInt(raw)
		result = truncateInt(i, dtype)
	default:
		return uop.UOp{}, false
	}

	return node.Arena().New(uop.OpConst, dtype, nil, result, nil), true
}

// constLike returns a Const with the same dtype as u and value v.
func constLike(u uop.UOp, v any) uop.UOp {
	return u.Arena().New(uop.OpConst, u.DType(), nil, v, nil)
}
