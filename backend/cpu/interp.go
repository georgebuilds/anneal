package cpu

import (
	"fmt"
	"math"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

// interpret evaluates one kernel ExecItem against host buffers.
//
// Kernel AST shape (per codegen/lower.go lowerSink):
//
//	Sink                    item.Ast
//	  End                   sink.Src[0]
//	    Store               end.Src[0]
//	      Index(Param(0))   store.Src[0]  - output address expression
//	      body              store.Src[1]  - kernel body, an expression tree
//	    ranges...           end.Src[1:]   - AxisLoop / AxisLocal / AxisWorkgroup
//
// For slice 1 the CPU backend executes static kernels only; every range
// must have a concrete OpConst bound. The interpreter materializes the
// loop nest in row-major (innermost = last) order, evaluates the body
// expression for each combination, and writes the result to the output
// buffer at the flat address computed from Store.Src[0].
//
// Reductions live inside the body as OpReduce nodes; their inner ranges
// are evaluated as nested loops with an accumulator that starts at the
// reduction identity (0 for Add, -Inf for Max).
//
// Supported op set (slice 1 MLP coverage):
//   - Loads / stores: Index(Param, ...), Store
//   - ALU (binary):   Add, Sub, Mul, FDiv, Max, Min, CmpLt, CmpNe, CmpEq
//   - ALU (unary):    Neg, Reciprocal, Sqrt, Exp2, Log2, Cast
//   - ALU (ternary):  Where, MulAcc
//   - Reductions:     Reduce(Add), Reduce(Max), Reduce(Min)
//   - Integer index:  Add, Sub, Mul, IDiv, Mod, Neg on Int32
//   - Const, Range
//
// Ops outside this set produce a clear "not yet implemented" error so the
// caller knows exactly what's missing.
func interpret(item schedule.ExecItem, bufs map[uint32]*Buffer) error {
	sink := item.Ast
	if !sink.Valid() || sink.Op() != uop.OpSink {
		return fmt.Errorf("interp: kernel AST is not a SINK (op=%s)", sink.Op())
	}
	end := sink.Src(0)
	if end.Op() != uop.OpEnd {
		return fmt.Errorf("interp: sink.Src[0] is not END (op=%s)", end.Op())
	}
	store := end.Src(0)
	if store.Op() != uop.OpStore {
		return fmt.Errorf("interp: end.Src[0] is not STORE (op=%s)", store.Op())
	}
	outIndex := store.Src(0)
	body := store.Src(1)

	// Collect the outer iteration ranges from End.Src[1:]. The CPU backend
	// has no notion of workgroup/local/dispatch, so all AxisLoop /
	// AxisWorkgroup / AxisLocal ranges are treated as plain Go for-loops.
	var ranges []uop.UOp
	for i := 1; i < end.NSrc(); i++ {
		r := end.Src(i)
		if r.Op() == uop.OpConst {
			// Const "ranges" appear for size-1 dims; treat as a single iter.
			ranges = append(ranges, r)
			continue
		}
		if r.Op() != uop.OpRange {
			return fmt.Errorf("interp: end.Src[%d] is %s; expected OpRange or OpConst", i, r.Op())
		}
		ra, ok := r.Arg().(uop.RangeArg)
		if !ok {
			return fmt.Errorf("interp: range arg type %T", r.Arg())
		}
		switch ra.Type {
		case uop.AxisLoop, uop.AxisWorkgroup, uop.AxisLocal:
			if uop.RangeIsSymbolic(r) {
				return fmt.Errorf("interp: symbolic outer range (RangeID=%d) not supported in slice 1", ra.ID)
			}
			ranges = append(ranges, r)
		case uop.AxisUpcast, uop.AxisVectorize:
			// BEAM optimizer hints; flatten into plain loops.
			if uop.RangeIsSymbolic(r) {
				return fmt.Errorf("interp: symbolic upcast/vectorize range (RangeID=%d) not supported in slice 1", ra.ID)
			}
			ranges = append(ranges, r)
		case uop.AxisReduce:
			// Reduce loops live inside OpReduce.Src[1:]; the rangeified
			// kernel also lists them in End.Src[1:] for bookkeeping. Skip
			// them at the outer level — OpReduce handles them.
			continue
		default:
			return fmt.Errorf("interp: unknown axis type %d", ra.Type)
		}
	}

	// Resolve the output buffer up front: Store -> Index(Param(0), ...).
	if outIndex.Op() != uop.OpIndex {
		return fmt.Errorf("interp: store.Src[0] is %s; expected OpIndex", outIndex.Op())
	}
	outParam := outIndex.Src(0)
	if outParam.Op() != uop.OpParam {
		return fmt.Errorf("interp: output index base is %s; expected OpParam", outParam.Op())
	}
	outIdx, ok := outParam.Arg().(int64)
	if !ok {
		return fmt.Errorf("interp: output param arg type %T", outParam.Arg())
	}
	if outIdx != 0 {
		return fmt.Errorf("interp: output param idx is %d; expected 0", outIdx)
	}
	if len(item.Bufs) == 0 {
		return fmt.Errorf("interp: kernel has no buffers")
	}
	outDesc := item.Bufs[0]
	outBuf := bufs[outDesc.UOpIdx]
	if outBuf == nil {
		return fmt.Errorf("interp: missing output buffer for UOpIdx=%d", outDesc.UOpIdx)
	}

	// Per-kernel evaluator state: caches Range loop-var values and Buffer
	// shapes (computed lazily on first index access).
	st := &state{
		item:        item,
		bufs:        bufs,
		rangeVal:    make(map[int]int64),
		paramShapes: make(map[int][]int64),
	}

	// Iterate the outer loop nest in lexicographic (outer→inner) order.
	rangeSizes := make([]int64, len(ranges))
	rangeIDs := make([]int, len(ranges))
	hasID := make([]bool, len(ranges))
	for i, r := range ranges {
		if r.Op() == uop.OpConst {
			rangeSizes[i] = 1
			continue
		}
		rangeSizes[i] = uop.RangeSize(r)
		rangeIDs[i] = r.Arg().(uop.RangeArg).ID
		hasID[i] = true
	}

	// Walk the nest with an explicit counter vector to avoid Go-stack growth
	// on deep nests (8+ dims for conv-style kernels are possible).
	counters := make([]int64, len(ranges))
	for {
		// Bind every range to its current counter value.
		for i, r := range ranges {
			if r.Op() == uop.OpConst {
				continue
			}
			_ = r
			st.rangeVal[rangeIDs[i]] = counters[i]
		}

		// Evaluate the output flat address and body expression.
		flatOut, ferr := st.evalIntIndex(outIndex)
		if ferr != nil {
			return ferr
		}
		val, verr := st.evalFloat(body)
		if verr != nil {
			return verr
		}

		dst := outBuf.asF32()
		if dst == nil {
			return fmt.Errorf("interp: output buffer is not f32-typed (dt=%s)", outBuf.DType())
		}
		if flatOut < 0 || flatOut >= int64(len(dst)) {
			return fmt.Errorf("interp: output flat index %d out of range [0,%d)", flatOut, len(dst))
		}
		dst[flatOut] = val

		// Increment counters (innermost = last index, wraps to outer).
		if !incCounters(counters, rangeSizes) {
			break
		}
	}
	return nil
}

// incCounters advances counters as a mixed-radix integer with the innermost
// digit at the highest index. Returns false when the iteration is exhausted.
func incCounters(counters, sizes []int64) bool {
	for i := len(counters) - 1; i >= 0; i-- {
		counters[i]++
		if counters[i] < sizes[i] {
			return true
		}
		counters[i] = 0
	}
	return false
}

// state is per-kernel interpreter state.
type state struct {
	item        schedule.ExecItem
	bufs        map[uint32]*Buffer
	rangeVal    map[int]int64
	paramShapes map[int][]int64
}

// evalFloat evaluates u as a scalar float32. Handles Index loads, ALU,
// constants, ranges (rare — usually wrapped in Cast), and OpReduce.
func (st *state) evalFloat(u uop.UOp) (float32, error) {
	switch u.Op() {
	case uop.OpConst:
		switch v := u.Arg().(type) {
		case float64:
			return float32(v), nil
		case int64:
			return float32(v), nil
		case bool:
			if v {
				return 1, nil
			}
			return 0, nil
		default:
			return 0, fmt.Errorf("interp: OpConst arg type %T", u.Arg())
		}
	case uop.OpRange:
		ra := u.Arg().(uop.RangeArg)
		v, ok := st.rangeVal[ra.ID]
		if !ok {
			return 0, fmt.Errorf("interp: range %d not bound", ra.ID)
		}
		return float32(v), nil
	case uop.OpIndex:
		return st.evalIndexLoadFloat(u)
	case uop.OpCast:
		// Slice 1 only handles f32-equivalent casts (f32↔i32). The actual
		// cast is a no-op at the float layer since both code paths round-trip
		// through float64 internally; we just dispatch on the inner type.
		inner := u.Src(0)
		if inner.DType() != nil && inner.DType().IsInt() {
			iv, err := st.evalInt(inner)
			if err != nil {
				return 0, err
			}
			return float32(iv), nil
		}
		return st.evalFloat(inner)
	case uop.OpAdd:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a + b, nil
	case uop.OpSub:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a - b, nil
	case uop.OpMul:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a * b, nil
	case uop.OpFDiv:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a / b, nil
	case uop.OpNeg:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		return -a, nil
	case uop.OpReciprocal:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		return 1.0 / a, nil
	case uop.OpSqrt:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		return float32(math.Sqrt(float64(a))), nil
	case uop.OpExp2:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		return float32(math.Exp2(float64(a))), nil
	case uop.OpLog2:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		return float32(math.Log2(float64(a))), nil
	case uop.OpMax:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a > b {
			return a, nil
		}
		return b, nil
	case uop.OpMin:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a < b {
			return a, nil
		}
		return b, nil
	case uop.OpCmpLt:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a < b {
			return 1, nil
		}
		return 0, nil
	case uop.OpCmpNe:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a != b {
			return 1, nil
		}
		return 0, nil
	case uop.OpCmpEq:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a == b {
			return 1, nil
		}
		return 0, nil
	case uop.OpWhere:
		cond, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		if cond != 0 {
			return st.evalFloat(u.Src(1))
		}
		return st.evalFloat(u.Src(2))
	case uop.OpMulAcc:
		a, err := st.evalFloat(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalFloat(u.Src(1))
		if err != nil {
			return 0, err
		}
		c, err := st.evalFloat(u.Src(2))
		if err != nil {
			return 0, err
		}
		return a*b + c, nil
	case uop.OpReduce:
		return st.evalReduce(u)
	default:
		return 0, fmt.Errorf("cpu: op %s not yet implemented", u.Op())
	}
}

// evalReduce evaluates an OpReduce node: accumulate the inner expression
// over all combinations of the reduce ranges in Src[1:].
//
// Reduction accumulator dtype: per SPEC §10, reductions widen to f32 even
// when operands are f16. We always accumulate in float64 (no precision
// regression for the static f32 MLP) and narrow on return.
func (st *state) evalReduce(u uop.UOp) (float32, error) {
	accOp, ok := u.Arg().(uop.Op)
	if !ok {
		return 0, fmt.Errorf("interp: OpReduce arg type %T", u.Arg())
	}
	body := u.Src(0)
	redRanges := make([]uop.UOp, u.NSrc()-1)
	for i := 1; i < u.NSrc(); i++ {
		redRanges[i-1] = u.Src(i)
	}
	sizes := make([]int64, len(redRanges))
	ids := make([]int, len(redRanges))
	hasID := make([]bool, len(redRanges))
	for i, r := range redRanges {
		if r.Op() == uop.OpConst {
			sizes[i] = 1
			continue
		}
		if r.Op() != uop.OpRange {
			return 0, fmt.Errorf("interp: reduce src[%d] is %s; expected OpRange", i+1, r.Op())
		}
		if uop.RangeIsSymbolic(r) {
			return 0, fmt.Errorf("interp: symbolic reduce range not supported in slice 1")
		}
		sizes[i] = uop.RangeSize(r)
		ids[i] = r.Arg().(uop.RangeArg).ID
		hasID[i] = true
	}

	var acc float64
	switch accOp {
	case uop.OpAdd:
		acc = 0
	case uop.OpMul:
		acc = 1
	case uop.OpMax:
		acc = math.Inf(-1)
	case uop.OpMin:
		acc = math.Inf(1)
	default:
		return 0, fmt.Errorf("cpu: reduce op %s not yet implemented", accOp)
	}

	// Save and restore the outer range bindings for any reduce-range IDs
	// (they shouldn't collide, but be defensive).
	saved := make(map[int]int64, len(ids))
	for i, r := range redRanges {
		if r.Op() == uop.OpConst || !hasID[i] {
			continue
		}
		if v, ok := st.rangeVal[ids[i]]; ok {
			saved[ids[i]] = v
		}
	}
	defer func() {
		for i, r := range redRanges {
			if r.Op() == uop.OpConst || !hasID[i] {
				continue
			}
			if v, ok := saved[ids[i]]; ok {
				st.rangeVal[ids[i]] = v
			} else {
				delete(st.rangeVal, ids[i])
			}
		}
	}()

	counters := make([]int64, len(redRanges))
	for {
		for i, r := range redRanges {
			if r.Op() == uop.OpConst {
				continue
			}
			st.rangeVal[ids[i]] = counters[i]
		}
		v, err := st.evalFloat(body)
		if err != nil {
			return 0, err
		}
		fv := float64(v)
		switch accOp {
		case uop.OpAdd:
			acc += fv
		case uop.OpMul:
			acc *= fv
		case uop.OpMax:
			if fv > acc {
				acc = fv
			}
		case uop.OpMin:
			if fv < acc {
				acc = fv
			}
		}
		if !incCounters(counters, sizes) {
			break
		}
	}
	return float32(acc), nil
}

// evalIndexLoadFloat reads from a buffer at an n-dim Index expression and
// returns the f32 value (casting from i32 if the buffer is int-typed).
func (st *state) evalIndexLoadFloat(u uop.UOp) (float32, error) {
	flat, err := st.evalIntIndex(u)
	if err != nil {
		return 0, err
	}
	param := u.Src(0)
	if param.Op() != uop.OpParam {
		return 0, fmt.Errorf("interp: Index.Src[0]=%s; expected OpParam", param.Op())
	}
	paramIdx := int(param.Arg().(int64))
	if paramIdx < 0 || paramIdx >= len(st.item.Bufs) {
		return 0, fmt.Errorf("interp: param idx %d out of range [0,%d)", paramIdx, len(st.item.Bufs))
	}
	desc := st.item.Bufs[paramIdx]
	buf := st.bufs[desc.UOpIdx]
	if buf == nil {
		return 0, fmt.Errorf("interp: missing buffer for UOpIdx=%d", desc.UOpIdx)
	}
	if f := buf.asF32(); f != nil {
		if flat < 0 || flat >= int64(len(f)) {
			return 0, fmt.Errorf("interp: f32 load flat=%d out of range [0,%d)", flat, len(f))
		}
		return f[flat], nil
	}
	if iSlice := buf.asI32(); iSlice != nil {
		if flat < 0 || flat >= int64(len(iSlice)) {
			return 0, fmt.Errorf("interp: i32 load flat=%d out of range [0,%d)", flat, len(iSlice))
		}
		return float32(iSlice[flat]), nil
	}
	return 0, fmt.Errorf("interp: buffer for UOpIdx=%d has no storage", desc.UOpIdx)
}

// paramShape returns the (cached) shape of a buffer, as int64 dims with
// symbolic positions left as 0 (matches schedule.Buffer.Shape).
func (st *state) paramShape(paramIdx int) []int64 {
	if sh, ok := st.paramShapes[paramIdx]; ok {
		return sh
	}
	desc := st.item.Bufs[paramIdx]
	sh := append([]int64(nil), desc.Shape...)
	st.paramShapes[paramIdx] = sh
	return sh
}

// evalIntIndex evaluates an OpIndex node to a flat element offset using the
// schedule.Buffer.Shape for the underlying buffer. Mirrors codegen's
// emitIndex stride-from-the-right convention.
func (st *state) evalIntIndex(u uop.UOp) (int64, error) {
	if u.Op() != uop.OpIndex {
		return 0, fmt.Errorf("interp: evalIntIndex on non-Index op %s", u.Op())
	}
	param := u.Src(0)
	if param.Op() != uop.OpParam {
		return 0, fmt.Errorf("interp: Index base is %s; expected OpParam (Slice 1 has no DefineLocal yet)", param.Op())
	}
	paramIdx := int(param.Arg().(int64))
	nDims := u.NSrc() - 1
	if nDims == 0 {
		return 0, nil
	}
	if nDims == 1 {
		return st.evalInt(u.Src(1))
	}
	shape := st.paramShape(paramIdx)
	if len(shape) < nDims {
		return 0, fmt.Errorf("interp: buffer shape len=%d < nDims=%d for paramIdx=%d", len(shape), nDims, paramIdx)
	}
	// stride[d] = product of shape[d+1..nDims-1].
	strides := make([]int64, nDims)
	strides[nDims-1] = 1
	for i := nDims - 2; i >= 0; i-- {
		next := shape[i+1]
		if next == 0 {
			return 0, fmt.Errorf("interp: symbolic dim in buffer %d shape; slice 1 is static-only", paramIdx)
		}
		strides[i] = strides[i+1] * next
	}
	var flat int64
	for d := 0; d < nDims; d++ {
		dv, err := st.evalInt(u.Src(d + 1))
		if err != nil {
			return 0, err
		}
		flat += dv * strides[d]
	}
	return flat, nil
}

// evalInt evaluates u as an integer (index arithmetic). Covers the subset
// emitted by rangeify for static buffer index expressions: ranges, consts,
// Add/Sub/Mul/IDiv/Mod/Neg.
func (st *state) evalInt(u uop.UOp) (int64, error) {
	switch u.Op() {
	case uop.OpConst:
		switch v := u.Arg().(type) {
		case int64:
			return v, nil
		case float64:
			return int64(v), nil
		case bool:
			if v {
				return 1, nil
			}
			return 0, nil
		default:
			return 0, fmt.Errorf("interp: int OpConst arg type %T", u.Arg())
		}
	case uop.OpRange:
		ra := u.Arg().(uop.RangeArg)
		v, ok := st.rangeVal[ra.ID]
		if !ok {
			return 0, fmt.Errorf("interp: range %d not bound (int)", ra.ID)
		}
		return v, nil
	case uop.OpCast:
		// Index dtype upcasts (i32↔i64) are no-ops at the host level.
		return st.evalInt(u.Src(0))
	case uop.OpAdd:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a + b, nil
	case uop.OpSub:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a - b, nil
	case uop.OpMul:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		return a * b, nil
	case uop.OpIDiv:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		if b == 0 {
			return 0, fmt.Errorf("interp: IDiv by zero")
		}
		// Floor division to match WGSL i32 semantics for non-negative operands
		// (rangeify-produced indices are non-negative by construction).
		return a / b, nil
	case uop.OpMod:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		if b == 0 {
			return 0, fmt.Errorf("interp: Mod by zero")
		}
		return a % b, nil
	case uop.OpNeg:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		return -a, nil
	case uop.OpMax:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a > b {
			return a, nil
		}
		return b, nil
	case uop.OpMin:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a < b {
			return a, nil
		}
		return b, nil
	case uop.OpCmpLt:
		a, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		b, err := st.evalInt(u.Src(1))
		if err != nil {
			return 0, err
		}
		if a < b {
			return 1, nil
		}
		return 0, nil
	case uop.OpWhere:
		c, err := st.evalInt(u.Src(0))
		if err != nil {
			return 0, err
		}
		if c != 0 {
			return st.evalInt(u.Src(1))
		}
		return st.evalInt(u.Src(2))
	case uop.OpIndex:
		// Gather idx loads are int-typed indirect loads; rare in MLP but
		// supported for completeness with the float-side path.
		flat, err := st.evalIntIndex(u)
		if err != nil {
			return 0, err
		}
		param := u.Src(0)
		paramIdx := int(param.Arg().(int64))
		desc := st.item.Bufs[paramIdx]
		buf := st.bufs[desc.UOpIdx]
		if i := buf.asI32(); i != nil {
			if flat < 0 || flat >= int64(len(i)) {
				return 0, fmt.Errorf("interp: i32 idx load flat=%d out of range", flat)
			}
			return int64(i[flat]), nil
		}
		return 0, fmt.Errorf("interp: int Index load on non-i32 buffer")
	default:
		return 0, fmt.Errorf("cpu: int op %s not yet implemented", u.Op())
	}
}
