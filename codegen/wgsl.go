package codegen

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

const workgroupSize = 64 // default 1-D workgroup size; tunable in Phase 8b

func init() {
	// Wire the WGSL renderer into the schedule cache so cacheStore can pre-render
	// and zero Ast, releasing the arena reference before storing.
	schedule.WGSLRenderFunc = RenderWGSL
}

// kernelUsesF16 reports whether any buffer in the kernel has an f16 element type.
func kernelUsesF16(item schedule.ExecItem) bool {
	for _, buf := range item.Bufs {
		if buf.DType != nil && buf.DType.Scalar() == uop.Dtypes.Float16 {
			return true
		}
	}
	return false
}

// CompileWGSL converts a kernel's SINK AST to WGSL.
func CompileWGSL(item schedule.ExecItem) (string, error) {
	instrs := Lower(item)
	return renderInstrs(instrs, item), nil
}

// RenderWGSL converts a kernel's SINK AST to a WGSL compute shader string.
// It is wired as schedule.WGSLRenderFunc and must not panic — the cache
// callback runs inside CreateSchedule where panics are uncaught.
func RenderWGSL(item schedule.ExecItem) string {
	instrs := Lower(item)
	return renderInstrs(instrs, item)
}

func renderInstrs(instrs []Instr, item schedule.ExecItem) string {
	var b strings.Builder

	// WGSL f16 extension — must be the first directive in the shader, before any
	// global declarations. Emitted only when the kernel actually uses f16 buffers.
	// The shader-f16 device feature must also be enabled at device-open time (slice 2).
	if kernelUsesF16(item) {
		b.WriteString("enable f16;\n\n")
	}

	// Detect whether any range is symbolic; if so we emit a trailing params_n
	// storage buffer that carries runtime dim values.
	hasSymDim := false
	for _, ins := range instrs {
		if ins.Symbolic && (ins.Kind == InstrBoundsCheck || ins.Kind == InstrGIDVar || ins.Kind == InstrLoopBegin) {
			hasSymDim = true
			break
		}
	}

	// ── storage buffer bindings ────────────────────────────────────────────
	ki := item.Ast.Arg().(uop.KernelInfo)
	for i := 0; i < ki.NumParams; i++ {
		access := "read"
		if i == 0 {
			access = "read_write"
		}
		elemType := "f32"
		if i < len(item.Bufs) && item.Bufs[i].DType != nil {
			elemType = wgslBufferElemType(item.Bufs[i].DType)
		}
		fmt.Fprintf(&b, "@group(0) @binding(%d) var<storage, %s> data%d: array<%s>;\n",
			i, access, i, elemType)
	}
	// Symbolic-shapes params uniform: holds runtime dim values as u32 words.
	// Uses a uniform buffer (not storage) to stay under Metal's 8-storage-buffer-
	// per-stage limit when backward-pass kernels already saturate the data slots.
	// Individual u32 fields (not array<u32,N>) because WGSL uniform-address-space
	// arrays have element stride = max(SizeOf, 16) = 16, which would make the
	// binding 64 bytes; individual fields have stride 4, keeping it at 16 bytes.
	// Binding slot = ki.NumParams (immediately after all data bindings).
	if hasSymDim {
		fmt.Fprintf(&b, "struct ParamsN { n0: u32, n1: u32, n2: u32, n3: u32 };\n")
		fmt.Fprintf(&b, "@group(0) @binding(%d) var<uniform> params_n: ParamsN;\n",
			ki.NumParams)
	}

	b.WriteString("\n")

	// ── compute entry point ────────────────────────────────────────────────
	fmt.Fprintf(&b, "@compute @workgroup_size(%d)\n", workgroupSize)
	b.WriteString("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n")
	b.WriteString("  let gid_x = gid.x;\n")

	depth := 1 // current indent depth (starts at 1 for inside fn)

	indent := func() string { return strings.Repeat("  ", depth) }

	for _, ins := range instrs {
		switch ins.Kind {
		case InstrBoundsCheck:
			if ins.Symbolic {
				// Bound is unknown at compile time; read from the params_n buffer.
				// For multi-dim symbolic ([batch, features]), multiply by the
				// concrete trailing dimension count so the total equals batch*features.
				if ins.ConcreteTrailing > 1 {
					fmt.Fprintf(&b, "%sif (gid_x >= params_n.n0 * %du) { return; }\n", indent(), ins.ConcreteTrailing)
				} else {
					fmt.Fprintf(&b, "%sif (gid_x >= params_n.n0) { return; }\n", indent())
				}
			} else {
				fmt.Fprintf(&b, "%sif (gid_x >= %du) { return; }\n", indent(), ins.TotalN)
			}

		case InstrGIDVar:
			// let r_ID: i32 = i32((gid_x / Stride) % Size);
			var expr string
			if ins.Symbolic {
				if ins.Stride == 1 {
					// Single-dim symbolic: gid_x IS the loop variable (bounds-checked above).
					// Avoids a runtime division by the symbolic bound.
					expr = "i32(gid_x)"
				} else {
					// Multi-dim with one symbolic axis: static strides still apply.
					expr = fmt.Sprintf("i32(gid_x / %du)", ins.Stride)
				}
			} else if ins.Stride == 1 && len(instrs) > 0 {
				// last (innermost) dimension: no division needed
				expr = fmt.Sprintf("i32(gid_x %% %du)", ins.RangeSize)
			} else {
				expr = fmt.Sprintf("i32((gid_x / %du) %% %du)", ins.Stride, ins.RangeSize)
			}
			fmt.Fprintf(&b, "%slet r%d: i32 = %s;\n", indent(), ins.RangeID, expr)

		case InstrLoopBegin:
			if ins.Symbolic {
				symField := [...]string{"n0", "n1", "n2", "n3"}[ins.SymParamIdx]
				fmt.Fprintf(&b, "%sfor (var r%d: i32 = 0; r%d < i32(params_n.%s); r%d++) {\n",
					indent(), ins.RangeID, ins.RangeID, symField, ins.RangeID)
			} else {
				fmt.Fprintf(&b, "%sfor (var r%d: i32 = 0; r%d < %d; r%d++) {\n",
					indent(), ins.RangeID, ins.RangeID, ins.RangeSize, ins.RangeID)
			}
			depth++

		case InstrLoopEnd:
			depth--
			fmt.Fprintf(&b, "%s}\n", indent())

		case InstrAccInit:
			fmt.Fprintf(&b, "%svar acc%d: %s = %s;\n",
				indent(), ins.AccIdx, ins.WGSLType, ins.Identity)

		case InstrAccUpdate:
			upd := accUpdateExpr(ins.AccOp, fmt.Sprintf("acc%d", ins.AccIdx), ins.Expr)
			fmt.Fprintf(&b, "%sacc%d = %s;\n", indent(), ins.AccIdx, upd)

		case InstrLet:
			wt := wgslDType(ins.DType)
			fmt.Fprintf(&b, "%slet t%d: %s = %s;\n", indent(), ins.NodeIdx, wt, ins.Expr)

		case InstrStore:
			var idxExpr string
			if !ins.Symbolic && ins.TotalN <= 1 {
				idxExpr = "0u"
			} else {
				idxExpr = "gid_x"
			}
			if ins.DType != nil && ins.DType.Scalar() == uop.Dtypes.BFloat16 {
				// bf16 storage: clear the low 16 mantissa bits to produce a bf16-precision
				// value stored in a u32 slot. bitcast<f32> at load time recovers the value.
				fmt.Fprintf(&b, "%sdata0[%s] = bitcast<u32>(%s) & 0xFFFF0000u;\n", indent(), idxExpr, ins.Expr)
			} else {
				fmt.Fprintf(&b, "%sdata0[%s] = %s;\n", indent(), idxExpr, ins.Expr)
			}
		}
	}

	b.WriteString("}\n")
	return b.String()
}

// accUpdateExpr builds the WGSL RHS for an accumulator update.
func accUpdateExpr(op uop.Op, accName, elemExpr string) string {
	switch op {
	case uop.OpAdd:
		return fmt.Sprintf("%s + %s", accName, elemExpr)
	case uop.OpMul:
		return fmt.Sprintf("%s * %s", accName, elemExpr)
	case uop.OpMax:
		return fmt.Sprintf("max(%s, %s)", accName, elemExpr)
	default:
		// Fallback: additive
		return fmt.Sprintf("%s + %s", accName, elemExpr)
	}
}

// aluExpr builds the WGSL expression string for one ALU node.
func aluExpr(op uop.Op, srcs []string, dtype *uop.DType) string {
	switch op {
	// Unary
	case uop.OpExp2:
		return fmt.Sprintf("exp2(%s)", srcs[0])
	case uop.OpLog2:
		return fmt.Sprintf("log2(%s)", srcs[0])
	case uop.OpSin:
		return fmt.Sprintf("sin(%s)", srcs[0])
	case uop.OpSqrt:
		return fmt.Sprintf("sqrt(%s)", srcs[0])
	case uop.OpReciprocal:
		return fmt.Sprintf("(1.0 / %s)", srcs[0])
	case uop.OpNeg:
		return fmt.Sprintf("(-%s)", srcs[0])
	case uop.OpTrunc:
		return fmt.Sprintf("trunc(%s)", srcs[0])
	// Binary
	case uop.OpAdd:
		return fmt.Sprintf("(%s + %s)", srcs[0], srcs[1])
	case uop.OpSub:
		return fmt.Sprintf("(%s - %s)", srcs[0], srcs[1])
	case uop.OpMul:
		return fmt.Sprintf("(%s * %s)", srcs[0], srcs[1])
	case uop.OpFDiv:
		return fmt.Sprintf("(%s / %s)", srcs[0], srcs[1])
	case uop.OpIDiv:
		// WGSL integer division truncates toward zero; index values are non-negative
		// so this matches Go's integer division semantics for valid indices.
		return fmt.Sprintf("(%s / %s)", srcs[0], srcs[1])
	case uop.OpMod:
		return fmt.Sprintf("(%s %% %s)", srcs[0], srcs[1])
	case uop.OpMax:
		return fmt.Sprintf("max(%s, %s)", srcs[0], srcs[1])
	case uop.OpShl:
		return fmt.Sprintf("(%s << u32(%s))", srcs[0], srcs[1])
	case uop.OpShr:
		return fmt.Sprintf("(%s >> u32(%s))", srcs[0], srcs[1])
	case uop.OpAnd:
		return fmt.Sprintf("(%s & %s)", srcs[0], srcs[1])
	case uop.OpOr:
		return fmt.Sprintf("(%s | %s)", srcs[0], srcs[1])
	case uop.OpXor:
		return fmt.Sprintf("(%s ^ %s)", srcs[0], srcs[1])
	case uop.OpCmpLt:
		return fmt.Sprintf("(%s < %s)", srcs[0], srcs[1])
	case uop.OpCmpNe:
		return fmt.Sprintf("(%s != %s)", srcs[0], srcs[1])
	case uop.OpCmpEq:
		return fmt.Sprintf("(%s == %s)", srcs[0], srcs[1])
	case uop.OpPow:
		return fmt.Sprintf("pow(%s, %s)", srcs[0], srcs[1])
	// Ternary
	case uop.OpWhere:
		// WHERE(cond, true_val, false_val) → WGSL select(false_val, true_val, cond)
		return fmt.Sprintf("select(%s, %s, %s)", srcs[2], srcs[1], srcs[0])
	case uop.OpMulAcc:
		return fmt.Sprintf("(%s + (%s * %s))", srcs[0], srcs[1], srcs[2])
	// Cast
	case uop.OpCast:
		return fmt.Sprintf("%s(%s)", wgslDType(dtype), srcs[0])
	case uop.OpBitcast:
		return fmt.Sprintf("bitcast<%s>(%s)", wgslDType(dtype), srcs[0])
	default:
		panic(fmt.Sprintf("codegen: unhandled ALU op %s", op))
	}
}

// constLiteral renders a CONST UOp as a WGSL literal.
func constLiteral(u uop.UOp) string {
	dtype := u.DType()
	isF16 := dtype != nil && dtype.Scalar() == uop.Dtypes.Float16
	switch v := u.Arg().(type) {
	case float64:
		if math.IsInf(v, -1) {
			if isF16 {
				// Most-negative finite f16 (0xFBFF = -65504).
				return "bitcast<f16>(0xFBFFu)"
			}
			// Most-negative finite f32 — used as max-reduce identity.
			return "bitcast<f32>(0xff7fffffu)"
		}
		if math.IsInf(v, 1) {
			if isF16 {
				// Most-positive finite f16 (0x7BFF = +65504).
				return "bitcast<f16>(0x7BFFu)"
			}
			return "bitcast<f32>(0x7f7fffffu)"
		}
		if math.IsNaN(v) {
			if isF16 {
				return "bitcast<f16>(0x7E00u)" // f16 qNaN
			}
			return "bitcast<f32>(0x7fc00000u)"
		}
		s := strconv.FormatFloat(v, 'f', -1, 32)
		if !strings.Contains(s, ".") {
			s += ".0"
		}
		if isF16 {
			return fmt.Sprintf("f16(%s)", s)
		}
		return s
	case int64:
		_ = dtype
		return strconv.FormatInt(v, 10)
	case bool:
		if v {
			return "true"
		}
		return "false"
	default:
		return "0"
	}
}

// reduceIdentity returns the WGSL literal for the identity element of a reduce op.
func reduceIdentity(op uop.Op, dtype *uop.DType) string {
	isF16 := dtype != nil && dtype.Scalar() == uop.Dtypes.Float16
	switch op {
	case uop.OpAdd:
		if dtype.IsFloat() {
			if isF16 {
				return "f16(0.0)"
			}
			return "0.0"
		}
		return "0"
	case uop.OpMul:
		if dtype.IsFloat() {
			if isF16 {
				return "f16(1.0)"
			}
			return "1.0"
		}
		return "1"
	case uop.OpMax:
		if dtype.IsFloat() {
			if isF16 {
				// bitcast of 0xFBFF = most negative finite f16 (-65504).
				return "bitcast<f16>(0xFBFFu)"
			}
			// bitcast of 0xFF7FFFFF = most negative finite f32 (~-3.4e38).
			// This is the correct lower bound; any real computation value dominates it.
			return "bitcast<f32>(0xff7fffffu)"
		}
		if dtype.IsUnsigned() {
			return "0u"
		}
		return "-2147483648" // INT32_MIN
	default:
		if dtype.IsFloat() {
			if isF16 {
				return "f16(0.0)"
			}
			return "0.0"
		}
		return "0"
	}
}

// wgslDType maps an anneal DType to its WGSL scalar type name.
// Index dtype maps to i32 (WGSL has no platform-pointer-sized integer in v1).
// Bool maps to bool (valid in expressions; storage buffers use wgslBufferElemType).
func wgslDType(d *uop.DType) string {
	if d == nil || d == uop.Dtypes.Void {
		return "f32"
	}
	s := d.Scalar()
	switch s {
	case uop.Dtypes.Float32:
		return "f32"
	case uop.Dtypes.Float16:
		return "f16" // requires "enable f16;" — noted but not emitted in v1
	case uop.Dtypes.Int32:
		return "i32"
	case uop.Dtypes.UInt32:
		return "u32"
	case uop.Dtypes.Index:
		return "i32"
	case uop.Dtypes.Bool:
		return "bool"
	case uop.Dtypes.Int8, uop.Dtypes.Int16:
		return "i32" // promoted
	case uop.Dtypes.UInt8, uop.Dtypes.UInt16:
		return "u32" // promoted
	case uop.Dtypes.Int64:
		return "i32" // WGSL has no i64; truncate (only affects non-standard workloads)
	case uop.Dtypes.UInt64:
		return "u32"
	default:
		if d.IsFloat() {
			return "f32"
		}
		return "i32"
	}
}

// wgslBufferElemType returns the WGSL element type for a storage buffer declaration.
// bf16 is stored as u32 (bf16 bits in the high 16, low 16 zeroed); load/store use
// bitcast<f32>/bitcast<u32> so all arithmetic runs in f32.
// Bool cannot be stored in WGSL storage buffers; we promote it to u32.
func wgslBufferElemType(d *uop.DType) string {
	if d == nil {
		return "f32"
	}
	if d.IsPtr() {
		d = d.Base()
	}
	if d.Scalar() == uop.Dtypes.BFloat16 {
		return "u32"
	}
	t := wgslDType(d.Scalar())
	if t == "bool" {
		return "u32"
	}
	return t
}
