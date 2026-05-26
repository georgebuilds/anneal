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

// RenderWGSL converts a kernel's SINK AST to a WGSL compute shader string.
// It runs the lowerer internally; callers only need to supply the ExecItem.
func RenderWGSL(item schedule.ExecItem) string {
	instrs := Lower(item)
	return renderInstrs(instrs, item)
}

func renderInstrs(instrs []Instr, item schedule.ExecItem) string {
	var b strings.Builder

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
			fmt.Fprintf(&b, "%sif (gid_x >= %du) { return; }\n", indent(), ins.TotalN)

		case InstrGIDVar:
			// let r_ID: i32 = i32((gid_x / Stride) % Size);
			var expr string
			if ins.Stride == 1 && len(instrs) > 0 {
				// last (innermost) dimension: no division needed
				expr = fmt.Sprintf("i32(gid_x %% %du)", ins.RangeSize)
			} else {
				expr = fmt.Sprintf("i32((gid_x / %du) %% %du)", ins.Stride, ins.RangeSize)
			}
			fmt.Fprintf(&b, "%slet r%d: i32 = %s;\n", indent(), ins.RangeID, expr)

		case InstrLoopBegin:
			fmt.Fprintf(&b, "%sfor (var r%d: i32 = 0; r%d < %d; r%d++) {\n",
				indent(), ins.RangeID, ins.RangeID, ins.RangeSize, ins.RangeID)
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
			if ins.TotalN <= 1 {
				idxExpr = "0u"
			} else {
				idxExpr = "gid_x"
			}
			fmt.Fprintf(&b, "%sdata0[%s] = %s;\n", indent(), idxExpr, ins.Expr)
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
	switch v := u.Arg().(type) {
	case float64:
		if math.IsInf(v, -1) {
			// Most-negative finite f32 — used as max-reduce identity.
			return "bitcast<f32>(0xff7fffffu)"
		}
		if math.IsInf(v, 1) {
			return "bitcast<f32>(0x7f7fffffu)"
		}
		if math.IsNaN(v) {
			return "bitcast<f32>(0x7fc00000u)"
		}
		s := strconv.FormatFloat(v, 'f', -1, 32)
		if !strings.Contains(s, ".") {
			s += ".0"
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
	switch op {
	case uop.OpAdd:
		if dtype.IsFloat() {
			return "0.0"
		}
		return "0"
	case uop.OpMul:
		if dtype.IsFloat() {
			return "1.0"
		}
		return "1"
	case uop.OpMax:
		if dtype.IsFloat() {
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
// Bool cannot be stored in WGSL storage buffers; we promote it to u32.
func wgslBufferElemType(d *uop.DType) string {
	if d == nil {
		return "f32"
	}
	// For pointer dtypes, use the base element type.
	if d.IsPtr() {
		d = d.Base()
	}
	t := wgslDType(d.Scalar())
	if t == "bool" {
		return "u32"
	}
	return t
}
