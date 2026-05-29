package codegen

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
)

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
func CompileWGSL(item schedule.ExecItem) (schedule.RenderResult, error) {
	instrs, ws, wc := Lower(item)
	return schedule.RenderResult{WGSL: renderInstrs(instrs, item, ws, wc), LocalSize: ws, WorkgroupCount: wc}, nil
}

// RenderWGSL converts a kernel's SINK AST to a WGSL compute shader string.
func RenderWGSL(item schedule.ExecItem) schedule.RenderResult {
	instrs, ws, wc := Lower(item)
	return schedule.RenderResult{WGSL: renderInstrs(instrs, item, ws, wc), LocalSize: ws, WorkgroupCount: wc}
}

func renderInstrs(instrs []Instr, item schedule.ExecItem, ws [3]int, wc [3]int) string {
	var b strings.Builder

	// WGSL f16 extension
	if kernelUsesF16(item) {
		b.WriteString("enable f16;\n\n")
	}

	// Detect whether any range is symbolic
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
	if hasSymDim {
		fmt.Fprintf(&b, "struct ParamsN { n0: u32, n1: u32, n2: u32, n3: u32 };\n")
		fmt.Fprintf(&b, "@group(0) @binding(%d) var<uniform> params_n: ParamsN;\n",
			ki.NumParams)
	}

	b.WriteString("\n")

	// Global flat index for symbolic/large grids.
	// WC and WS components are used to compute the stride of each dimension.
	// For B1, we spread components into Y and Z if X overflows.
	fmt.Fprintf(&b, "@compute @workgroup_size(%d, %d, %d)\n", ws[0], ws[1], ws[2])

	// Declare workgroup variables at module scope
	for _, ins := range instrs {
		if ins.Kind == InstrDefineLocal {
			fmt.Fprintf(&b, "var<workgroup> %s: array<%s, %d>;\n",
				ins.LocalName, wgslDType(ins.DType), ins.LocalSize)
		}
	}

	b.WriteString("fn main(\n")
	b.WriteString("  @builtin(global_invocation_id) gid: vec3<u32>,\n")
	b.WriteString("  @builtin(workgroup_id) wid: vec3<u32>,\n")
	b.WriteString("  @builtin(local_invocation_id) lid: vec3<u32>\n")
	b.WriteString(") {\n")

	// Detect if Dimension 0 was spread into Y/Z.
	// This happens in lowerSink if dim 1 and 2 were originally unused.
	// We check if Dimension 1 was used by any GIDVar.
	dim1Used := false
	dim2Used := false
	for _, ins := range instrs {
		if ins.Kind == InstrGIDVar {
			if ins.Component == 1 {
				dim1Used = true
			}
			if ins.Component == 2 {
				dim2Used = true
			}
		}
	}

	// Compute flattened component IDs.
	// If dim1/dim2 are unused by the IR, then gid.y/z are for spreading dim 0.
	dimX := int64(wc[0] * ws[0])
	dimY := int64(wc[1] * ws[1])

	if !dim1Used && !dim2Used {
		fmt.Fprintf(&b, "  let flat_gid_x = gid.x + (gid.y * %du) + (gid.z * %du);\n", dimX, dimX*dimY)
		fmt.Fprintf(&b, "  let flat_wid_x = wid.x + (wid.y * %du) + (wid.z * %du);\n", wc[0], wc[0]*wc[1])
	} else {
		b.WriteString("  let flat_gid_x = gid.x;\n")
		b.WriteString("  let flat_wid_x = wid.x;\n")
	}
	b.WriteString("  let flat_gid_y = gid.y;\n")
	b.WriteString("  let flat_wid_y = wid.y;\n")
	b.WriteString("  let flat_gid_z = gid.z;\n")
	b.WriteString("  let flat_wid_z = wid.z;\n")

	// gid_x is always the full linear index for stores and symbolic bounds.
	fmt.Fprintf(&b, "  let gid_x = gid.x + (gid.y * %du) + (gid.z * %du);\n", dimX, dimX*dimY)

	depth := 1
	indent := func() string { return strings.Repeat("  ", depth) }

	for _, ins := range instrs {
		switch ins.Kind {
		case InstrBoundsCheck:
			if ins.Symbolic {
				if ins.ConcreteTrailing > 1 {
					fmt.Fprintf(&b, "%sif (gid_x >= params_n.n0 * %du) { return; }\n", indent(), ins.ConcreteTrailing)
				} else {
					fmt.Fprintf(&b, "%sif (gid_x >= params_n.n0) { return; }\n", indent())
				}
			}

		case InstrGIDVar:
			comp := [...]string{"x", "y", "z"}[ins.Component]
			level := [...]string{"gid", "wid", "lid"}[ins.Level]
			base := fmt.Sprintf("%s.%s", level, comp)
			if level != "lid" {
				base = fmt.Sprintf("flat_%s_%s", level, comp)
			}

			var expr string
			if ins.Symbolic {
				if ins.Stride == 1 {
					expr = fmt.Sprintf("i32(%s)", base)
				} else {
					expr = fmt.Sprintf("i32(%s / %du)", base, ins.Stride)
				}
			} else if ins.Stride == 1 {
				expr = fmt.Sprintf("i32(%s %% %du)", base, ins.RangeSize)
			} else {
				expr = fmt.Sprintf("i32((%s / %du) %% %du)", base, ins.Stride, ins.RangeSize)
			}
			fmt.Fprintf(&b, "%slet r%d: i32 = %s;\n", indent(), ins.RangeID, expr)
			// Axis guard: mask out threads in the padding of a workgroup dimension.
			if !ins.Symbolic && ins.RangeSize > 1 {
				fmt.Fprintf(&b, "%sif (r%d >= %d) { return; }\n", indent(), ins.RangeID, ins.RangeSize)
			}

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
			if ins.WGSLType != "" {
				wt = ins.WGSLType
			}
			if ins.Name != "" {
				fmt.Fprintf(&b, "%slet %s: %s = %s;\n", indent(), ins.Name, wt, ins.Expr)
			} else {
				fmt.Fprintf(&b, "%slet t%d: %s = %s;\n", indent(), ins.NodeIdx, wt, ins.Expr)
			}

		case InstrDefineLocal:
			// already rendered at module scope

		case InstrBarrier:
			fmt.Fprintf(&b, "%sworkgroupBarrier();\n", indent())

		case InstrIf:
			fmt.Fprintf(&b, "%sif (%s) {\n", indent(), ins.Expr)
			depth++

		case InstrEndIf:
			depth--
			fmt.Fprintf(&b, "%s}\n", indent())

		case InstrAssign:
			fmt.Fprintf(&b, "%s%s = %s;\n", indent(), ins.IndexExpr, ins.Expr)

		case InstrStore:
			idxExpr := ins.IndexExpr
			if ins.DType != nil && ins.DType.Scalar() == uop.Dtypes.BFloat16 {
				fmt.Fprintf(&b, "%sdata0[%s] = bitcast<u32>(%s) & 0xFFFF0000u;\n", indent(), idxExpr, ins.Expr)
			} else {
				fmt.Fprintf(&b, "%sdata0[%s] = %s;\n", indent(), idxExpr, ins.Expr)
			}
		}
	}

	b.WriteString("}\n")
	return b.String()
}

func accUpdateExpr(op uop.Op, accName, elemExpr string) string {
	switch op {
	case uop.OpAdd:
		return fmt.Sprintf("%s + %s", accName, elemExpr)
	case uop.OpMul:
		return fmt.Sprintf("%s * %s", accName, elemExpr)
	case uop.OpMax:
		return fmt.Sprintf("max(%s, %s)", accName, elemExpr)
	default:
		return fmt.Sprintf("%s + %s", accName, elemExpr)
	}
}

func aluExpr(op uop.Op, srcs []string, dtype *uop.DType) string {
	switch op {
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
	case uop.OpAdd:
		return fmt.Sprintf("(%s + %s)", srcs[0], srcs[1])
	case uop.OpSub:
		return fmt.Sprintf("(%s - %s)", srcs[0], srcs[1])
	case uop.OpMul:
		return fmt.Sprintf("(%s * %s)", srcs[0], srcs[1])
	case uop.OpFDiv:
		return fmt.Sprintf("(%s / %s)", srcs[0], srcs[1])
	case uop.OpIDiv:
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
	case uop.OpWhere:
		return fmt.Sprintf("select(%s, %s, %s)", srcs[2], srcs[1], srcs[0])
	case uop.OpMulAcc:
		return fmt.Sprintf("(%s + (%s * %s))", srcs[0], srcs[1], srcs[2])
	case uop.OpCast:
		return fmt.Sprintf("%s(%s)", wgslDType(dtype), srcs[0])
	case uop.OpBitcast:
		return fmt.Sprintf("bitcast<%s>(%s)", wgslDType(dtype), srcs[0])
	default:
		panic(fmt.Sprintf("codegen: unhandled ALU op %s", op))
	}
}

func constLiteral(u uop.UOp) string {
	dtype := u.DType()
	isF16 := dtype != nil && dtype.Scalar() == uop.Dtypes.Float16
	switch v := u.Arg().(type) {
	case float64:
		if math.IsInf(v, -1) {
			if isF16 {
				return "bitcast<f16>(0xFBFFu)"
			}
			return "bitcast<f32>(0xff7fffffu)"
		}
		if math.IsInf(v, 1) {
			if isF16 {
				return "bitcast<f16>(0x7BFFu)"
			}
			return "bitcast<f32>(0x7f7fffffu)"
		}
		if math.IsNaN(v) {
			if isF16 {
				return "bitcast<f16>(0x7E00u)"
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
				return "bitcast<f16>(0xFBFFu)"
			}
			return "bitcast<f32>(0xff7fffffu)"
		}
		if dtype.IsUnsigned() {
			return "0u"
		}
		return "-2147483648"
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

func wgslDType(d *uop.DType) string {
	if d == nil || d == uop.Dtypes.Void {
		return "f32"
	}
	s := d.Scalar()
	switch s {
	case uop.Dtypes.Float32:
		return "f32"
	case uop.Dtypes.Float16:
		return "f16"
	case uop.Dtypes.Int32:
		return "i32"
	case uop.Dtypes.UInt32:
		return "u32"
	case uop.Dtypes.Index:
		return "i32"
	case uop.Dtypes.Bool:
		return "bool"
	case uop.Dtypes.Int8, uop.Dtypes.Int16:
		return "i32"
	case uop.Dtypes.UInt8, uop.Dtypes.UInt16:
		return "u32"
	case uop.Dtypes.Int64:
		return "i32"
	case uop.Dtypes.UInt64:
		return "u32"
	default:
		if d.IsFloat() {
			return "f32"
		}
		return "i32"
	}
}

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
