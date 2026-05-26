package main

import (
	"fmt"
	"io"
	"os"

	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/uop"
)

func graphCmd(args []string) int {
	return graphCmdW(args, os.Stdout)
}

func graphCmdW(args []string, w io.Writer) int {
	_, rest, err := parseFlags("graph", args)
	if err != nil {
		fmt.Fprintln(w, err)
		return 1
	}

	if len(rest) == 0 {
		fmt.Fprintln(w, "usage: anneal graph <model>")
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "available models:")
		for _, e := range examples.All() {
			fmt.Fprintf(w, "  %-12s  %s\n", e.Name, e.Summary)
		}
		return 1
	}

	name := rest[0]
	ex, err := examples.Get(name)
	if err != nil {
		fmt.Fprintln(w, formatError(err.Error()))
		return 1
	}

	result, err := ex.Build("webgpu")
	if err != nil {
		fmt.Fprintf(w, "build error: %v\n", err)
		return 1
	}

	// Count reachable nodes for header.
	nodes := topoSortNodes(result.Output.Node())

	fmt.Fprintf(w, "graph: %s — %s\n", ex.Name, ex.Summary)
	fmt.Fprintf(w, "device: webgpu\n")
	fmt.Fprintf(w, "UOp nodes: %d\n", len(nodes))
	fmt.Fprintln(w)

	dumpDAG(w, result.Output.Node())
	return 0
}

// dumpDAG prints all UOp nodes reachable from root in topological order.
// Format: index (4 chars), op (12 chars), dtype (8 chars), then args/shape info.
func dumpDAG(w io.Writer, root uop.UOp) {
	nodes := topoSortNodes(root)
	for _, u := range nodes {
		idx := u.Index()
		op := u.Op().String()
		dt := u.DType()
		dtName := "?"
		if dt != nil && dt != uop.Dtypes.Void {
			dtName = dtTypeName(dt)
		} else if dt == uop.Dtypes.Void {
			dtName = "void"
		}

		line := fmt.Sprintf("%4d  %-12s  %-8s", idx, op, dtName)

		switch u.Op() {
		case uop.OpBuffer:
			// Show shape, and <leaf> if it has data.
			sh := bufferShape(u)
			line += fmt.Sprintf("shape=%v", sh)
			_, hasData := u.Arena().Leaf(idx)
			if hasData {
				line += "   <leaf>"
			}

		case uop.OpConst:
			line += fmt.Sprintf("arg=%v", u.Arg())

		case uop.OpReduceAxis:
			ra, _ := u.Arg().(uop.ReduceArg)
			srcs := srcList(u)
			line += fmt.Sprintf("srcs=%v  op=%s axes=%v", srcs, ra.Op, ra.Axes)

		default:
			if u.NSrc() > 0 {
				srcs := srcList(u)
				line += fmt.Sprintf("srcs=%v", srcs)
				if u.Arg() != nil {
					line += fmt.Sprintf("  arg=%v", u.Arg())
				}
			} else if u.Arg() != nil {
				line += fmt.Sprintf("arg=%v", u.Arg())
			}
		}

		fmt.Fprintln(w, line)
	}
}

// topoSortNodes returns all nodes reachable from root in topological post-order
// (sources before consumers) using iterative DFS.
func topoSortNodes(root uop.UOp) []uop.UOp {
	seen := make(map[uint32]bool)
	var order []uop.UOp

	type frame struct {
		u       uop.UOp
		nextSrc int
	}
	stack := []frame{{root, 0}}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		u := f.u

		if seen[u.Index()] {
			stack = stack[:len(stack)-1]
			continue
		}

		pushed := false
		for f.nextSrc < u.NSrc() {
			child := u.Src(f.nextSrc)
			f.nextSrc++
			if !seen[child.Index()] {
				stack = append(stack, frame{child, 0})
				pushed = true
				break
			}
		}
		if !pushed {
			seen[u.Index()] = true
			order = append(order, u)
			stack = stack[:len(stack)-1]
		}
	}

	return order
}

// srcList returns the arena indices of u's sources as a slice for display.
func srcList(u uop.UOp) []uint32 {
	out := make([]uint32, u.NSrc())
	for i := 0; i < u.NSrc(); i++ {
		out[i] = u.Src(i).Index()
	}
	return out
}

// bufferShape extracts the shape from a Buffer node's arg.
func bufferShape(u uop.UOp) []int64 {
	switch v := u.Arg().(type) {
	case []int64:
		return v
	case int64:
		return []int64{v}
	}
	return nil
}

// dtTypeName returns a short dtype name for display.
func dtTypeName(dt *uop.DType) string {
	if dt == nil {
		return "?"
	}
	// Map from the DType singleton names to short display names.
	switch dt {
	case uop.Dtypes.Float32:
		return "f32"
	case uop.Dtypes.Float16:
		return "f16"
	case uop.Dtypes.Float64:
		return "f64"
	case uop.Dtypes.Int32:
		return "i32"
	case uop.Dtypes.UInt32:
		return "u32"
	case uop.Dtypes.Int64:
		return "i64"
	case uop.Dtypes.UInt64:
		return "u64"
	case uop.Dtypes.Int8:
		return "i8"
	case uop.Dtypes.UInt8:
		return "u8"
	case uop.Dtypes.Bool:
		return "bool"
	case uop.Dtypes.Index:
		return "index"
	case uop.Dtypes.Void:
		return "void"
	}
	return dt.String()
}
