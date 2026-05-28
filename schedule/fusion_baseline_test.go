package schedule_test

import (
	"fmt"
	"testing"

	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestFusionBaselineReport(t *testing.T) {
	cases := []struct {
		name string
		build func(a *uop.Arena) []*tensor.Tensor
	}{
		{
			name: "1. pure elementwise chain",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{1024}, uop.Dtypes.Float32, "cpu")
				y := tensor.NewLeaf(a, []int64{1024}, uop.Dtypes.Float32, "cpu")
				z := tensor.NewLeaf(a, []int64{1024}, uop.Dtypes.Float32, "cpu")
				w := tensor.NewLeaf(a, []int64{1024}, uop.Dtypes.Float32, "cpu")
				// (x+y)*z - w
				res := x.Add(y).Mul(z).Sub(w)
				return []*tensor.Tensor{res}
			},
		},
		{
			name: "2. single matmul",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "cpu")
				w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "cpu")
				res := x.Matmul(w)
				return []*tensor.Tensor{res}
			},
		},
		{
			name: "3. matmul + bias add",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "cpu")
				w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "cpu")
				b := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float32, "cpu")
				res := x.Matmul(w).Add(b)
				return []*tensor.Tensor{res}
			},
		},
		{
			name: "4. matmul + bias + relu",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "cpu")
				w := tensor.NewLeaf(a, []int64{32, 64}, uop.Dtypes.Float32, "cpu")
				b := tensor.NewLeaf(a, []int64{64}, uop.Dtypes.Float32, "cpu")
				res := nn.ReLU(x.Matmul(w).Add(b))
				return []*tensor.Tensor{res}
			},
		},
		{
			name: "5. small MLP forward",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "cpu")
				l1 := nn.NewLinear(a, 32, 64, true, uop.Dtypes.Float32, "cpu")
				l2 := nn.NewLinear(a, 64, 10, true, uop.Dtypes.Float32, "cpu")
				l1.Weight.Load(a)
				l1.Bias.Load(a)
				l2.Weight.Load(a)
				l2.Bias.Load(a)
				
				h := nn.ReLU(l1.Forward(x))
				out := l2.Forward(h)
				return []*tensor.Tensor{out}
			},
		},
		{
			name: "6. small MLP forward + backward",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{16, 32}, uop.Dtypes.Float32, "cpu")
				tgt := tensor.NewLeaf(a, []int64{16, 10}, uop.Dtypes.Float32, "cpu")
				l1 := nn.NewLinear(a, 32, 64, true, uop.Dtypes.Float32, "cpu")
				l2 := nn.NewLinear(a, 64, 10, true, uop.Dtypes.Float32, "cpu")
				l1.Weight.Load(a)
				l1.Bias.Load(a)
				l2.Weight.Load(a)
				l2.Bias.Load(a)
				
				h := nn.ReLU(l1.Forward(x))
				out := l2.Forward(h)
				
				diff := out.Sub(tgt)
				loss := diff.Mul(diff).Sum(nil, false)
				
				leaves := []*tensor.Tensor{l1.Weight.T, l1.Bias.T, l2.Weight.T, l2.Bias.T}
				grads := tensor.Backward(loss, leaves)
				
				results := []*tensor.Tensor{loss}
				for _, l := range leaves {
					if g, ok := grads[l]; ok {
						results = append(results, g)
					}
				}
				return results
			},
		},
		{
			name: "7. one conv forward",
			build: func(a *uop.Arena) []*tensor.Tensor {
				x := tensor.NewLeaf(a, []int64{1, 3, 32, 32}, uop.Dtypes.Float32, "cpu")
				conv := nn.NewConv2d(a, 3, 16, [2]int64{3, 3}, [2]int{1, 1}, [2]int{1, 1}, true, uop.Dtypes.Float32, "cpu")
				conv.Weight.Load(a)
				conv.Bias.Load(a)
				out := conv.Forward(x)
				return []*tensor.Tensor{out}
			},
		},
	}

	fmt.Println("FUSION BASELINE REPORT")
	fmt.Println("======================")

	for _, tc := range cases {
		a := uop.NewArena(1024 * 1024)
		tensors := tc.build(a)
		
		sinkNodes := make([]uop.UOp, len(tensors))
		for i, t := range tensors {
			sinkNodes[i] = t.Node()
		}
		sink := a.New(uop.OpSink, uop.Dtypes.Void, sinkNodes, nil, nil)
		
		items := schedule.CreateSchedule(sink, "cpu")
		
		fmt.Printf("\nGraph: %s\n", tc.name)
		fmt.Printf("  Total Kernel Count: %d\n", len(items))
		
		maxBufs := 0
		for i, item := range items {
			nBufs := len(item.Bufs)
			if nBufs > maxBufs {
				maxBufs = nBufs
			}
			
			// Analyze the kernel to see if it's a reduce or elementwise or other.
			isReduce := false
			isOther := false
			var walk func(u uop.UOp)
			seen := make(map[uint32]bool)
			walk = func(u uop.UOp) {
				if !u.Valid() || seen[u.Index()] {
					return
				}
				seen[u.Index()] = true
				if u.Op() == uop.OpReduce {
					isReduce = true
					// Don't return, keep walking to see if there are other interesting ops
				}
				// We also check for signs of "other" hard boundaries if they were to survive,
				// though in kernel AST they are usually dissolved.
				// For v1, the producer of the bufferize was the tensor-level op.
				// Since we only have the kernel AST, we infer from the content.
				for j := 0; j < u.NSrc(); j++ {
					walk(u.Src(j))
				}
			}
			walk(item.Ast)
			
			prodOp := "elementwise"
			if isReduce {
				prodOp = "reduce"
			} else if isOther {
				prodOp = "other"
			}
			
			// In v1, all BUFFERIZE are forced (Removable=false).
			// We report it as forced.
			fmt.Printf("  Kernel %d: producer=%s, bufs=%d, removable=false\n", i, prodOp, nBufs)
		}
		fmt.Printf("  Max Buffers-per-Kernel: %d (WebGPU cap: 8)\n", maxBufs)
	}
}
