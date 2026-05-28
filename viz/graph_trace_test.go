//go:build !js

package viz

import (
	"testing"

	"github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

func TestBuildGraph_GradRuleAttribution(t *testing.T) {
	// Build MLP graph
	g, err := BuildGraph("mlp")
	if err != nil {
		t.Fatal(err)
	}

	// Find an adjoint of a Mul op.
	// In MLP: x @ W.T involves Mul (or Matmul which expands to Mul).
	foundMul := false
	for _, n := range g.Nodes {
		if n.Class == ClassBackward && n.GradRule == "Mul" {
			foundMul = true
			if n.GradFiredSeq < 0 {
				t.Errorf("node %d has GradRule 'Mul' but GradFiredSeq %d", n.ID, n.GradFiredSeq)
			}
		}
	}
	if !foundMul {
		t.Error("could not find any backward node attributed to 'Mul' rule")
	}
}

func TestBuildGraph_AccumulationNodeUnattributed(t *testing.T) {
	// Register a custom example with a diamond pattern.
	examples.Register(&examples.Example{
		Name: "diamond",
		Build: func(device string) (*examples.BuildResult, error) {
			a := uop.NewArena(1024)
			x := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Float32, device)
			// y = x + x
			y := x.Add(x)
			return &examples.BuildResult{
				Arena:  a,
				Output: y,
				Leaves: []*tensor.Tensor{x},
				Device: device,
			}, nil
		},
	})

	g, err := BuildGraph("diamond")
	if err != nil {
		t.Fatal(err)
	}

	// Find the accumulation node. It should be an OpAdd in ClassBackward.
	// The backward pass of 'Add(x, x)' produces two adjoints (ones, ones),
	// and the driver sums them.
	foundAccum := false
	for _, n := range g.Nodes {
		// Accumulation nodes are added by the driver and don't come from a rule firing.
		if n.Class == ClassBackward && n.Op == "Add" && n.GradRule == "" {
			foundAccum = true
		}
	}
	if !foundAccum {
		t.Error("expected an unattributed OpAdd backward node for accumulation")
	}
}

func TestGradTraceOrder(t *testing.T) {
	// Build a simple chain: x -> Sin -> Sin -> loss
	a := uop.NewArena(1024)
	x := tensor.NewLeaf(a, []int64{1}, uop.Dtypes.Float32, "cpu")
	y := x.Sin()
	z := y.Sin()
	loss := z.Sum(nil, false)

	_, trace := tensor.BackwardWithTrace(loss, []*tensor.Tensor{x})
	if trace == nil {
		t.Fatal("trace is nil")
	}

	// Expected firing order (reverse-topo of forward):
	// 1. Sum rule (on z) - OpReduceAxis
	// 2. Sin rule (on y) - OpSin
	// 3. Sin rule (on x) - OpSin

	if len(trace.Events) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(trace.Events))
	}

	// Verify Seq is strictly increasing
	for i := 0; i < len(trace.Events); i++ {
		if trace.Events[i].Seq != i {
			t.Errorf("event %d: expected Seq %d, got %d", i, i, trace.Events[i].Seq)
		}
	}

	// Verify reverse-topo order of ops
	if trace.Events[0].ForwardOp != uop.OpReduceAxis {
		t.Errorf("event 0: expected OpReduceAxis, got %s", trace.Events[0].ForwardOp)
	}
	if trace.Events[1].ForwardOp != uop.OpSin {
		t.Errorf("event 1: expected OpSin, got %s", trace.Events[1].ForwardOp)
	}
	if trace.Events[2].ForwardOp != uop.OpSin {
		t.Errorf("event 2: expected OpSin, got %s", trace.Events[2].ForwardOp)
	}
}
