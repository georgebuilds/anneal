// Package examples provides a registry of named model examples that can be
// built and trained via the anneal CLI. Each example exposes a Build function
// that constructs a forward-graph root and a Train function that runs a full
// training loop with periodic loss logging.
package examples

import (
	"fmt"

	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// BuildResult is the output of an example's Build function.
type BuildResult struct {
	Arena  *uop.Arena
	Output *tensor.Tensor   // unrealized forward graph root
	Device string
	Leaves []*tensor.Tensor // leaf parameter tensors for the backward pass; nil = not set
}

// TrainConfig holds hyperparameters for a training loop.
type TrainConfig struct {
	Steps    int
	LR       float32
	LogEvery int          // log loss every N steps
	OnStep   func(int)    // called on every step (nil = no-op); used by the TUI for smooth progress
}

// Example is a named, runnable model example.
type Example struct {
	Name    string
	Summary string
	Build   func(device string) (*BuildResult, error)
	Train   func(device string, cfg TrainConfig, logFn func(step int, loss float32)) error
}

var (
	order    []*Example
	registry = map[string]*Example{}
)

// Register adds an Example to the registry. Called from init() in each example file.
func Register(e *Example) {
	if _, dup := registry[e.Name]; dup {
		panic(fmt.Sprintf("examples: duplicate registration: %q", e.Name))
	}
	registry[e.Name] = e
	order = append(order, e)
}

// Get returns the named example or an error if it is not registered.
func Get(name string) (*Example, error) {
	e, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("example %q not found — available: %s", name, listNames())
	}
	return e, nil
}

// All returns all registered examples in registration order.
func All() []*Example {
	out := make([]*Example, len(order))
	copy(out, order)
	return out
}

func listNames() string {
	if len(order) == 0 {
		return "(none)"
	}
	s := ""
	for i, e := range order {
		if i > 0 {
			s += ", "
		}
		s += e.Name
	}
	return s
}
