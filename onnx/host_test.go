package onnx

import (
	"fmt"
	"strings"
	"testing"
)

// TestNewHostState_Empty checks the freshly-allocated HostState is empty.
func TestNewHostState_Empty(t *testing.T) {
	s := NewHostState()
	if s == nil {
		t.Fatalf("NewHostState returned nil")
	}
	if _, ok := s.Get("missing"); ok {
		t.Errorf("Get on empty state returned ok=true")
	}
}

func TestHostState_SetThenGet(t *testing.T) {
	s := NewHostState()
	want := HostInt64(42)
	s.Set("x", want)
	got, ok := s.Get("x")
	if !ok {
		t.Fatalf("Get(x) ok=false after Set")
	}
	if got.Kind != want.Kind || got.Int64() != 42 {
		t.Errorf("Get(x)=%+v, want %+v", got, want)
	}
}

func TestHostState_GetAbsent(t *testing.T) {
	s := NewHostState()
	s.Set("present", HostInt64(1))
	v, ok := s.Get("missing")
	if ok {
		t.Errorf("Get(missing) ok=true, want false")
	}
	if v.Kind != KindUnset {
		t.Errorf("Get(missing) returned non-zero Value: %+v", v)
	}
}

func TestHostState_Overwrite(t *testing.T) {
	s := NewHostState()
	s.Set("k", HostInt64(1))
	s.Set("k", HostInt64(2))
	got, ok := s.Get("k")
	if !ok {
		t.Fatalf("Get(k) ok=false")
	}
	if got.Int64() != 2 {
		t.Errorf("Get(k)=%d, want 2 (overwrite semantics)", got.Int64())
	}
}

// ── host op registry ──────────────────────────────────────────────────────────

// uniqueOpName returns a unique op-type string per test so the package-level
// hostOps map doesn't leak across tests in the same binary.
func uniqueOpName(t *testing.T) string {
	t.Helper()
	return "TestHostOp_" + t.Name() + "_" + fmt.Sprintf("%p", t)
}

func TestRegisterHostOp_ThenIsHostOp(t *testing.T) {
	name := uniqueOpName(t)
	if IsHostOp(name) {
		t.Fatalf("op %q registered before test", name)
	}
	RegisterHostOp(name, func(node *Node, inputs []Value, st *HostState) (Value, error) {
		return HostInt64(7), nil
	})
	if !IsHostOp(name) {
		t.Errorf("IsHostOp(%q)=false after register, want true", name)
	}
}

func TestIsHostOp_Unregistered(t *testing.T) {
	if IsHostOp("DefinitelyNotRegistered_zxqvyw") {
		t.Errorf("IsHostOp returned true for unregistered op")
	}
}

func TestRegisterHostOp_DoubleRegisterPanics(t *testing.T) {
	name := uniqueOpName(t)
	RegisterHostOp(name, func(node *Node, inputs []Value, st *HostState) (Value, error) {
		return HostInt64(1), nil
	})
	defer func() {
		p := recover()
		if p == nil {
			t.Fatalf("expected panic on double register, got none")
		}
		msg := fmt.Sprintf("%v", p)
		if !strings.Contains(msg, "already registered") {
			t.Errorf("panic %q missing 'already registered'", msg)
		}
		if !strings.Contains(msg, name) {
			t.Errorf("panic %q missing op name %q", msg, name)
		}
	}()
	RegisterHostOp(name, func(node *Node, inputs []Value, st *HostState) (Value, error) {
		return HostInt64(2), nil
	})
}

// ── evalHost dispatch ─────────────────────────────────────────────────────────

func TestEvalHost_Dispatches(t *testing.T) {
	name := uniqueOpName(t)
	called := false
	RegisterHostOp(name, func(node *Node, inputs []Value, st *HostState) (Value, error) {
		called = true
		if node.OpType != name {
			t.Errorf("handler saw OpType=%q, want %q", node.OpType, name)
		}
		// Verify the host state pointer is the one we passed in.
		st.Set("__sentinel__", HostInt64(123))
		return HostInts([]int64{1, 2, 3}), nil
	})

	st := NewHostState()
	node := &Node{OpType: name}
	got, err := evalHost(node, nil, st)
	if err != nil {
		t.Fatalf("evalHost err=%v", err)
	}
	if !called {
		t.Errorf("registered handler was not invoked")
	}
	if got.Kind != KindHostInts {
		t.Fatalf("evalHost returned kind=%d, want KindHostInts", got.Kind)
	}
	ints := got.Ints()
	if len(ints) != 3 || ints[0] != 1 || ints[1] != 2 || ints[2] != 3 {
		t.Errorf("evalHost result=%v, want [1 2 3]", ints)
	}
	sentinel, ok := st.Get("__sentinel__")
	if !ok || sentinel.Int64() != 123 {
		t.Errorf("handler did not write to the passed HostState")
	}
}

func TestEvalHost_Unregistered(t *testing.T) {
	st := NewHostState()
	node := &Node{OpType: "TotallyUnregisteredOp_abc"}
	_, err := evalHost(node, nil, st)
	if err == nil {
		t.Fatalf("expected error on unregistered op, got nil")
	}
	if !strings.Contains(err.Error(), "not implemented") {
		t.Errorf("error %q missing 'not implemented'", err.Error())
	}
	if !strings.Contains(err.Error(), "TotallyUnregisteredOp_abc") {
		t.Errorf("error %q missing op name", err.Error())
	}
}

func TestEvalHost_HandlerError(t *testing.T) {
	name := uniqueOpName(t)
	RegisterHostOp(name, func(node *Node, inputs []Value, st *HostState) (Value, error) {
		return Value{}, fmt.Errorf("nope")
	})
	st := NewHostState()
	node := &Node{OpType: name}
	_, err := evalHost(node, nil, st)
	if err == nil {
		t.Fatalf("expected handler error to propagate, got nil")
	}
	if err.Error() != "nope" {
		t.Errorf("propagated error=%q, want %q", err.Error(), "nope")
	}
}
