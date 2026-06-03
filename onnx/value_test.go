package onnx

import (
	"fmt"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
	"github.com/georgebuilds/anneal/uop"
)

// expectPanic runs fn and returns the recovered panic value, or fails the test
// if fn returned normally.
func expectPanic(t *testing.T, name string, fn func()) any {
	t.Helper()
	var got any
	func() {
		defer func() { got = recover() }()
		fn()
	}()
	if got == nil {
		t.Fatalf("%s: expected panic, got none", name)
	}
	return got
}

// panicMessage stringifies a recover() value for substring assertions.
func panicMessage(p any) string {
	if s, ok := p.(string); ok {
		return s
	}
	if e, ok := p.(error); ok {
		return e.Error()
	}
	return fmt.Sprintf("%v", p)
}

// ── constructors round-trip ───────────────────────────────────────────────────

func TestHostInt64_RoundTrip(t *testing.T) {
	v := HostInt64(42)
	if v.Kind != KindHostInt64 {
		t.Errorf("Kind=%d, want KindHostInt64", v.Kind)
	}
	if got := v.Int64(); got != 42 {
		t.Errorf("Int64=%d, want 42", got)
	}
	if !v.IsHost() {
		t.Errorf("IsHost=false, want true for KindHostInt64")
	}
	if v.IsDevice() {
		t.Errorf("IsDevice=true, want false for KindHostInt64")
	}
}

func TestHostInts_RoundTrip(t *testing.T) {
	src := []int64{1, 2, 3, -4}
	v := HostInts(src)
	if v.Kind != KindHostInts {
		t.Errorf("Kind=%d, want KindHostInts", v.Kind)
	}
	got := v.Ints()
	if len(got) != len(src) {
		t.Fatalf("Ints length=%d, want %d", len(got), len(src))
	}
	for i := range src {
		if got[i] != src[i] {
			t.Errorf("Ints[%d]=%d, want %d", i, got[i], src[i])
		}
	}
	if !v.IsInts() {
		t.Errorf("IsInts=false, want true")
	}
	if v.IsSints() {
		t.Errorf("IsSints=true, want false")
	}
	if !v.IsHost() {
		t.Errorf("IsHost=false, want true")
	}
}

func TestHostInts_NilAndEmpty(t *testing.T) {
	cases := []struct {
		name string
		in   []int64
	}{
		{"nil", nil},
		{"empty", []int64{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := HostInts(c.in)
			if v.Kind != KindHostInts {
				t.Errorf("Kind=%d, want KindHostInts", v.Kind)
			}
			if got := v.Ints(); len(got) != 0 {
				t.Errorf("Ints length=%d, want 0", len(got))
			}
		})
	}
}

func TestHostSints_RoundTrip(t *testing.T) {
	src := []shape.Sint{shape.Const(2), shape.Const(3)}
	v := HostSints(src)
	if v.Kind != KindHostSints {
		t.Errorf("Kind=%d, want KindHostSints", v.Kind)
	}
	got := v.Sints()
	if len(got) != len(src) {
		t.Fatalf("Sints length=%d, want %d", len(got), len(src))
	}
	for i, s := range src {
		gv, gok := got[i].ConstValue()
		sv, sok := s.ConstValue()
		if gok != sok || gv != sv {
			t.Errorf("Sints[%d]=(%d,%v), want (%d,%v)", i, gv, gok, sv, sok)
		}
	}
	if !v.IsSints() {
		t.Errorf("IsSints=false, want true")
	}
	if v.IsInts() {
		t.Errorf("IsInts=true, want false")
	}
	if !v.IsHost() {
		t.Errorf("IsHost=false, want true")
	}
}

func TestHostFloat64_RoundTrip(t *testing.T) {
	v := HostFloat64(2.5)
	if v.Kind != KindHostFloat64 {
		t.Errorf("Kind=%d, want KindHostFloat64", v.Kind)
	}
	if got := v.Float64(); got != 2.5 {
		t.Errorf("Float64=%v, want 2.5", got)
	}
	if !v.IsHost() {
		t.Errorf("IsHost=false, want true")
	}
}

func TestHostFloats_RoundTrip(t *testing.T) {
	src := []float64{1.0, -2.5, 3.25}
	v := HostFloats(src)
	if v.Kind != KindHostFloats {
		t.Errorf("Kind=%d, want KindHostFloats", v.Kind)
	}
	got := v.Floats()
	if len(got) != len(src) {
		t.Fatalf("Floats length=%d, want %d", len(got), len(src))
	}
	for i := range src {
		if got[i] != src[i] {
			t.Errorf("Floats[%d]=%v, want %v", i, got[i], src[i])
		}
	}
	if !v.IsHost() {
		t.Errorf("IsHost=false, want true")
	}
}

func TestDevice_RoundTrip(t *testing.T) {
	arena := uop.NewArena(8)
	leaf := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	leaf.SetData([]float32{1})
	v := Device(leaf)
	if v.Kind != KindDevice {
		t.Errorf("Kind=%d, want KindDevice", v.Kind)
	}
	if got := v.Tensor(); got != leaf {
		t.Errorf("Tensor() != original leaf")
	}
	if !v.IsDevice() {
		t.Errorf("IsDevice=false, want true")
	}
	if v.IsHost() {
		t.Errorf("IsHost=true, want false for Device")
	}
}

// ── cross-kind accessor panics ────────────────────────────────────────────────

func TestValue_CrossKindAccessorPanics(t *testing.T) {
	// Each (constructor, illegal accessor) pair must panic with a message
	// that mentions the accessor and the actual kind.
	arena := uop.NewArena(4)
	leaf := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")

	cases := []struct {
		name    string
		v       Value
		fn      func(Value)
		wantSub string
	}{
		{"Int64-on-HostFloat64", HostFloat64(1.0), func(v Value) { _ = v.Int64() }, "Int64"},
		{"Ints-on-HostInt64", HostInt64(1), func(v Value) { _ = v.Ints() }, "Ints"},
		{"Sints-on-HostInts", HostInts([]int64{1}), func(v Value) { _ = v.Sints() }, "Sints"},
		{"Float64-on-HostInt64", HostInt64(1), func(v Value) { _ = v.Float64() }, "Float64"},
		{"Floats-on-HostInt64", HostInt64(1), func(v Value) { _ = v.Floats() }, "Floats"},
		{"Tensor-on-HostInt64", HostInt64(1), func(v Value) { _ = v.Tensor() }, "Tensor"},
		{"Int64-on-Device", Device(leaf), func(v Value) { _ = v.Int64() }, "Int64"},
		{"Tensor-on-HostInts", HostInts([]int64{1}), func(v Value) { _ = v.Tensor() }, "Tensor"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := expectPanic(t, c.name, func() { c.fn(c.v) })
			msg := panicMessage(p)
			if !strings.Contains(msg, c.wantSub) {
				t.Errorf("panic message %q does not contain %q", msg, c.wantSub)
			}
			// Panic also mentions kind=<n> so the caller can diagnose.
			if !strings.Contains(msg, "kind=") {
				t.Errorf("panic message %q does not contain 'kind='", msg)
			}
		})
	}
}

// ── IsHost / IsDevice exhaustive ──────────────────────────────────────────────

func TestValue_IsHostExhaustive(t *testing.T) {
	arena := uop.NewArena(4)
	leaf := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	cases := []struct {
		name string
		v    Value
		host bool
		dev  bool
	}{
		{"Unset", Value{}, false, false},
		{"HostInt64", HostInt64(1), true, false},
		{"HostInts", HostInts([]int64{1}), true, false},
		{"HostSints", HostSints([]shape.Sint{shape.Const(1)}), true, false},
		{"HostFloat64", HostFloat64(1.0), true, false},
		{"HostFloats", HostFloats([]float64{1.0}), true, false},
		{"Device", Device(leaf), false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.v.IsHost(); got != c.host {
				t.Errorf("IsHost=%v, want %v", got, c.host)
			}
			if got := c.v.IsDevice(); got != c.dev {
				t.Errorf("IsDevice=%v, want %v", got, c.dev)
			}
		})
	}
}

// TestValue_IsIntsSemantics pins down that IsInts() is *strict*: only
// KindHostInts returns true. KindHostInt64 returns false even though the
// scalar can be converted to a length-1 int slice via ToSints.
func TestValue_IsIntsSemantics(t *testing.T) {
	if HostInt64(1).IsInts() {
		t.Errorf("HostInt64.IsInts=true, want false (IsInts is strict to KindHostInts)")
	}
	if !HostInts([]int64{1}).IsInts() {
		t.Errorf("HostInts.IsInts=false, want true")
	}
	if HostSints([]shape.Sint{shape.Const(1)}).IsInts() {
		t.Errorf("HostSints.IsInts=true, want false")
	}
}

// ── ToSints conversion ────────────────────────────────────────────────────────

func TestToSints_HostInt64(t *testing.T) {
	v := HostInt64(7)
	got := v.ToSints()
	if len(got) != 1 {
		t.Fatalf("ToSints length=%d, want 1", len(got))
	}
	gv, ok := got[0].ConstValue()
	if !ok {
		t.Fatalf("ToSints[0] not concrete")
	}
	if gv != 7 {
		t.Errorf("ToSints[0]=%d, want 7", gv)
	}
}

func TestToSints_HostInts(t *testing.T) {
	src := []int64{1, 2, 3}
	v := HostInts(src)
	got := v.ToSints()
	if len(got) != len(src) {
		t.Fatalf("ToSints length=%d, want %d", len(got), len(src))
	}
	for i, want := range src {
		gv, ok := got[i].ConstValue()
		if !ok {
			t.Errorf("ToSints[%d] not concrete", i)
			continue
		}
		if gv != want {
			t.Errorf("ToSints[%d]=%d, want %d", i, gv, want)
		}
	}
}

func TestToSints_HostSintsPassthrough(t *testing.T) {
	src := []shape.Sint{shape.Const(10), shape.Const(20)}
	v := HostSints(src)
	got := v.ToSints()
	if len(got) != len(src) {
		t.Fatalf("ToSints length=%d, want %d", len(got), len(src))
	}
	for i, want := range src {
		wv, _ := want.ConstValue()
		gv, ok := got[i].ConstValue()
		if !ok || gv != wv {
			t.Errorf("ToSints[%d]=(%d,%v), want (%d,true)", i, gv, ok, wv)
		}
	}
}

func TestToSints_PanicOnNonInteger(t *testing.T) {
	arena := uop.NewArena(4)
	leaf := tensor.NewLeaf(arena, []int64{1}, uop.Dtypes.Float32, "test")
	cases := []struct {
		name string
		v    Value
	}{
		{"HostFloat64", HostFloat64(1.0)},
		{"HostFloats", HostFloats([]float64{1.0})},
		{"Device", Device(leaf)},
		{"Unset", Value{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := expectPanic(t, c.name, func() { _ = c.v.ToSints() })
			msg := panicMessage(p)
			if !strings.Contains(msg, "ToSints") {
				t.Errorf("panic %q missing 'ToSints'", msg)
			}
			if !strings.Contains(msg, "kind=") {
				t.Errorf("panic %q missing 'kind='", msg)
			}
		})
	}
}
