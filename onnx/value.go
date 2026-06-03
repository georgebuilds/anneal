package onnx

import (
	"fmt"

	"github.com/georgebuilds/anneal/shape"
	"github.com/georgebuilds/anneal/tensor"
)

// ValueKind discriminates the host/device split mandated by the importer's
// architectural spine (see implementation plan §1). Shape-arithmetic walks the
// host tier; tensor math walks the device tier. A Value is exactly one of
// these; mismatches at runtime panic with a descriptive message so that
// importer bugs surface as loud failures rather than silently-wrong outputs.
type ValueKind int

const (
	// KindUnset is the zero value. Indicates a value was never assigned;
	// accessing one in that state is a programmer error.
	KindUnset ValueKind = iota
	// KindHostInt64 carries a single concrete int64 scalar.
	KindHostInt64
	// KindHostInts carries a concrete []int64 vector (e.g. a Shape result).
	KindHostInts
	// KindHostSints carries a []shape.Sint vector that may include symbolic
	// dims (e.g. Shape of a tensor with a dim_param input).
	KindHostSints
	// KindHostFloat64 carries a single concrete float64 scalar.
	KindHostFloat64
	// KindHostFloats carries a concrete []float64 vector.
	KindHostFloats
	// KindDevice carries an anneal *tensor.Tensor (a node in the UOp graph).
	KindDevice
)

// Value is the host/device sum type. The kind discriminant determines which
// field is populated. Helpers (HostInt64, HostInts, ...) are the only sanctioned
// way to construct a Value; direct field access is allowed but discouraged.
type Value struct {
	Kind ValueKind

	// Host payload fields. Exactly one is meaningful for a given Kind.
	I  int64
	F  float64
	Is []int64
	Fs []float64
	Ss []shape.Sint

	// Device payload.
	T *tensor.Tensor
}

// HostInt64 wraps a scalar int64 (e.g. a Size or a scalar Constant).
func HostInt64(v int64) Value { return Value{Kind: KindHostInt64, I: v} }

// HostInts wraps an int64 vector (e.g. the result of Shape on a fully-concrete
// tensor, a Reshape target, or a Slice's starts).
func HostInts(v []int64) Value { return Value{Kind: KindHostInts, Is: v} }

// HostSints wraps a Sint vector (e.g. Shape of a tensor with a symbolic dim).
// Use this whenever any element may be symbolic; promote pure []int64 via
// ToSints() when symbolic arithmetic is needed.
func HostSints(v []shape.Sint) Value { return Value{Kind: KindHostSints, Ss: v} }

// HostFloat64 wraps a scalar float64.
func HostFloat64(v float64) Value { return Value{Kind: KindHostFloat64, F: v} }

// HostFloats wraps a float64 vector.
func HostFloats(v []float64) Value { return Value{Kind: KindHostFloats, Fs: v} }

// Device wraps a *tensor.Tensor in a Value.
func Device(t *tensor.Tensor) Value { return Value{Kind: KindDevice, T: t} }

// IsHost reports whether v is any host-tier kind.
func (v Value) IsHost() bool {
	switch v.Kind {
	case KindHostInt64, KindHostInts, KindHostSints, KindHostFloat64, KindHostFloats:
		return true
	}
	return false
}

// IsDevice reports whether v is a device tensor.
func (v Value) IsDevice() bool { return v.Kind == KindDevice }

// IsInts reports whether v carries an []int64 vector.
func (v Value) IsInts() bool { return v.Kind == KindHostInts }

// IsSints reports whether v carries a []shape.Sint vector.
func (v Value) IsSints() bool { return v.Kind == KindHostSints }

// Int64 returns the scalar int64; panics if Kind is not KindHostInt64.
func (v Value) Int64() int64 {
	if v.Kind != KindHostInt64 {
		panic(fmt.Sprintf("onnx.Value.Int64: kind=%d, want KindHostInt64", v.Kind))
	}
	return v.I
}

// Ints returns the int64 vector; panics if Kind is not KindHostInts.
func (v Value) Ints() []int64 {
	if v.Kind != KindHostInts {
		panic(fmt.Sprintf("onnx.Value.Ints: kind=%d, want KindHostInts", v.Kind))
	}
	return v.Is
}

// Sints returns the Sint vector; panics if Kind is not KindHostSints.
func (v Value) Sints() []shape.Sint {
	if v.Kind != KindHostSints {
		panic(fmt.Sprintf("onnx.Value.Sints: kind=%d, want KindHostSints", v.Kind))
	}
	return v.Ss
}

// Float64 returns the scalar float64; panics if Kind is not KindHostFloat64.
func (v Value) Float64() float64 {
	if v.Kind != KindHostFloat64 {
		panic(fmt.Sprintf("onnx.Value.Float64: kind=%d, want KindHostFloat64", v.Kind))
	}
	return v.F
}

// Floats returns the float64 vector; panics if Kind is not KindHostFloats.
func (v Value) Floats() []float64 {
	if v.Kind != KindHostFloats {
		panic(fmt.Sprintf("onnx.Value.Floats: kind=%d, want KindHostFloats", v.Kind))
	}
	return v.Fs
}

// Tensor returns the device tensor; panics if Kind is not KindDevice.
func (v Value) Tensor() *tensor.Tensor {
	if v.Kind != KindDevice {
		panic(fmt.Sprintf("onnx.Value.Tensor: kind=%d, want KindDevice", v.Kind))
	}
	return v.T
}

// ToSints promotes a host-tier integer Value to a []shape.Sint. KindHostSints
// returns its payload unchanged; KindHostInts lifts each int64 via shape.Const;
// KindHostInt64 lifts the scalar to a length-1 slice. Any other kind panics.
//
// This is the convergence helper for control-input handlers (Reshape's shape,
// Slice's starts/ends/axes/steps, etc.) that must accept either concrete or
// symbolic shape vectors uniformly.
func (v Value) ToSints() []shape.Sint {
	switch v.Kind {
	case KindHostSints:
		return v.Ss
	case KindHostInts:
		out := make([]shape.Sint, len(v.Is))
		for i, x := range v.Is {
			out[i] = shape.Const(x)
		}
		return out
	case KindHostInt64:
		return []shape.Sint{shape.Const(v.I)}
	default:
		panic(fmt.Sprintf("onnx.Value.ToSints: kind=%d, want host-tier integer", v.Kind))
	}
}
