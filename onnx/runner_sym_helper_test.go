package onnx

import (
	"github.com/georgebuilds/anneal/shape"
)

// symInt narrows a shape.Sint to shape.SymInt, exposing the underlying UOp.
// Panics if s is not symbolic - caller must check ConstValue() first.
func symInt(s any) shape.SymInt {
	return s.(shape.SymInt)
}
