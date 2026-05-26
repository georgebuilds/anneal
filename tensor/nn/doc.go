// Package nn provides minimal neural network layers built from tensor primitives.
// All layers build UOp graphs; no computation happens until tensor.Realize().
//
// Phase 6 (autodiff) differentiates w.r.t. Parameters; Phase 9 (optimizer) updates them.
package nn
