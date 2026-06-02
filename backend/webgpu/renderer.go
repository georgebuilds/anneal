package webgpu

import (
	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
)

// renderer is the WebGPU implementation of backend.Renderer. It is a thin
// shim over codegen.RenderWGSL — the codegen package owns the lowerer and
// WGSL emission; this type exists so the orchestrator can speak to a
// backend-agnostic Renderer interface.
//
// Threading: Render performs no GPU work. It is pure CPU and may be called
// from any goroutine.
type renderer struct{}

// Render lowers item to WGSL via codegen.RenderWGSL.
func (renderer) Render(item schedule.ExecItem) schedule.RenderResult {
	return codegen.RenderWGSL(item)
}

// Compile-time assertion that renderer satisfies backend.Renderer.
var _ backend.Renderer = renderer{}
