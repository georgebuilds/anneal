package backend

import "github.com/georgebuilds/anneal/schedule"

// Renderer turns a single schedule.ExecItem into a backend-specific kernel
// source (WGSL today; future backends may emit PTX, CUDA C, or SPIR-V).
//
// Threading: Render performs no GPU work. It is pure CPU and may be called from
// any goroutine.
//
// Renderer is a thin contract over the per-backend kernel rendering function in
// the codegen package: the backend's Renderer implementation forwards to e.g.
// codegen.RenderWGSL — the codegen package owns the rendering logic.
type Renderer interface {
	Render(item schedule.ExecItem) schedule.RenderResult
}
