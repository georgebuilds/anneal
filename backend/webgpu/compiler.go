package webgpu

import (
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/georgebuilds/anneal/backend"
)

// compiler is the WebGPU implementation of backend.Compiler. It owns the
// pipeline caches: Compile returns a cached *program when called with the same
// source string, otherwise it creates the shader module, bind-group layout,
// pipeline layout, and compute pipeline and caches the result.
//
// Caches are keyed by the exact rendered WGSL source string. No normalization
// is applied at the cache layer; instead, WGSL identifier stability in codegen
// (SPEC §7.7c, normalizeWGSL covers per-run-varying identifiers like t{N} /
// r{N} / sm{N}) guarantees that the same kernel structure produces byte-
// identical source across process restarts. The BEAM disk cache relies on the
// same contract; if a future codegen change introduces new per-run-varying
// identifiers, extend normalizeWGSL or both caches silently invalidate on
// restart.
//
// Static and symbolic kernels live in separate sub-caches (staticCache and
// symCache, keyed on KernelMeta.HasParamsUniform): this lets the dynbatch /
// compile-once tests assert SymbolicCount() == 0 after pure-static workloads
// without polluting the symbolic counter with concrete kernels.
//
// Threading: Compile touches *wgpu.* and must be called from the GPU-owner
// goroutine. The cache maps themselves are not synchronised (the GPU-owner
// goroutine is the only writer).
type compiler struct {
	dev         *Device
	staticCache map[string]*program // HasParamsUniform == false
	symCache    map[string]*program // HasParamsUniform == true
}

func newCompiler(dev *Device) *compiler {
	return &compiler{
		dev:         dev,
		staticCache: make(map[string]*program),
		symCache:    make(map[string]*program),
	}
}

// SymbolicCount returns the number of distinct symbolic-kernel programs that
// have been compiled and are still resident in the cache. Drives the
// SymCompiledCount() public observable used by compile-once regression tests.
func (c *compiler) SymbolicCount() int { return len(c.symCache) }

// Compile returns a Program for src. If a Program with this source already
// exists in the cache, the cached entry is returned (and the orchestrator
// must NOT release it — the cache owns the lifetime; Close drains the cache).
//
// meta.NumStorageBuffers is the number of storage-buffer bindings (PARAM 0
// is read_write; PARAM 1..N-1 are read-only). meta.HasParamsUniform adds a
// trailing uniform-buffer binding at slot NumStorageBuffers, used for the
// symbolic params_n payload.
func (c *compiler) Compile(src string, meta backend.KernelMeta) (backend.Program, error) {
	cache := c.staticCache
	if meta.HasParamsUniform {
		cache = c.symCache
	}
	if p, ok := cache[src]; ok {
		return p, nil
	}

	shader, err := c.dev.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: src})
	if err != nil {
		return nil, fmt.Errorf("CreateShaderModule: %w\n--- WGSL ---\n%s", err, src)
	}

	// Storage-buffer entries: PARAM(0) is read_write (the output), the rest
	// are read-only storage.
	total := meta.NumStorageBuffers
	if meta.HasParamsUniform {
		total++
	}
	layoutEntries := make([]gputypes.BindGroupLayoutEntry, total)
	for i := 0; i < meta.NumStorageBuffers; i++ {
		bt := gputypes.BufferBindingTypeReadOnlyStorage
		if i == 0 {
			bt = gputypes.BufferBindingTypeStorage
		}
		layoutEntries[i] = gputypes.BindGroupLayoutEntry{
			Binding:    uint32(i),
			Visibility: gputypes.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: bt},
		}
	}
	if meta.HasParamsUniform {
		layoutEntries[meta.NumStorageBuffers] = gputypes.BindGroupLayoutEntry{
			Binding:    uint32(meta.NumStorageBuffers),
			Visibility: gputypes.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform},
		}
	}

	bgLayout, err := c.dev.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: layoutEntries,
	})
	if err != nil {
		shader.Release()
		return nil, fmt.Errorf("CreateBindGroupLayout: %w", err)
	}

	pipelineLayout, err := c.dev.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("CreatePipelineLayout: %w", err)
	}

	pipeline, err := c.dev.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "main",
	})
	if err != nil {
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("CreateComputePipeline: %w\n--- WGSL ---\n%s", err, src)
	}

	p := &program{
		dev:              c.dev,
		shader:           shader,
		bgLayout:         bgLayout,
		pipelineLayout:   pipelineLayout,
		pipeline:         pipeline,
		numStorageBufs:   meta.NumStorageBuffers,
		hasParamsUniform: meta.HasParamsUniform,
	}
	cache[src] = p
	return p, nil
}

// releaseAll releases every cached program and clears both caches. Called from
// Device.Close.
func (c *compiler) releaseAll() {
	for _, p := range c.staticCache {
		p.releaseGPU()
	}
	for _, p := range c.symCache {
		p.releaseGPU()
	}
	c.staticCache = make(map[string]*program)
	c.symCache = make(map[string]*program)
}

// Compile-time assertion that *compiler satisfies backend.Compiler.
var _ backend.Compiler = (*compiler)(nil)
