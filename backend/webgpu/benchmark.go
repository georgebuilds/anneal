package webgpu

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/georgebuilds/anneal/backend"
	"github.com/georgebuilds/anneal/codegen"
	"github.com/georgebuilds/anneal/schedule"
	"github.com/georgebuilds/anneal/uop"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

// Benchmark implements backend.Benchmarker.
func (d *Device) Benchmark(item schedule.ExecItem, warmup, iterations int) (backend.BenchmarkResult, error) {
	var res backend.BenchmarkResult
	err := d.onGPU(func() error {
		var berr error
		res, berr = d.benchmarkLocked(item, warmup, iterations)
		return berr
	})
	return res, err
}

func (d *Device) benchmarkLocked(item schedule.ExecItem, warmup, iterations int) (backend.BenchmarkResult, error) {
	if iterations <= 0 {
		return backend.BenchmarkResult{}, fmt.Errorf("benchmark: iterations must be > 0")
	}

	// ── Render WGSL ───────────────────────────────────────────────────────
	var ws, wc [3]int
	wgsl := item.WGSL
	if wgsl == "" {
		res := codegen.RenderWGSL(item)
		wgsl = res.WGSL
		ws = res.LocalSize
		wc = res.WorkgroupCount
		item.WGSL = wgsl
		item.LocalSize = ws
		item.WorkgroupCount = wc
	} else {
		ws = item.LocalSize
		wc = item.WorkgroupCount
	}

	// ── Get or compile pipeline ───────────────────────────────────────────
	pipe, ok := d.pipelineCache[wgsl]
	if !ok {
		var err error
		pipe, err = d.compilePipelineLocked(item, wgsl)
		if err != nil {
			return backend.BenchmarkResult{}, err
		}
		d.pipelineCache[wgsl] = pipe
	}

	// ── Allocate temporary buffers ────────────────────────────────────────
	gpuBufs := make([]*wgpu.Buffer, len(item.Bufs))
	for i, buf := range item.Bufs {
		usage := gputypes.BufferUsageStorage
		if i == 0 {
			// Output buffer might need CopySrc if we ever want to read it back,
			// but for timing just Storage is enough.
		}
		gb, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("bench_buf_%d", i),
			Usage: usage,
			Size:  uint64(buf.Size) * elemBytes(buf.DType),
		})
		if err != nil {
			// Cleanup previously allocated buffers
			for j := 0; j < i; j++ {
				gpuBufs[j].Release()
			}
			return backend.BenchmarkResult{}, fmt.Errorf("benchmark: CreateBuffer %d: %w", i, err)
		}
		gpuBufs[i] = gb
	}
	defer func() {
		for _, gb := range gpuBufs {
			gb.Release()
		}
	}()

	// ── Create Bind Group ─────────────────────────────────────────────────
	entries := make([]wgpu.BindGroupEntry, len(item.Bufs))
	for i, gb := range gpuBufs {
		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  gb,
			Size:    uint64(item.Bufs[i].Size) * elemBytes(item.Bufs[i].DType),
		}
	}
	bg, err := d.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  pipe.bgLayout,
		Entries: entries,
	})
	if err != nil {
		return backend.BenchmarkResult{}, fmt.Errorf("benchmark: CreateBindGroup: %w", err)
	}
	defer bg.Release()

	// ── Prepare Command Buffer ────────────────────────────────────────────
	createCmd := func() (*wgpu.CommandBuffer, error) {
		enc, err := d.device.CreateCommandEncoder(nil)
		if err != nil {
			return nil, err
		}
		pass, err := enc.BeginComputePass(nil)
		if err != nil {
			return nil, err
		}
		pass.SetPipeline(pipe.pipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.Dispatch(uint32(wc[0]), uint32(wc[1]), uint32(wc[2]))
		if err := pass.End(); err != nil {
			return nil, err
		}
		cmd, err := enc.Finish()
		if err != nil {
			return nil, err
		}
		return cmd, nil
	}

	// ── Warmup ────────────────────────────────────────────────────────────
	for i := 0; i < warmup; i++ {
		cmd, err := createCmd()
		if err != nil {
			return backend.BenchmarkResult{}, fmt.Errorf("benchmark: warmup createCmd %d: %w", i, err)
		}
		if _, err := d.queue.Submit(cmd); err != nil {
			cmd.Release()
			return backend.BenchmarkResult{}, fmt.Errorf("benchmark: warmup submit %d: %w", i, err)
		}
		d.device.Poll(wgpu.PollWait)
		cmd.Release()
	}

	// ── Iterations ────────────────────────────────────────────────────────
	cmds := make([]*wgpu.CommandBuffer, iterations)
	for i := 0; i < iterations; i++ {
		cmd, err := createCmd()
		if err != nil {
			for j := 0; j < i; j++ {
				cmds[j].Release()
			}
			return backend.BenchmarkResult{}, fmt.Errorf("benchmark: iteration createCmd %d: %w", i, err)
		}
		cmds[i] = cmd
	}
	defer func() {
		for _, cmd := range cmds {
			cmd.Release()
		}
	}()

	samples := make([]float64, iterations)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		if _, err := d.queue.Submit(cmds[i]); err != nil {
			return backend.BenchmarkResult{}, fmt.Errorf("benchmark: iteration submit %d: %w", i, err)
		}
		d.device.Poll(wgpu.PollWait)
		samples[i] = float64(time.Since(start).Nanoseconds()) / 1000.0
	}

	// ── Statistics ────────────────────────────────────────────────────────
	sort.Float64s(samples)
	min := samples[0]
	max := samples[len(samples)-1]

	var sum float64
	for _, s := range samples {
		sum += s
	}
	mean := sum / float64(len(samples))

	var sumSqDiff float64
	for _, s := range samples {
		diff := s - mean
		sumSqDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSqDiff / float64(len(samples)))
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}

	var median float64
	if len(samples)%2 == 0 {
		median = (samples[len(samples)/2-1] + samples[len(samples)/2]) / 2
	} else {
		median = samples[len(samples)/2]
	}

	return backend.BenchmarkResult{
		MinMicros:    min,
		MedianMicros: median,
		MaxMicros:    max,
		MeanMicros:   mean,
		StdDevMicros: stdDev,
		CV:           cv,
	}, nil
}

func (d *Device) compilePipelineLocked(item schedule.ExecItem, wgsl string) (*kernelPipeline, error) {
	var nParams int
	if item.Ast.Valid() {
		nParams = item.Ast.Arg().(uop.KernelInfo).NumParams
	} else {
		nParams = len(item.Bufs)
	}

	shader, err := d.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSL: wgsl})
	if err != nil {
		return nil, fmt.Errorf("benchmark: CreateShaderModule: %w", err)
	}

	layoutEntries := make([]gputypes.BindGroupLayoutEntry, nParams)
	for i := range layoutEntries {
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
	bgLayout, err := d.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: layoutEntries,
	})
	if err != nil {
		shader.Release()
		return nil, fmt.Errorf("benchmark: CreateBindGroupLayout: %w", err)
	}

	pipelineLayout, err := d.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgLayout},
	})
	if err != nil {
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("benchmark: CreatePipelineLayout: %w", err)
	}

	pipeline, err := d.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "main",
	})
	if err != nil {
		pipelineLayout.Release()
		bgLayout.Release()
		shader.Release()
		return nil, fmt.Errorf("benchmark: CreateComputePipeline: %w", err)
	}

	return &kernelPipeline{
		shader:         shader,
		bgLayout:       bgLayout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
	}, nil
}
