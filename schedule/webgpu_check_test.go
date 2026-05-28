package schedule_test

import (
	"testing"
	"github.com/georgebuilds/anneal/backend/webgpu"
)

func TestWebGPUAvailable(t *testing.T) {
	dev, err := webgpu.Open()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer dev.Close()
	t.Logf("WebGPU available: %s", dev.AdapterName())
}
