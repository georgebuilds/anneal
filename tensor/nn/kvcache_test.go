package nn_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/georgebuilds/anneal/tensor/nn"
	"github.com/georgebuilds/anneal/uop"
)

func TestNewKVCache_AllocatesPerLayer(t *testing.T) {
	c := nn.NewKVCache(3, 4, 8, 32)
	if c.NumLayers != 3 || c.NumHeads != 4 || c.HeadDim != 8 || c.MaxSeqLen != 32 {
		t.Fatalf("dims mismatch: got %+v", c)
	}
	if c.Pos != 0 {
		t.Fatalf("Pos: got %d want 0", c.Pos)
	}
	if len(c.Keys) != 3 || len(c.Values) != 3 {
		t.Fatalf("layer count: got K=%d V=%d", len(c.Keys), len(c.Values))
	}
	want := 4 * 32 * 8
	for i := 0; i < 3; i++ {
		if len(c.Keys[i]) != want || len(c.Values[i]) != want {
			t.Fatalf("layer %d size: got K=%d V=%d want %d", i, len(c.Keys[i]), len(c.Values[i]), want)
		}
		for j, v := range c.Keys[i] {
			if v != 0 {
				t.Fatalf("layer %d Keys[%d]: got %v want 0", i, j, v)
			}
		}
		for j, v := range c.Values[i] {
			if v != 0 {
				t.Fatalf("layer %d Values[%d]: got %v want 0", i, j, v)
			}
		}
	}
}

func TestNewKVCache_PanicsOnZeroOrNegativeDim(t *testing.T) {
	cases := []struct {
		name                                 string
		numLayers, numHeads, headDim, maxSeq int
	}{
		{"zero layers", 0, 4, 8, 32},
		{"neg layers", -1, 4, 8, 32},
		{"zero heads", 3, 0, 8, 32},
		{"zero dim", 3, 4, 0, 32},
		{"zero maxSeq", 3, 4, 8, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic, got none")
				}
			}()
			nn.NewKVCache(tc.numLayers, tc.numHeads, tc.headDim, tc.maxSeq)
		})
	}
}

func TestKVCache_StoreLayerKV_WritesAtCurrentSlot(t *testing.T) {
	c := nn.NewKVCache(2, 3, 4, 5) // 2 layers, 3 heads, 4 dim, 5 max
	// Build kNew, vNew: 3 heads * 4 dim = 12 floats. Head-major.
	kNew := []float32{
		1, 2, 3, 4, // head 0
		5, 6, 7, 8, // head 1
		9, 10, 11, 12, // head 2
	}
	vNew := []float32{
		-1, -2, -3, -4,
		-5, -6, -7, -8,
		-9, -10, -11, -12,
	}

	c.Pos = 2 // write into slot 2
	c.StoreLayerKV(1, kNew, vNew)

	// Layer-1 layout: [3 heads, 5 maxSeq, 4 dim] flat row-major.
	// Head h slot p starts at h*5*4 + p*4 = h*20 + 8.
	for h := 0; h < 3; h++ {
		baseSrc := h * 4
		baseDst := h*5*4 + 2*4
		for d := 0; d < 4; d++ {
			if c.Keys[1][baseDst+d] != kNew[baseSrc+d] {
				t.Fatalf("K[layer=1 head=%d dim=%d] got %v want %v", h, d, c.Keys[1][baseDst+d], kNew[baseSrc+d])
			}
			if c.Values[1][baseDst+d] != vNew[baseSrc+d] {
				t.Fatalf("V[layer=1 head=%d dim=%d] got %v want %v", h, d, c.Values[1][baseDst+d], vNew[baseSrc+d])
			}
		}
		// Other slots in this head should remain zero.
		for p := 0; p < 5; p++ {
			if p == 2 {
				continue
			}
			for d := 0; d < 4; d++ {
				idx := h*5*4 + p*4 + d
				if c.Keys[1][idx] != 0 || c.Values[1][idx] != 0 {
					t.Fatalf("layer=1 head=%d slot=%d dim=%d: expected zero", h, p, d)
				}
			}
		}
	}
	// Other layer should still be all-zero.
	for j, v := range c.Keys[0] {
		if v != 0 {
			t.Fatalf("layer 0 Keys[%d] should be 0 got %v", j, v)
		}
	}
}

func TestKVCache_StoreLayerKV_PanicsOnBadInputs(t *testing.T) {
	c := nn.NewKVCache(2, 3, 4, 5)
	good := make([]float32, 12)
	cases := []struct {
		name   string
		layer  int
		k, v   []float32
		setPos int
	}{
		{"layer too low", -1, good, good, 0},
		{"layer too high", 2, good, good, 0},
		{"bad k length", 0, make([]float32, 11), good, 0},
		{"bad v length", 0, good, make([]float32, 13), 0},
		{"pos too high", 0, good, good, 5},
		{"pos negative", 0, good, good, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c2 := nn.NewKVCache(2, 3, 4, 5)
			c2.Pos = tc.setPos
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic, got none")
				}
			}()
			c2.StoreLayerKV(tc.layer, tc.k, tc.v)
			_ = c // keep used
		})
	}
}

func TestKVCache_PosOneHotData(t *testing.T) {
	c := nn.NewKVCache(1, 1, 1, 5)
	c.Pos = 3
	got := c.PosOneHotData()
	want := []float32{0, 0, 0, 1, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	// At Pos==MaxSeqLen the buffer is all-zero (defensive).
	c.Pos = 5
	got = c.PosOneHotData()
	want = []float32{0, 0, 0, 0, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Pos==MaxSeqLen got %v want %v", got, want)
	}
}

func TestKVCache_LengthMaskData(t *testing.T) {
	c := nn.NewKVCache(1, 1, 1, 5)
	c.Pos = 0
	if got, want := c.LengthMaskData(), ([]float32{1, 0, 0, 0, 0}); !reflect.DeepEqual(got, want) {
		t.Fatalf("Pos=0 got %v want %v", got, want)
	}
	c.Pos = 2
	if got, want := c.LengthMaskData(), ([]float32{1, 1, 1, 0, 0}); !reflect.DeepEqual(got, want) {
		t.Fatalf("Pos=2 got %v want %v", got, want)
	}
	c.Pos = 4
	if got, want := c.LengthMaskData(), ([]float32{1, 1, 1, 1, 1}); !reflect.DeepEqual(got, want) {
		t.Fatalf("Pos=4 got %v want %v", got, want)
	}
	// Past end is clamped so we never write out of bounds.
	c.Pos = 7
	if got, want := c.LengthMaskData(), ([]float32{1, 1, 1, 1, 1}); !reflect.DeepEqual(got, want) {
		t.Fatalf("Pos=7 got %v want %v", got, want)
	}
}

func TestKVCache_Reset_ZeroesAndResetsPos(t *testing.T) {
	c := nn.NewKVCache(2, 2, 2, 3)
	for i := range c.Keys {
		for j := range c.Keys[i] {
			c.Keys[i][j] = float32(j + 1)
			c.Values[i][j] = float32(-(j + 1))
		}
	}
	c.Pos = 2
	c.Reset()
	if c.Pos != 0 {
		t.Fatalf("Pos: got %d want 0", c.Pos)
	}
	for i := range c.Keys {
		for j := range c.Keys[i] {
			if c.Keys[i][j] != 0 || c.Values[i][j] != 0 {
				t.Fatalf("layer %d offset %d not zero", i, j)
			}
		}
	}
}

func TestKVCache_Advance(t *testing.T) {
	c := nn.NewKVCache(1, 1, 1, 3)
	for i := 0; i < 3; i++ {
		if c.Pos != i {
			t.Fatalf("step %d: Pos got %d want %d", i, c.Pos, i)
		}
		c.Advance()
	}
	if c.Pos != 3 {
		t.Fatalf("after 3 advances Pos got %d want 3", c.Pos)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic at MaxSeqLen, got none")
		}
		if !strings.Contains(fmtPanic(r), "MaxSeqLen") {
			t.Fatalf("panic message missing MaxSeqLen: %v", r)
		}
	}()
	c.Advance()
}

func TestKVCache_UploadKLeaf_Shape(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(2, 3, 4, 5)
	// Seed layer 1 so we can confirm SetData round-trips.
	for i := range c.Keys[1] {
		c.Keys[1][i] = float32(i)
	}
	leaf := c.UploadKLeaf(a, 1, uop.Dtypes.Float32, "test")
	sh := leaf.Shape()
	want := []int64{1, 3, 5, 4}
	if !reflect.DeepEqual(sh, want) {
		t.Fatalf("shape got %v want %v", sh, want)
	}
	if leaf.DType() != uop.Dtypes.Float32 {
		t.Fatalf("dtype: got %s want Float32", leaf.DType())
	}
	got := leaf.Data()
	if len(got) != len(c.Keys[1]) {
		t.Fatalf("data len: got %d want %d", len(got), len(c.Keys[1]))
	}
	for i := range got {
		if got[i] != c.Keys[1][i] {
			t.Fatalf("data[%d]: got %v want %v", i, got[i], c.Keys[1][i])
		}
	}
}

func TestKVCache_UploadVLeaf_Shape(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(1, 2, 3, 4)
	for i := range c.Values[0] {
		c.Values[0][i] = -float32(i)
	}
	leaf := c.UploadVLeaf(a, 0, uop.Dtypes.Float32, "test")
	sh := leaf.Shape()
	want := []int64{1, 2, 4, 3}
	if !reflect.DeepEqual(sh, want) {
		t.Fatalf("shape got %v want %v", sh, want)
	}
	got := leaf.Data()
	for i := range got {
		if got[i] != c.Values[0][i] {
			t.Fatalf("data[%d]: got %v want %v", i, got[i], c.Values[0][i])
		}
	}
}

func TestKVCache_UploadKLeaf_PanicOnBadLayer(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(2, 1, 1, 1)
	for _, layer := range []int{-1, 2, 5} {
		t.Run("layer", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("expected panic on layer %d", layer)
				}
			}()
			c.UploadKLeaf(a, layer, uop.Dtypes.Float32, "test")
		})
	}
}

func TestKVCache_UploadVLeaf_PanicOnBadLayer(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(2, 1, 1, 1)
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic on bad layer")
		}
	}()
	c.UploadVLeaf(a, 5, uop.Dtypes.Float32, "test")
}

func TestKVCache_UploadPosOneHotLeaf_Shape(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(1, 1, 1, 6)
	c.Pos = 4
	leaf := c.UploadPosOneHotLeaf(a, uop.Dtypes.Float32, "test")
	sh := leaf.Shape()
	want := []int64{1, 1, 6, 1}
	if !reflect.DeepEqual(sh, want) {
		t.Fatalf("shape got %v want %v", sh, want)
	}
	got := leaf.Data()
	wantData := []float32{0, 0, 0, 0, 1, 0}
	if !reflect.DeepEqual(got, wantData) {
		t.Fatalf("data got %v want %v", got, wantData)
	}
}

func TestKVCache_UploadLengthMaskLeaf_Shape(t *testing.T) {
	a := uop.NewArena(1 << 16)
	c := nn.NewKVCache(1, 1, 1, 4)
	c.Pos = 1
	leaf := c.UploadLengthMaskLeaf(a, uop.Dtypes.Float32, "test")
	sh := leaf.Shape()
	want := []int64{1, 1, 1, 4}
	if !reflect.DeepEqual(sh, want) {
		t.Fatalf("shape got %v want %v", sh, want)
	}
	got := leaf.Data()
	wantData := []float32{1, 1, 0, 0}
	if !reflect.DeepEqual(got, wantData) {
		t.Fatalf("data got %v want %v", got, wantData)
	}
}

func fmtPanic(r any) string {
	switch v := r.(type) {
	case string:
		return v
	case error:
		return v.Error()
	default:
		return ""
	}
}
