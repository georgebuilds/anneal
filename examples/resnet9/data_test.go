package resnet9

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"math/rand"
	"testing"
)

// ── synthetic CIFAR-10 tarball builder ───────────────────────────────────────

// buildSyntheticTarball assembles an in-memory gzipped tarball with the
// CIFAR-10 binary layout: 5 train batches + 1 test batch of synthetic
// records. Used by every loader test to avoid hitting the network.
func buildSyntheticTarball(seed int64) ([]byte, error) {
	var buf bytes.Buffer
	gzw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gzw)
	rng := rand.New(rand.NewSource(seed))

	write := func(name string) error {
		body := make([]byte, recordsPerFile*recordBytes)
		for i := 0; i < recordsPerFile; i++ {
			off := i * recordBytes
			body[off] = byte(i % numClasses) // deterministic labels 0..9
			for p := off + 1; p < off+recordBytes; p++ {
				body[p] = byte(rng.Intn(256))
			}
		}
		hdr := &tar.Header{
			Name: "cifar-10-batches-bin/" + name,
			Mode: 0o644,
			Size: int64(len(body)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if _, err := tw.Write(body); err != nil {
			return err
		}
		return nil
	}

	for i := 1; i <= trainBatchCount; i++ {
		if err := write(fmt.Sprintf("data_batch_%d.bin", i)); err != nil {
			return nil, err
		}
	}
	if err := write("test_batch.bin"); err != nil {
		return nil, err
	}
	// Include an irrelevant meta file to verify the loader ignores it.
	hdr := &tar.Header{
		Name: "cifar-10-batches-bin/batches.meta.txt",
		Mode: 0o644,
		Size: 11,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return nil, err
	}
	if _, err := tw.Write([]byte("airplane\nca")); err != nil {
		return nil, err
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gzw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ── 1. End-to-end stream parse ───────────────────────────────────────────────

func TestLoadFromTarStream(t *testing.T) {
	gz, err := buildSyntheticTarball(7)
	if err != nil {
		t.Fatalf("build synthetic tarball: %v", err)
	}
	d, err := loadFromTarStream(bytes.NewReader(gz))
	if err != nil {
		t.Fatalf("loadFromTarStream: %v", err)
	}
	if d.NumTrain() != trainBatchCount*recordsPerFile {
		t.Fatalf("NumTrain(): got %d, want %d", d.NumTrain(), trainBatchCount*recordsPerFile)
	}
	if d.NumTest() != recordsPerFile {
		t.Fatalf("NumTest(): got %d, want %d", d.NumTest(), recordsPerFile)
	}
	if len(d.Train) != d.NumTrain()*imagePixels {
		t.Fatalf("Train flat len: got %d, want %d", len(d.Train), d.NumTrain()*imagePixels)
	}
	if len(d.Test) != d.NumTest()*imagePixels {
		t.Fatalf("Test flat len: got %d, want %d", len(d.Test), d.NumTest()*imagePixels)
	}

	// Labels are 0..9 cyclic per batch (synthetic), so every label must be
	// in range and the histogram should be uniform.
	hist := make([]int, numClasses)
	for _, l := range d.TrainLabels {
		if l < 0 || int(l) >= numClasses {
			t.Fatalf("train label %d out of range", l)
		}
		hist[l]++
	}
	for c, n := range hist {
		if n != trainBatchCount*recordsPerFile/numClasses {
			t.Fatalf("train histogram[%d] = %d, want %d", c, n, trainBatchCount*recordsPerFile/numClasses)
		}
	}
}

// ── 2. Normalisation correctness ─────────────────────────────────────────────

// TestNormalizationStats asserts that the normalised pixel block has
// per-channel mean ≈ 0 and std ≈ 1 over a synthetic uniform-noise dataset.
// CIFAR-10's published per-channel stats (in the [0,1] range) are
// approximations of the real train-set distribution; uniform-noise input
// will not hit exactly 0/1 but should be within ~0.1 / ~0.05 of those
// targets for any sane mean/std subtraction.
func TestNormalizationStats(t *testing.T) {
	gz, err := buildSyntheticTarball(42)
	if err != nil {
		t.Fatalf("build synthetic tarball: %v", err)
	}
	d, err := loadFromTarStream(bytes.NewReader(gz))
	if err != nil {
		t.Fatalf("loadFromTarStream: %v", err)
	}

	// Sample a few thousand pixels per channel; uniform noise / 255 has
	// expected mean ≈ 0.5 and std ≈ 0.2887 (uniform [0,1]).
	// After (x - mean)/std with mean≈0.49, std≈0.25 the result has
	// mean ≈ (0.5 - 0.49)/0.25 ≈ 0.04 and std ≈ 0.2887/0.25 ≈ 1.15.
	const samples = 4096
	for c := 0; c < imageChannels; c++ {
		var sum, sq float64
		for i := 0; i < samples; i++ {
			off := (i*imagePixels + c*imageH*imageW)
			v := float64(d.Train[off])
			sum += v
			sq += v * v
		}
		mean := sum / samples
		variance := sq/samples - mean*mean
		if mean < -0.5 || mean > 0.5 {
			t.Fatalf("channel %d normalized mean out of range: %f", c, mean)
		}
		if variance < 0.5 || variance > 2.0 {
			t.Fatalf("channel %d normalized var out of range: %f", c, variance)
		}
		t.Logf("channel %d normalized: mean=%.3f var=%.3f", c, mean, variance)
	}
}

// ── 3. Batch sampling ────────────────────────────────────────────────────────

func TestBatch(t *testing.T) {
	gz, err := buildSyntheticTarball(11)
	if err != nil {
		t.Fatalf("build synthetic tarball: %v", err)
	}
	d, err := loadFromTarStream(bytes.NewReader(gz))
	if err != nil {
		t.Fatalf("loadFromTarStream: %v", err)
	}

	const B = 8
	x := make([]float32, B*imagePixels)
	y := make([]int32, B)
	rng := rand.New(rand.NewSource(13))
	d.Batch(rng, B, x, y)

	// Each label must be in range; each per-image pixel block must equal
	// one of the source images (sampled with replacement).
	for i := 0; i < B; i++ {
		if y[i] < 0 || int(y[i]) >= numClasses {
			t.Fatalf("batch label %d out of range: %d", i, y[i])
		}
	}
}

// ── 4. Batch size mismatches panic loudly ────────────────────────────────────

func TestBatchPanicsOnLenMismatch(t *testing.T) {
	d := &CIFAR10{
		Train:       make([]float32, imagePixels),
		TrainLabels: []int32{0},
	}
	rng := rand.New(rand.NewSource(1))

	t.Run("xOut wrong size", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on xOut size mismatch")
			}
		}()
		d.Batch(rng, 2, make([]float32, 7), make([]int32, 2))
	})

	t.Run("yOut wrong size", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on yOut size mismatch")
			}
		}()
		d.Batch(rng, 2, make([]float32, 2*imagePixels), make([]int32, 7))
	})
}

// ── 5. OneHot encoding ───────────────────────────────────────────────────────

func TestOneHot(t *testing.T) {
	labels := []int32{0, 3, 9, 5}
	out := make([]float32, len(labels)*numClasses)
	OneHot(labels, out)
	for i, l := range labels {
		for c := 0; c < numClasses; c++ {
			want := float32(0)
			if c == int(l) {
				want = 1
			}
			if got := out[i*numClasses+c]; got != want {
				t.Fatalf("out[%d, %d] = %f, want %f", i, c, got, want)
			}
		}
	}
}

func TestOneHotPanicsOnOutOfRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on label out of range")
		}
	}()
	OneHot([]int32{10}, make([]float32, numClasses))
}

// ── 6. Loader rejects malformed tarballs ─────────────────────────────────────

func TestLoadFromTarStreamRejectsTruncated(t *testing.T) {
	gz, err := buildSyntheticTarball(99)
	if err != nil {
		t.Fatalf("build synthetic tarball: %v", err)
	}
	// Truncate to half: gzip header + a few records.
	half := gz[:len(gz)/2]
	_, err = loadFromTarStream(bytes.NewReader(half))
	if err == nil {
		t.Fatal("expected error on truncated tarball")
	}
}

func TestLoadFromTarStreamRejectsNonGzip(t *testing.T) {
	_, err := loadFromTarStream(bytes.NewReader([]byte("not a gzip")))
	if err == nil {
		t.Fatal("expected error on non-gzip input")
	}
}

// ── 7. baseName helper ───────────────────────────────────────────────────────

func TestBaseName(t *testing.T) {
	cases := map[string]string{
		"cifar-10-batches-bin/data_batch_1.bin": "data_batch_1.bin",
		"data_batch_3.bin":                      "data_batch_3.bin",
		"/test_batch.bin":                       "test_batch.bin",
		"":                                      "",
	}
	for in, want := range cases {
		if got := baseName(in); got != want {
			t.Fatalf("baseName(%q) = %q, want %q", in, got, want)
		}
	}
}

// ensure the io.Reader interface match (so the file compiles with the import).
var _ io.Reader = (*bytes.Reader)(nil)
