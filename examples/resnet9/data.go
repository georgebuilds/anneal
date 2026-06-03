// Package resnet9 packages the ResNet-9 example: CIFAR-10 host pipeline,
// training loop, and the anneal CLI entry point.
package resnet9

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strings"

	"github.com/georgebuilds/anneal/internal/assets"
)

// CIFAR-10 binary format (Krizhevsky 2009):
//
//	tarball cifar-10-binary.tar.gz   ~170 MB, gzipped tar
//	  cifar-10-batches-bin/
//	    data_batch_1.bin .. data_batch_5.bin  (10000 records each)
//	    test_batch.bin                        (10000 records)
//	    batches.meta.txt                      (10 class names)
//
// Each record is 3073 bytes: 1 byte label (0-9) + 3072 bytes pixel data
// laid out as 1024 R, 1024 G, 1024 B (each plane row-major over 32x32).
//
// Train set: 5 * 10000 = 50000 images. Test set: 10000.
const (
	numClasses      = 10
	imageH          = 32
	imageW          = 32
	imageChannels   = 3
	imagePixels     = imageH * imageW * imageChannels // 3072
	recordBytes     = imagePixels + 1                 // 3073
	recordsPerFile  = 10000
	trainBatchCount = 5
	cifarAssetName  = "cifar10-binary"
)

// CIFAR10 holds the full dataset in host memory. Train is 50000 images x
// 3072 pixels, normalized to mean 0 / std 1 per channel (CIFAR-10 standard
// preprocessing). TrainLabels and TestLabels are dense int32 0..9.
type CIFAR10 struct {
	Train       []float32 // [50000, 3, 32, 32] flat row-major
	TrainLabels []int32   // [50000]
	Test        []float32 // [10000, 3, 32, 32] flat row-major
	TestLabels  []int32   // [10000]
}

// NumTrain / NumTest are the dataset sizes after a successful Load.
func (d *CIFAR10) NumTrain() int { return len(d.TrainLabels) }
func (d *CIFAR10) NumTest() int  { return len(d.TestLabels) }

// canonical per-channel statistics on the CIFAR-10 train set, scaled to
// the [0, 1] pixel range. These match the values used by every PyTorch /
// JAX reference for ResNet on CIFAR-10.
var (
	cifarMean = [3]float32{0.4914, 0.4822, 0.4465}
	cifarStd  = [3]float32{0.2470, 0.2435, 0.2616}
)

// Load returns the CIFAR-10 dataset, fetching the binary tarball into the
// per-user cache on first call (SHA-verified) and streaming the records
// through gzip+tar without an intermediate extract step. Subsequent calls
// hit the cached tarball.
//
// When ANNEAL_OFFLINE=1 is set and the tarball is not cached, Load returns
// an error directing the caller to fetch manually.
func Load() (*CIFAR10, error) {
	path, err := assets.Get(cifarAssetName)
	if err != nil {
		return nil, fmt.Errorf("resnet9: fetch CIFAR-10 tarball: %w", err)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("resnet9: open tarball %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	return loadFromTarStream(f)
}

// loadFromTarStream reads a gzipped tar stream containing the CIFAR-10
// binary batch files and returns the assembled CIFAR10 struct. Factored
// out so tests can feed synthetic streams without touching the network.
func loadFromTarStream(r io.Reader) (*CIFAR10, error) {
	gzr, err := gzip.NewReader(r)
	if err != nil {
		return nil, fmt.Errorf("resnet9: gzip: %w", err)
	}
	defer func() { _ = gzr.Close() }()
	tr := tar.NewReader(gzr)

	d := &CIFAR10{
		Train:       make([]float32, 0, trainBatchCount*recordsPerFile*imagePixels),
		TrainLabels: make([]int32, 0, trainBatchCount*recordsPerFile),
		Test:        nil,
		TestLabels:  nil,
	}

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("resnet9: tar: %w", err)
		}
		base := baseName(hdr.Name)
		switch {
		case strings.HasPrefix(base, "data_batch_") && strings.HasSuffix(base, ".bin"):
			x, y, err := readBatch(tr)
			if err != nil {
				return nil, fmt.Errorf("resnet9: read %s: %w", base, err)
			}
			d.Train = append(d.Train, x...)
			d.TrainLabels = append(d.TrainLabels, y...)
		case base == "test_batch.bin":
			x, y, err := readBatch(tr)
			if err != nil {
				return nil, fmt.Errorf("resnet9: read %s: %w", base, err)
			}
			d.Test = x
			d.TestLabels = y
		default:
			// Skip non-data files (meta, readme).
		}
	}

	if len(d.TrainLabels) != trainBatchCount*recordsPerFile {
		return nil, fmt.Errorf("resnet9: train set size %d != %d", len(d.TrainLabels), trainBatchCount*recordsPerFile)
	}
	if len(d.TestLabels) != recordsPerFile {
		return nil, fmt.Errorf("resnet9: test set size %d != %d", len(d.TestLabels), recordsPerFile)
	}
	return d, nil
}

// readBatch consumes one CIFAR-10 batch file (10000 records, 30730000 bytes)
// from r and returns the normalized pixel block + dense int32 labels.
func readBatch(r io.Reader) ([]float32, []int32, error) {
	const total = recordsPerFile * recordBytes
	raw := make([]byte, total)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, nil, fmt.Errorf("read %d bytes: %w", total, err)
	}
	x := make([]float32, recordsPerFile*imagePixels)
	y := make([]int32, recordsPerFile)
	for i := 0; i < recordsPerFile; i++ {
		off := i * recordBytes
		y[i] = int32(raw[off])
		px := raw[off+1 : off+recordBytes]
		// CIFAR layout per record: 1024 R, 1024 G, 1024 B (each plane
		// row-major over H*W). Anneal expects [C, H, W] flat: that's
		// exactly the same layout, so the order is direct.
		outBase := i * imagePixels
		for c := 0; c < imageChannels; c++ {
			mean := cifarMean[c]
			std := cifarStd[c]
			planeBase := c * imageH * imageW
			for p := 0; p < imageH*imageW; p++ {
				v := float32(px[planeBase+p]) / 255.0
				x[outBase+planeBase+p] = (v - mean) / std
			}
		}
	}
	return x, y, nil
}

// baseName returns the final path component of a tar header name, handling
// both "cifar-10-batches-bin/data_batch_1.bin" and "data_batch_1.bin".
func baseName(name string) string {
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		return name[i+1:]
	}
	return name
}

// Batch fills xOut and yOut with batchSize records sampled uniformly with
// replacement from the train set. The caller owns the slices; they should
// be sized batchSize*imagePixels and batchSize respectively.
//
// xOut is laid out [B, C, H, W] flat row-major (3072 floats per batch
// element); yOut is the dense int32 label.
func (d *CIFAR10) Batch(rng *rand.Rand, batchSize int, xOut []float32, yOut []int32) {
	if len(xOut) != batchSize*imagePixels {
		panic(fmt.Sprintf("resnet9: Batch: xOut len %d != %d", len(xOut), batchSize*imagePixels))
	}
	if len(yOut) != batchSize {
		panic(fmt.Sprintf("resnet9: Batch: yOut len %d != %d", len(yOut), batchSize))
	}
	n := d.NumTrain()
	for b := 0; b < batchSize; b++ {
		idx := rng.Intn(n)
		copy(xOut[b*imagePixels:(b+1)*imagePixels], d.Train[idx*imagePixels:(idx+1)*imagePixels])
		yOut[b] = d.TrainLabels[idx]
	}
}

// OneHot writes a [batchSize, numClasses] one-hot encoding of labels into
// out. Caller owns out; it should be batchSize*numClasses floats.
func OneHot(labels []int32, out []float32) {
	for i := range out {
		out[i] = 0
	}
	for i, l := range labels {
		if l < 0 || int(l) >= numClasses {
			panic(fmt.Sprintf("resnet9: OneHot: label %d out of range [0,%d)", l, numClasses))
		}
		out[i*numClasses+int(l)] = 1
	}
}
