package resnet9

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"strings"
	"testing"
)

// buildCustomTarball assembles a gzipped CIFAR-10-shaped tarball from an
// explicit list of (name, byteLen) entries. Bodies are zero-filled (label 0,
// black pixels) which is sufficient for the structural error-path tests that
// only care about record counts and truncation, not pixel values.
func buildCustomTarball(t *testing.T, entries []struct {
	name string
	size int
}) []byte {
	t.Helper()
	var buf bytes.Buffer
	gzw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gzw)
	for _, e := range entries {
		body := make([]byte, e.size)
		hdr := &tar.Header{Name: "cifar-10-batches-bin/" + e.name, Mode: 0o644, Size: int64(len(body))}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("write header %s: %v", e.name, err)
		}
		if _, err := tw.Write(body); err != nil {
			t.Fatalf("write body %s: %v", e.name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gzw.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return buf.Bytes()
}

const fullBatch = recordsPerFile * recordBytes

// TestLoadFromTarStream_TrainBatchTruncated covers the readBatch-error branch
// for a data_batch_*.bin file whose body is shorter than one full batch.
func TestLoadFromTarStream_TrainBatchTruncated(t *testing.T) {
	gz := buildCustomTarball(t, []struct {
		name string
		size int
	}{
		{"data_batch_1.bin", fullBatch - 100}, // short read inside readBatch
	})
	_, err := loadFromTarStream(bytes.NewReader(gz))
	if err == nil || !strings.Contains(err.Error(), "read data_batch_1.bin") {
		t.Fatalf("truncated train batch: got %v, want read-error", err)
	}
}

// TestLoadFromTarStream_TestBatchTruncated covers the readBatch-error branch
// for the test_batch.bin file. We supply 5 full train batches first so the
// loader reaches the test-batch case before failing.
func TestLoadFromTarStream_TestBatchTruncated(t *testing.T) {
	entries := []struct {
		name string
		size int
	}{}
	for i := 1; i <= trainBatchCount; i++ {
		entries = append(entries, struct {
			name string
			size int
		}{fmt.Sprintf("data_batch_%d.bin", i), fullBatch})
	}
	entries = append(entries, struct {
		name string
		size int
	}{"test_batch.bin", fullBatch - 50}) // short test batch
	gz := buildCustomTarball(t, entries)
	_, err := loadFromTarStream(bytes.NewReader(gz))
	if err == nil || !strings.Contains(err.Error(), "read test_batch.bin") {
		t.Fatalf("truncated test batch: got %v, want read-error", err)
	}
}

// TestLoadFromTarStream_TrainSizeMismatch covers the train-set-size guard:
// fewer than 5 train batches yields a short train set, no test batch needed
// because the train guard fires first.
func TestLoadFromTarStream_TrainSizeMismatch(t *testing.T) {
	gz := buildCustomTarball(t, []struct {
		name string
		size int
	}{
		{"data_batch_1.bin", fullBatch}, // only 1 of 5 train batches
	})
	_, err := loadFromTarStream(bytes.NewReader(gz))
	if err == nil || !strings.Contains(err.Error(), "train set size") {
		t.Fatalf("missing train batches: got %v, want train-size error", err)
	}
}

// TestLoadFromTarStream_TestSizeMismatch covers the test-set-size guard: all 5
// train batches present and correctly sized, but no test_batch.bin, so the
// test set is empty.
func TestLoadFromTarStream_TestSizeMismatch(t *testing.T) {
	entries := []struct {
		name string
		size int
	}{}
	for i := 1; i <= trainBatchCount; i++ {
		entries = append(entries, struct {
			name string
			size int
		}{fmt.Sprintf("data_batch_%d.bin", i), fullBatch})
	}
	// no test_batch.bin
	gz := buildCustomTarball(t, entries)
	_, err := loadFromTarStream(bytes.NewReader(gz))
	if err == nil || !strings.Contains(err.Error(), "test set size") {
		t.Fatalf("missing test batch: got %v, want test-size error", err)
	}
}

// TestLoadOfflineMissingAsset covers Load's error branch: with an empty cache
// dir and ANNEAL_OFFLINE=1, assets.Get fails and Load wraps the error.
func TestLoadOfflineMissingAsset(t *testing.T) {
	t.Setenv("ANNEAL_CACHE_DIR", t.TempDir())
	t.Setenv("ANNEAL_OFFLINE", "1")
	_, err := Load()
	if err == nil {
		t.Fatal("Load should error when CIFAR-10 tarball is uncached + offline")
	}
	if !strings.Contains(err.Error(), "fetch CIFAR-10 tarball") {
		t.Errorf("Load error = %v, want fetch-tarball wrapper", err)
	}
}
