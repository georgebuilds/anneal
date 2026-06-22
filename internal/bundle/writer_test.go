package bundle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewWriterCreatesBundleDirectory(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "mlp", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer func() { _ = w.Close() }()

	if w.BundleID() == "" {
		t.Fatalf("BundleID is empty")
	}
	if !IsValidBundleID(string(w.BundleID())) {
		t.Fatalf("BundleID %q is not a valid bundle id", w.BundleID())
	}
	if !strings.HasPrefix(w.Path(), root) {
		t.Fatalf("Path %q not under root %q", w.Path(), root)
	}

	// Directory and kernels/ subdir must exist.
	if st, err := os.Stat(w.Path()); err != nil || !st.IsDir() {
		t.Fatalf("bundle dir missing or not a dir: %v", err)
	}
	if st, err := os.Stat(filepath.Join(w.Path(), "kernels")); err != nil || !st.IsDir() {
		t.Fatalf("kernels dir missing: %v", err)
	}
	// Stub manifest must be present with bundle_version=1.
	mb, err := os.ReadFile(filepath.Join(w.Path(), "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var stub Manifest
	if err := json.Unmarshal(mb, &stub); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if stub.BundleVersion != 1 {
		t.Errorf("stub bundle_version = %d, want 1", stub.BundleVersion)
	}
	if stub.Kind != KindTrain {
		t.Errorf("stub kind = %v, want train", stub.Kind)
	}
	if stub.Model != "mlp" {
		t.Errorf("stub model = %q, want mlp", stub.Model)
	}
	if stub.CreatedAt.IsZero() {
		t.Errorf("stub created_at is zero")
	}
}

func TestNewWriterSanitizesModelName(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "../My Model/v2", KindSaved)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer func() { _ = w.Close() }()
	if strings.Contains(string(w.BundleID()), "..") {
		t.Errorf("BundleID contains '..': %q", w.BundleID())
	}
	if !IsValidBundleID(string(w.BundleID())) {
		t.Errorf("BundleID %q is not valid after sanitization", w.BundleID())
	}
}

func TestNewWriterRejectsEmpty(t *testing.T) {
	if _, err := NewWriter("", "m", KindTrain); err == nil {
		t.Errorf("NewWriter with empty root: want error")
	}
	if _, err := NewWriter(t.TempDir(), "", KindTrain); err == nil {
		t.Errorf("NewWriter with empty model: want error")
	}
}

func TestWriterRoundTrip(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "conv", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	// Provenance
	binds := map[string]int64{"B": 16, "T": 128}
	if err := w.SetProvenance("0.0.0-dev", "abc123", "Apple M3", "Metal",
		"deadbeef", binds); err != nil {
		t.Fatalf("SetProvenance: %v", err)
	}

	// Config
	cfg := Config{
		Model:  "conv",
		Device: "webgpu",
		Hyperparams: map[string]any{
			"lr":    float64(0.05),
			"steps": float64(100),
		},
	}
	if err := w.WriteConfig(cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}

	// Graph + schedule
	graphBytes := []byte(`{"nodes":[{"id":0,"op":"add"}]}`)
	if err := w.WriteGraph(graphBytes); err != nil {
		t.Fatalf("WriteGraph: %v", err)
	}
	scheduleBytes := []byte(`{"kernels":[{"id":"K0"}]}`)
	if err := w.WriteSchedule(scheduleBytes); err != nil {
		t.Fatalf("WriteSchedule: %v", err)
	}

	// Kernels
	if err := w.WriteKernel("K0", "@compute fn k0() {}"); err != nil {
		t.Fatalf("WriteKernel K0: %v", err)
	}
	if err := w.WriteKernel("1", "@compute fn k1() {}"); err != nil {
		t.Fatalf("WriteKernel 1: %v", err)
	}

	// Streaming sinks
	losses := []LossRow{
		{Step: 0, Loss: 1.5, WallMs: 10},
		{Step: 10, Loss: 0.9, WallMs: 200},
		{Step: 20, Loss: 0.5, WallMs: 380},
	}
	for _, r := range losses {
		if err := w.AppendLoss(r); err != nil {
			t.Fatalf("AppendLoss: %v", err)
		}
	}
	gens := []GenerationRow{
		{Step: 0, TokenID: 42, TokenText: "the", LogitArgmax: 42, LogitSummary: "..."},
		{Step: 1, TokenID: 7, TokenText: " quick", LogitArgmax: 7, LogitSummary: "..."},
	}
	for _, r := range gens {
		if err := w.AppendGeneration(r); err != nil {
			t.Fatalf("AppendGeneration: %v", err)
		}
	}
	events := []Event{
		{Type: "step", Payload: json.RawMessage(`{"step":0,"loss":1.5}`)},
		{Type: "dispatch", Payload: json.RawMessage(`{"kernel":"K0"}`)},
	}
	for _, e := range events {
		if err := w.AppendEvent(e); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}

	if err := w.Finalize(1234); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	// Reopen
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	mf := r.Manifest()
	if mf.BundleVersion != 1 {
		t.Errorf("bundle_version = %d, want 1", mf.BundleVersion)
	}
	if mf.Kind != KindTrain {
		t.Errorf("kind = %v, want train", mf.Kind)
	}
	if mf.Model != "conv" {
		t.Errorf("model = %q, want conv", mf.Model)
	}
	if mf.AnnealVersion != "0.0.0-dev" {
		t.Errorf("anneal_version = %q", mf.AnnealVersion)
	}
	if mf.GitRev != "abc123" {
		t.Errorf("git_rev = %q", mf.GitRev)
	}
	if mf.DeviceName != "Apple M3" {
		t.Errorf("device_name = %q", mf.DeviceName)
	}
	if mf.Adapter != "Metal" {
		t.Errorf("adapter = %q", mf.Adapter)
	}
	if mf.WGSLHash != "deadbeef" {
		t.Errorf("wgsl_hash = %q", mf.WGSLHash)
	}
	if mf.DurationMs != 1234 {
		t.Errorf("duration_ms = %d, want 1234", mf.DurationMs)
	}
	if got := mf.SymBinds["B"]; got != 16 {
		t.Errorf("sym_binds[B] = %d, want 16", got)
	}
	if got := mf.SymBinds["T"]; got != 128 {
		t.Errorf("sym_binds[T] = %d, want 128", got)
	}

	gotCfg, err := r.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if gotCfg.Model != "conv" || gotCfg.Device != "webgpu" {
		t.Errorf("config round-trip mismatch: %+v", gotCfg)
	}
	if gotCfg.Hyperparams["lr"].(float64) != 0.05 {
		t.Errorf("config.lr = %v", gotCfg.Hyperparams["lr"])
	}

	gotGraph, err := r.Graph()
	if err != nil {
		t.Fatalf("Graph: %v", err)
	}
	if !bytes.Equal(gotGraph, graphBytes) {
		t.Errorf("graph bytes mismatch: got %q, want %q", gotGraph, graphBytes)
	}
	gotSched, err := r.Schedule()
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if !bytes.Equal(gotSched, scheduleBytes) {
		t.Errorf("schedule bytes mismatch")
	}

	names, err := r.KernelNames()
	if err != nil {
		t.Fatalf("KernelNames: %v", err)
	}
	if len(names) != 2 {
		t.Errorf("KernelNames: got %d, want 2 (%v)", len(names), names)
	}
	for _, n := range []string{"K0.wgsl", "K1.wgsl"} {
		wgsl, err := r.Kernel(n)
		if err != nil {
			t.Fatalf("Kernel %s: %v", n, err)
		}
		if wgsl == "" {
			t.Errorf("Kernel %s empty", n)
		}
	}

	gotLoss, err := r.Loss()
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if len(gotLoss) != len(losses) {
		t.Fatalf("loss rows: got %d, want %d", len(gotLoss), len(losses))
	}
	for i, want := range losses {
		if gotLoss[i].Step != want.Step || gotLoss[i].WallMs != want.WallMs {
			t.Errorf("loss[%d] mismatch: got %+v, want %+v", i, gotLoss[i], want)
		}
		// float32 round-trip via %g - exact for these values.
		if gotLoss[i].Loss != want.Loss {
			t.Errorf("loss[%d].Loss = %v, want %v", i, gotLoss[i].Loss, want.Loss)
		}
	}

	gotGen, err := r.Generation()
	if err != nil {
		t.Fatalf("Generation: %v", err)
	}
	if len(gotGen) != len(gens) {
		t.Errorf("generation rows: got %d, want %d", len(gotGen), len(gens))
	}
	for i, want := range gens {
		if gotGen[i].TokenID != want.TokenID || gotGen[i].TokenText != want.TokenText {
			t.Errorf("gen[%d] mismatch: got %+v, want %+v", i, gotGen[i], want)
		}
	}

	gotEvents, err := r.Events()
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(gotEvents) != len(events) {
		t.Errorf("events: got %d, want %d", len(gotEvents), len(events))
	}
	for i, want := range events {
		if gotEvents[i].Type != want.Type {
			t.Errorf("event[%d].Type = %q, want %q", i, gotEvents[i].Type, want.Type)
		}
		if !bytes.Equal(gotEvents[i].Payload, want.Payload) {
			t.Errorf("event[%d].Payload = %s, want %s", i, gotEvents[i].Payload, want.Payload)
		}
	}
}

func TestWriterConcurrentAppends(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "concurrent", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer func() { _ = w.Close() }()

	const goroutines = 4
	const perG = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				step := g*perG + i
				if err := w.AppendLoss(LossRow{Step: step, Loss: float32(step), WallMs: int64(step)}); err != nil {
					t.Errorf("AppendLoss: %v", err)
					return
				}
				if err := w.AppendEvent(Event{Type: "step", Payload: json.RawMessage(`{}`)}); err != nil {
					t.Errorf("AppendEvent: %v", err)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	if err := w.Finalize(0); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	loss, err := r.Loss()
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if len(loss) != goroutines*perG {
		t.Errorf("loss row count = %d, want %d", len(loss), goroutines*perG)
	}
	events, err := r.Events()
	if err != nil {
		t.Fatalf("Events: %v", err)
	}
	if len(events) != goroutines*perG {
		t.Errorf("event count = %d, want %d", len(events), goroutines*perG)
	}
}

func TestWriterPartialBundleCloseWithoutFinalize(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "partial", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	graph := []byte(`{"partial":true}`)
	if err := w.WriteGraph(graph); err != nil {
		t.Fatalf("WriteGraph: %v", err)
	}
	for i := 0; i < 5; i++ {
		if err := w.AppendLoss(LossRow{Step: i, Loss: float32(i), WallMs: int64(i)}); err != nil {
			t.Fatalf("AppendLoss: %v", err)
		}
	}
	// Simulate kill: Close without Finalize.
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle partial: %v", err)
	}
	if r.Manifest().DurationMs != 0 {
		t.Errorf("partial bundle duration_ms = %d, want 0", r.Manifest().DurationMs)
	}
	gotGraph, err := r.Graph()
	if err != nil {
		t.Fatalf("Graph on partial: %v", err)
	}
	if !bytes.Equal(gotGraph, graph) {
		t.Errorf("graph mismatch on partial bundle")
	}
	loss, err := r.Loss()
	if err != nil {
		t.Fatalf("Loss on partial: %v", err)
	}
	if len(loss) != 5 {
		t.Errorf("loss rows = %d, want 5", len(loss))
	}
}

func TestWriterFinalizeIdempotent(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "idem", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Finalize(42); err != nil {
		t.Fatalf("Finalize 1: %v", err)
	}
	if err := w.Finalize(99); err != nil {
		t.Fatalf("Finalize 2: %v", err)
	}
	r, err := OpenBundle(w.Path())
	if err != nil {
		t.Fatalf("OpenBundle: %v", err)
	}
	if r.Manifest().DurationMs != 42 {
		t.Errorf("duration_ms = %d, want 42 (first Finalize wins)", r.Manifest().DurationMs)
	}
}

func TestWriterAppendAfterFinalizeFails(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "noappend", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Finalize(0); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	if err := w.AppendLoss(LossRow{}); err == nil {
		t.Errorf("AppendLoss after Finalize: want error")
	}
	if err := w.AppendGeneration(GenerationRow{}); err == nil {
		t.Errorf("AppendGeneration after Finalize: want error")
	}
	if err := w.AppendEvent(Event{}); err == nil {
		t.Errorf("AppendEvent after Finalize: want error")
	}
	if err := w.WriteGraph([]byte(`{}`)); err == nil {
		t.Errorf("WriteGraph after Finalize: want error")
	}
}

func TestWriteKernelRejectsBadNames(t *testing.T) {
	root := t.TempDir()
	w, err := NewWriter(root, "knames", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer func() { _ = w.Close() }()
	for _, bad := range []string{"", "../escape", "foo/bar", "a..b"} {
		if err := w.WriteKernel(bad, "wgsl"); err == nil {
			t.Errorf("WriteKernel(%q): want error", bad)
		}
	}
}

func TestManifestKeyOrderingDeterministic(t *testing.T) {
	// Two manifests with identical content should marshal to byte-identical
	// JSON, including the SymBinds map.
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	m1 := Manifest{
		BundleVersion: 1,
		Kind:          KindTrain,
		Model:         "mlp",
		SymBinds:      map[string]int64{"B": 16, "T": 128, "A": 1},
		CreatedAt:     now,
	}
	m2 := Manifest{
		BundleVersion: 1,
		Kind:          KindTrain,
		Model:         "mlp",
		// Same map, different insertion order - relies on sorted output.
		SymBinds:  map[string]int64{"T": 128, "A": 1, "B": 16},
		CreatedAt: now,
	}
	b1, err := json.Marshal(m1)
	if err != nil {
		t.Fatalf("marshal m1: %v", err)
	}
	b2, err := json.Marshal(m2)
	if err != nil {
		t.Fatalf("marshal m2: %v", err)
	}
	if !bytes.Equal(b1, b2) {
		t.Errorf("manifest marshalling is nondeterministic:\n  %s\n  %s", b1, b2)
	}
	// Verify the SymBinds keys appear in lexicographic order.
	s := string(b1)
	posA := strings.Index(s, `"A":`)
	posB := strings.Index(s, `"B":`)
	posT := strings.Index(s, `"T":`)
	if posA >= posB || posB >= posT {
		t.Errorf("sym_binds keys not in lexicographic order: A=%d B=%d T=%d", posA, posB, posT)
	}
}

func TestBundleKindJSON(t *testing.T) {
	for _, c := range []struct {
		k BundleKind
		s string
	}{
		{KindTrain, `"train"`},
		{KindGenerate, `"generate"`},
		{KindSaved, `"saved"`},
	} {
		b, err := json.Marshal(c.k)
		if err != nil {
			t.Fatalf("marshal %v: %v", c.k, err)
		}
		if string(b) != c.s {
			t.Errorf("marshal %v = %s, want %s", c.k, b, c.s)
		}
		var got BundleKind
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal %s: %v", b, err)
		}
		if got != c.k {
			t.Errorf("round-trip kind: got %v, want %v", got, c.k)
		}
	}
	var k BundleKind
	if err := json.Unmarshal([]byte(`"asdf"`), &k); err == nil {
		t.Errorf("unmarshal unknown kind: want error")
	}
}

func TestLossRowParseAndCSV(t *testing.T) {
	r := LossRow{Step: 5, Loss: 0.125, WallMs: 100}
	got, err := ParseLossRow(r.CSVRow())
	if err != nil {
		t.Fatalf("ParseLossRow(%q): %v", r.CSVRow(), err)
	}
	if got != r {
		t.Errorf("ParseLossRow round-trip: got %+v, want %+v", got, r)
	}
	// Trims whitespace.
	if _, err := ParseLossRow(" 1, 2.0 , 3 "); err != nil {
		t.Errorf("ParseLossRow with whitespace: %v", err)
	}
	for _, bad := range []string{
		"only,two",
		"abc,1.0,1",
		"1,xyz,1",
		"1,1.0,xyz",
	} {
		if _, err := ParseLossRow(bad); err == nil {
			t.Errorf("ParseLossRow(%q): want error", bad)
		}
	}
}

func TestAtomicWriteTempCleanup(t *testing.T) {
	// Atomic write should not leave a .tmp file behind after success.
	root := t.TempDir()
	w, err := NewWriter(root, "atomic", KindTrain)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	defer func() { _ = w.Close() }()
	if err := w.WriteGraph([]byte(`{}`)); err != nil {
		t.Fatalf("WriteGraph: %v", err)
	}
	entries, err := os.ReadDir(w.Path())
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("found leftover .tmp file: %s", e.Name())
		}
	}
}

// TestShortHashUniqueness guards against the random shorthash being
// reused. Generate many bundles in quick succession; all ids must be
// distinct.
func TestShortHashUniqueness(t *testing.T) {
	root := t.TempDir()
	seen := map[BundleID]bool{}
	for i := 0; i < 50; i++ {
		w, err := NewWriter(root, fmt.Sprintf("m%d", i), KindTrain)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}
		if seen[w.BundleID()] {
			t.Errorf("duplicate BundleID: %s", w.BundleID())
		}
		seen[w.BundleID()] = true
		_ = w.Close()
	}
}
