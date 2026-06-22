//go:build !js

// View smoke E2E for the studio. One test per view: navigate to the route
// via the SPA, assert a stable landmark renders. The intent is the same as
// `curl` checks but with the full History-API router + view dispatcher in
// the loop, so a regression in the JS router shows up here.

package main

import (
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
)

// viewSmoke runs a single SPA navigation and waits for a known landmark in
// the target view. The wait is bounded by the per-test browser context, so
// a hung renderer fails fast instead of timing out the whole `go test` run.
func viewSmoke(t *testing.T, path, landmark string) {
	t.Helper()
	base, ctx := newE2E(t)
	if err := chromedp.Run(ctx,
		chromedp.Navigate(base+path),
		chromedp.WaitVisible(landmark, chromedp.ByQuery),
		// Settle: give async renderers (doctor: fetch /api/device, explain:
		// wasm.call rejection path) one tick to finish so a follow-up
		// Evaluate would see a stable DOM. WaitVisible already returned; the
		// 50ms is for the polite-live-region updates.
		chromedp.Sleep(50*time.Millisecond),
	); err != nil {
		t.Fatalf("view %s: WaitVisible(%q): %v", path, landmark, err)
	}
}

// TestE2E_ViewDoctor - /d. The doctor renderer always inserts native + browser
// cards; the <article class="doctor-card"> wrapper is in the HTML shell.
func TestE2E_ViewDoctor(t *testing.T) {
	viewSmoke(t, "/d", `#view-doctor.active h1`)
}

// TestE2E_ViewExplainAdd - /x/Add. The h1 is in the shell; the rules list
// may stay empty (no WASM) but the section is reachable.
func TestE2E_ViewExplainAdd(t *testing.T) {
	viewSmoke(t, "/x/Add", `#view-explain.active h1`)
}

// TestE2E_ViewVisualizeMLP - /v/mlp. The iframe wrapper is in the shell.
// We only assert the iframe element is in the active view; we don't wait
// on its inner document to load (the viz artifact is a separate concern
// and pulls in WASM that may not exist in this build).
func TestE2E_ViewVisualizeMLP(t *testing.T) {
	viewSmoke(t, "/v/mlp", `#view-visualize.active #viz-iframe`)
}

// TestE2E_ViewKernelsMLP - /k/mlp. The WGSL <pre> is the landmark; renders
// even when the WASM bridge rejects.
func TestE2E_ViewKernelsMLP(t *testing.T) {
	viewSmoke(t, "/k/mlp", `#view-kernels.active #k-wgsl`)
}

// TestE2E_ViewTrainMLP - /t/mlp. The start button is the landmark; we don't
// actually start a run.
func TestE2E_ViewTrainMLP(t *testing.T) {
	viewSmoke(t, "/t/mlp", `#view-train.active #train-start`)
}

// TestE2E_ViewGenerateNanogpt - /g/nanogpt. The prompt input is the
// landmark; we don't fire generate.
func TestE2E_ViewGenerateNanogpt(t *testing.T) {
	viewSmoke(t, "/g/nanogpt", `#view-generate.active #gen-prompt`)
}

// TestE2E_ViewHistory - /h. The history view is a stub in W0; the h1 is the
// only DOM that's guaranteed to render.
func TestE2E_ViewHistory(t *testing.T) {
	viewSmoke(t, "/h", `#view-history.active h1`)
}

// TestE2E_StudioHomeHasDropzones - `/`. The studio home includes the
// ONNX + tensor dropzones (W8 + W9). Both should render even without WASM.
func TestE2E_StudioHomeHasDropzones(t *testing.T) {
	base, ctx := newE2E(t)
	var onnxLabel string
	if err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`#onnx-dropzone`, chromedp.ByQuery),
		chromedp.WaitVisible(`#tensor-dropzone`, chromedp.ByQuery),
		chromedp.AttributeValue(`#onnx-dropzone`, "aria-label", &onnxLabel, nil, chromedp.ByQuery),
	); err != nil {
		t.Fatalf("studio dropzones: %v", err)
	}
	if !strings.Contains(strings.ToLower(onnxLabel), "onnx") {
		t.Errorf("ONNX dropzone aria-label = %q, want it to mention 'ONNX'", onnxLabel)
	}
}
