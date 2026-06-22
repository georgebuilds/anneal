//go:build !js

// End-to-end browser tests for the `anneal web` studio.
//
// The existing cmd_web_*_test.go suite grep the HTML/CSS/JS strings the
// server returns. They never load the studio in a real browser, so routing,
// the theme cycle, keyboard chords (`g d`, `g v`, ...), the `?` help modal +
// its focus trap, the polite live-region announcements, and the `aria-current`
// updates are untested as behaviour.
//
// These tests close that gap with chromedp (headless Chrome driven from Go).
// Each test:
//   1. boots `httptest.NewServer(serveMux())` (same handler the binary serves);
//   2. opens a fresh chromedp context with a sensible timeout;
//   3. drives the studio and asserts ONE behaviour.
//
// If Chrome is not available on the host, the harness skips cleanly - tests
// must NOT fail when the browser is missing. Mirrors the spirit of requireGPU
// in cli_test.go.

package main

import (
	"context"
	"errors"
	"net/http/httptest"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
)

// e2eTimeout caps each browser test. Chrome cold-start on macOS is ~600ms;
// 25s gives a generous ceiling for the slowest CI runner without making
// failures hang forever.
const e2eTimeout = 25 * time.Second

// findChrome returns the absolute path to a Chrome / Chromium executable, or
// an error if none is reachable. We probe the same locations chromedp's own
// findExecPath does so callers can produce a useful Skip message before we
// pay the cost of starting the browser.
func findChrome() (string, error) {
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Chromium.app/Contents/MacOS/Chromium",
			"google-chrome",
			"chromium",
		}
	case "windows":
		candidates = []string{
			"chrome.exe",
			"chrome",
		}
	default:
		candidates = []string{
			"google-chrome",
			"google-chrome-stable",
			"chromium",
			"chromium-browser",
			"headless_shell",
			"headless-shell",
		}
	}
	for _, c := range candidates {
		if p, err := exec.LookPath(c); err == nil {
			return p, nil
		}
	}
	return "", errors.New("no Chrome / Chromium executable found in PATH or known macOS app locations")
}

// newE2E boots an in-process studio server and a chromedp browser context.
// Returns the server URL and a context the caller can hand to chromedp.Run.
// All resources are torn down via t.Cleanup; the test only has to call Run.
//
// If Chrome cannot be located OR the allocator can't spawn the browser, the
// test is skipped (not failed). This matches the contract from the task brief
// - the foundation tests should never block CI on a missing browser.
func newE2E(t *testing.T) (string, context.Context) {
	t.Helper()

	chromePath, err := findChrome()
	if err != nil {
		t.Skipf("e2e: %v", err)
	}

	srv := httptest.NewServer(serveMux())
	t.Cleanup(srv.Close)

	// Allocator flags: headless + no-sandbox covers macOS local dev AND Linux
	// CI containers without granting CAP_SYS_ADMIN. disable-gpu is harmless
	// and avoids the WebGL banner in headless mode. We also explicitly set
	// ExecPath so the allocator doesn't waste time on its own PATH scan.
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.ExecPath(chromePath),
		chromedp.NoSandbox,
		chromedp.DisableGPU,
		chromedp.Flag("headless", true),
		chromedp.Flag("hide-scrollbars", true),
		chromedp.Flag("mute-audio", true),
	)

	allocCtx, allocCancel := chromedp.NewExecAllocator(context.Background(), opts...)
	t.Cleanup(allocCancel)

	browserCtx, browserCancel := chromedp.NewContext(allocCtx)
	t.Cleanup(browserCancel)

	// Apply the per-test timeout to the browser context itself so any action
	// - including the implicit Start() inside the first chromedp.Run - is
	// bounded. Without this, a missing-Chrome binary could hang for the full
	// `go test` timeout (10m default).
	timedCtx, timeoutCancel := context.WithTimeout(browserCtx, e2eTimeout)
	t.Cleanup(timeoutCancel)

	// Probe that the browser actually starts. If it doesn't (e.g. the binary
	// exists but lacks the right runtime libs in a stripped CI image), Skip.
	if err := chromedp.Run(timedCtx); err != nil {
		t.Skipf("e2e: cannot start Chrome: %v", err)
	}

	return srv.URL, timedCtx
}

// ── 1. foundation ───────────────────────────────────────────────────────────

// TestE2E_RootLoads verifies the studio shell renders in a real browser.
// Pins document.title and the visible h1 in the active view. The h1 is in
// the studio section which is the default active view on `/`.
func TestE2E_RootLoads(t *testing.T) {
	base, ctx := newE2E(t)
	var title string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`.hero-quote`, chromedp.ByQuery),
		chromedp.Title(&title),
	)
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	if !strings.Contains(strings.ToLower(title), "anneal") {
		t.Fatalf("document.title = %q, want it to contain 'anneal'", title)
	}
}

// TestE2E_NoMissingWasmConsoleError verifies that without the
// <meta name="anneal-worker"> tag (the W0 default), the page boots without
// throwing a "failed to fetch anneal.wasm" console error. We don't try to
// catch console events here; instead we smoke-check that a late-bound DOM
// node (the SVG inside the brand cell) gets rendered, which only happens
// when the page's JS evaluates without an unhandled exception.
func TestE2E_NoMissingWasmConsoleError(t *testing.T) {
	base, ctx := newE2E(t)
	var workerLoaded bool
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`#brand-mark`, chromedp.ByQuery),
		// document.querySelector('meta[name="anneal-worker"]') must be null
		// in W0; if it ever stops being null, this test will tell us to
		// update the wasm-shipped assertions too.
		chromedp.Evaluate(`!!document.querySelector('meta[name="anneal-worker"]')`, &workerLoaded),
	)
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	if workerLoaded {
		t.Fatalf("expected anneal-worker meta tag to be absent in W0, but it was present")
	}
}

// ── 2. theme controller ─────────────────────────────────────────────────────

// TestE2E_ThemeCycleAndPersist clicks #themeToggle and verifies the
// data-theme attribute on <html> follows the documented system→dark→light
// cycle, AND that the choice survives a page reload (localStorage).
func TestE2E_ThemeCycleAndPersist(t *testing.T) {
	base, ctx := newE2E(t)

	// Clear any persisted theme before we start - other tests may have left
	// state in this browser profile (unlikely; chromedp gives us a clean
	// user-data-dir per allocator, but belt-and-braces).
	var before, afterOne, afterReload string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`#themeToggle`, chromedp.ByQuery),
		chromedp.Evaluate(`localStorage.removeItem('anneal-theme'); document.documentElement.setAttribute('data-theme','system'); document.documentElement.getAttribute('data-theme')`, &before),
		chromedp.Click(`#themeToggle`, chromedp.ByQuery),
		chromedp.Evaluate(`document.documentElement.getAttribute('data-theme')`, &afterOne),
		chromedp.Reload(),
		chromedp.WaitVisible(`#themeToggle`, chromedp.ByQuery),
		chromedp.Evaluate(`document.documentElement.getAttribute('data-theme')`, &afterReload),
	)
	if err != nil {
		t.Fatalf("theme cycle: %v", err)
	}
	if before != "system" {
		t.Fatalf("pre-click data-theme = %q, want %q", before, "system")
	}
	// system → dark per the documented cycle order.
	if afterOne != "dark" {
		t.Fatalf("after one click data-theme = %q, want %q", afterOne, "dark")
	}
	if afterReload != "dark" {
		t.Fatalf("after reload data-theme = %q, want %q (localStorage persistence broken)", afterReload, "dark")
	}
}

// TestE2E_ThemeQueryParamOverride confirms ?theme=light forces data-theme=light
// regardless of any previously persisted choice. Documented in spec §10 and
// README-DEV.md.
func TestE2E_ThemeQueryParamOverride(t *testing.T) {
	base, ctx := newE2E(t)
	var theme string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		// Seed localStorage with something OTHER than light so the URL
		// override is the only path that can produce 'light'.
		chromedp.Evaluate(`localStorage.setItem('anneal-theme','dark')`, nil),
		chromedp.Navigate(base+"/?theme=light"),
		chromedp.WaitVisible(`#themeToggle`, chromedp.ByQuery),
		chromedp.Evaluate(`document.documentElement.getAttribute('data-theme')`, &theme),
	)
	if err != nil {
		t.Fatalf("theme override: %v", err)
	}
	if theme != "light" {
		t.Fatalf("?theme=light produced data-theme=%q, want %q", theme, "light")
	}
}

// Keyboard chord / focus-trap / aria-current tests live in
// web_e2e_keyboard_test.go. View smoke tests live in web_e2e_views_test.go.
