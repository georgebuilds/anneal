//go:build !js

// E2E coverage for the studio's keyboard surface: `g <key>` chords, the `/`
// search focus, the `?` help modal (open / Esc-close / focus trap), and the
// `aria-current` flip on navigation. All assertions live in the real DOM —
// the chord state machine in studio.js runs end to end.

package main

import (
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
	"github.com/chromedp/chromedp/kb"
)

// pressChord types `g` then `dest` with a tiny delay so the leader timer in
// studio.js (1s window) does not lapse between presses. Each KeyEvent is its
// own keydown so the leader is armed on the first event before the second
// arrives.
func pressChord(dest string) chromedp.Action {
	return chromedp.Tasks{
		chromedp.KeyEvent("g"),
		chromedp.Sleep(50 * time.Millisecond),
		chromedp.KeyEvent(dest),
	}
}

// TestE2E_ChordNavigation drives every documented `g <key>` chord and
// asserts the URL path matches the routing table. We don't navigate via the
// nav buttons here — that's what TestE2E_AriaCurrent... covers — we drive
// the keyboard handler so the chord + leader-timer logic is real.
func TestE2E_ChordNavigation(t *testing.T) {
	base, ctx := newE2E(t)

	type chord struct {
		key  string
		path string
	}
	chords := []chord{
		{"d", "/"},
		{"v", "/v/mlp"},
		{"k", "/k/mlp"},
		{"x", "/x/Add"},
		{"t", "/t/mlp"},
		{"g", "/g/nanogpt"},
		{"h", "/h"},
		{"r", "/d"},
	}

	if err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`.nav-item[data-view="studio"]`, chromedp.ByQuery),
	); err != nil {
		t.Fatalf("initial load: %v", err)
	}

	for _, c := range chords {
		// Reset to a path that isn't the destination so a no-op chord is
		// detectable. The studio path is `/`; for `g d` we seed from `/h`.
		seed := "/h"
		if c.path == "/h" {
			seed = "/"
		}
		var got string
		err := chromedp.Run(ctx,
			chromedp.Navigate(base+seed),
			chromedp.WaitVisible(`.nav-item[data-view="studio"]`, chromedp.ByQuery),
			// Focus <main> first so document.activeElement isn't an input
			// (the keyboard handler bails on INPUT/TEXTAREA/SELECT).
			chromedp.Focus(`#main`, chromedp.ByQuery),
			pressChord(c.key),
			chromedp.Sleep(150*time.Millisecond),
			chromedp.Evaluate(`window.location.pathname`, &got),
		)
		if err != nil {
			t.Fatalf("chord g %s: %v", c.key, err)
		}
		if got != c.path {
			t.Errorf("chord g %s: location.pathname = %q, want %q", c.key, got, c.path)
		}
	}
}

// TestE2E_SlashFocusesSearch verifies pressing `/` puts focus on #search.
func TestE2E_SlashFocusesSearch(t *testing.T) {
	base, ctx := newE2E(t)
	var activeID string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`#search`, chromedp.ByQuery),
		chromedp.Focus(`#main`, chromedp.ByQuery),
		chromedp.KeyEvent("/"),
		chromedp.Sleep(80*time.Millisecond),
		chromedp.Evaluate(`document.activeElement && document.activeElement.id`, &activeID),
	)
	if err != nil {
		t.Fatalf("slash focus: %v", err)
	}
	if activeID != "search" {
		t.Fatalf("after `/`, document.activeElement.id = %q, want %q", activeID, "search")
	}
}

// TestE2E_HelpModalOpenCloseAndFocusTrap verifies:
//   - `?` opens the keyboard-help modal (#keyboard-help visible);
//   - Esc closes it (hidden again);
//   - while open, Tab from the LAST focusable cycles back to the first
//     (the focus-trap behaviour in initKeyboardHelp).
func TestE2E_HelpModalOpenCloseAndFocusTrap(t *testing.T) {
	base, ctx := newE2E(t)

	var hiddenBefore, hiddenAfterOpen, hiddenAfterEsc bool
	var trappedID string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`#helpToggle`, chromedp.ByQuery),
		chromedp.Evaluate(`document.getElementById('keyboard-help').hidden`, &hiddenBefore),

		// Open via `?`. chromedp.KeyEvent("?") sends Shift+/ on US layouts,
		// which is what the handler checks for.
		chromedp.Focus(`#main`, chromedp.ByQuery),
		chromedp.KeyEvent("?"),
		chromedp.Sleep(80*time.Millisecond),
		chromedp.Evaluate(`document.getElementById('keyboard-help').hidden`, &hiddenAfterOpen),

		// Focus the LAST focusable inside the dialog, then Tab. The trap
		// should send focus back to the FIRST focusable (the close button).
		chromedp.Evaluate(`(function(){
			var dlg = document.getElementById('keyboard-help');
			var sel = 'a[href], button:not([disabled]), textarea, input:not([disabled]), select, [tabindex]:not([tabindex="-1"])';
			var f = Array.from(dlg.querySelectorAll(sel)).filter(function(el){ return !el.hasAttribute('disabled') && el.offsetParent !== null; });
			if (f.length === 0) return '';
			f[f.length-1].focus();
			return document.activeElement && document.activeElement.className;
		})()`, nil),
		chromedp.KeyEvent("\t"),
		chromedp.Sleep(50*time.Millisecond),
		chromedp.Evaluate(`document.activeElement && document.activeElement.className`, &trappedID),

		// Esc closes the modal.
		chromedp.KeyEvent(kb.Escape),
		chromedp.Sleep(80*time.Millisecond),
		chromedp.Evaluate(`document.getElementById('keyboard-help').hidden`, &hiddenAfterEsc),
	)
	if err != nil {
		t.Fatalf("help modal: %v", err)
	}
	if !hiddenBefore {
		t.Fatalf("help modal should be hidden before `?` press")
	}
	if hiddenAfterOpen {
		t.Fatalf("help modal should be visible after `?` press")
	}
	if !strings.Contains(trappedID, "kbd-help-close") {
		t.Errorf("focus trap: after Tab from last focusable, activeElement.className = %q, want it to contain 'kbd-help-close' (focus should wrap to the first interactive)", trappedID)
	}
	if !hiddenAfterEsc {
		t.Fatalf("help modal should be hidden after Esc")
	}
}

// TestE2E_AriaCurrentTracksActiveView clicks the visualize nav item and
// confirms the active button gets aria-current="page" while the previous
// one loses it. This is the load-bearing screen-reader signal in
// setActiveView (studio.js).
func TestE2E_AriaCurrentTracksActiveView(t *testing.T) {
	base, ctx := newE2E(t)

	var initialStudio, afterClickStudio, afterClickVisualize string
	err := chromedp.Run(ctx,
		chromedp.Navigate(base+"/"),
		chromedp.WaitVisible(`.nav-item[data-view="studio"]`, chromedp.ByQuery),
		chromedp.AttributeValue(`.nav-item[data-view="studio"]`, "aria-current", &initialStudio, nil, chromedp.ByQuery),
		chromedp.Click(`.nav-item[data-view="visualize"]`, chromedp.ByQuery),
		chromedp.Sleep(150*time.Millisecond),
		chromedp.AttributeValue(`.nav-item[data-view="studio"]`, "aria-current", &afterClickStudio, nil, chromedp.ByQuery),
		chromedp.AttributeValue(`.nav-item[data-view="visualize"]`, "aria-current", &afterClickVisualize, nil, chromedp.ByQuery),
	)
	if err != nil {
		t.Fatalf("aria-current: %v", err)
	}
	if initialStudio != "page" {
		t.Errorf("initial studio aria-current = %q, want %q", initialStudio, "page")
	}
	if afterClickStudio == "page" {
		t.Errorf("after click on visualize, studio still has aria-current=%q (should be cleared)", afterClickStudio)
	}
	if afterClickVisualize != "page" {
		t.Errorf("after click on visualize, visualize aria-current = %q, want %q", afterClickVisualize, "page")
	}
}
