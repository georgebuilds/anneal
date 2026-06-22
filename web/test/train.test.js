// Tests for the train view (W6) in web/studio.js.
//
// The train view is SSE-driven: trainStart() opens an EventSource against
// /sse/train and pushes each Snapshot frame into the DOM + a loss ring buffer.
// jsdom (and the test Node) have no EventSource, so we stub a controllable
// FakeEventSource via vi.stubGlobal and drive frames synchronously through
// its registered listeners. This lets us assert the real DOM/state
// transitions that each snapshot produces.
//
// Covered seam entries: __studio.renderTrainView, trainStart, trainCancel,
// modelFromTrainPath. The render assertions check control wiring + URL deep
// link sync; the stream assertions check progress/metrics/sparkline DOM and
// the start/cancel disabled-button lifecycle; the no-EventSource branch
// checks the catch path falls back to the idle (cancelled) UI.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { loadStudio } from './harness.js';

// FakeEventSource: a synchronous, inspectable stand-in for the browser
// EventSource. Constructing one records the URL and registers it as the
// "live" instance so a test can grab it and emit frames. close() flips
// readyState to CLOSED (2), matching the real contract studio.js checks.
class FakeEventSource {
  static instances = [];
  static last() { return FakeEventSource.instances[FakeEventSource.instances.length - 1]; }
  static reset() { FakeEventSource.instances = []; }

  constructor(url) {
    this.url = url;
    this.readyState = 1; // OPEN
    this.listeners = {};
    this.closed = false;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  // emit dispatches a named SSE event with a JSON-string `data` payload,
  // mirroring how the browser delivers MessageEvents to studio.js.
  emit(type, data) {
    const ev = { data: data == null ? undefined : JSON.stringify(data) };
    (this.listeners[type] || []).forEach((fn) => fn(ev));
  }
  // emitRaw dispatches with the data already a string (or undefined), so a
  // test can feed malformed JSON to exercise the parse-guard.
  emitRaw(type, raw) {
    (this.listeners[type] || []).forEach((fn) => fn({ data: raw }));
  }
  close() {
    this.closed = true;
    this.readyState = 2; // CLOSED
  }
}
// Match the EventSource.CLOSED constant studio.js reads in its error handler.
FakeEventSource.CLOSED = 2;

let studio;

beforeEach(async () => {
  FakeEventSource.reset();
  vi.stubGlobal('EventSource', FakeEventSource);
  studio = await loadStudio({ path: '/t/conv' });
});

afterEach(() => {
  // Ensure no stream is left open between tests.
  try { studio.__studio.trainCancel(); } catch (_) {}
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('modelFromTrainPath', () => {
  it('parses the model slug from /t/<model>', () => {
    expect(studio.__studio.modelFromTrainPath('/t/conv')).toBe('conv');
    expect(studio.__studio.modelFromTrainPath('/t/nanogpt')).toBe('nanogpt');
  });

  it('decodes percent-encoded slugs', () => {
    expect(studio.__studio.modelFromTrainPath('/t/my%20model')).toBe('my model');
  });

  it('ignores query/hash suffixes', () => {
    expect(studio.__studio.modelFromTrainPath('/t/mlp?steps=5')).toBe('mlp');
    expect(studio.__studio.modelFromTrainPath('/t/mlp#frag')).toBe('mlp');
  });

  it('defaults to mlp for non-matching paths', () => {
    expect(studio.__studio.modelFromTrainPath('/')).toBe('mlp');
    expect(studio.__studio.modelFromTrainPath('/g/gpt2')).toBe('mlp');
    expect(studio.__studio.modelFromTrainPath('/t/')).toBe('mlp');
  });
});

describe('renderTrainView', () => {
  it('syncs the model <select> from the URL deep link', () => {
    // Booted at /t/conv, so render (run via setActiveView on boot) should
    // have selected the conv option.
    const sel = document.getElementById('train-model');
    expect(sel.value).toBe('conv');
  });

  it('leaves the default selection for an unknown URL model', () => {
    window.history.replaceState({}, '', '/t/does-not-exist');
    const sel = document.getElementById('train-model');
    const before = sel.value;
    studio.__studio.renderTrainView();
    // Unknown model -> option does not exist -> selection unchanged, but
    // the internal state still tracks the requested model.
    expect(sel.value).toBe(before);
  });

  it('wires start/cancel click handlers', () => {
    const start = document.getElementById('train-start');
    const cancel = document.getElementById('train-cancel');
    studio.__studio.renderTrainView();
    expect(typeof start.onclick).toBe('function');
    expect(typeof cancel.onclick).toBe('function');
  });

  it('save-run button announces bundle status when bundle is checked', () => {
    vi.useFakeTimers();
    studio.__studio.renderTrainView();
    const bundle = document.getElementById('train-bundle');
    bundle.checked = true;
    document.getElementById('train-save-run').onclick();
    vi.advanceTimersByTime(250); // announce() defers via setTimeout(200)
    const live = document.getElementById('live-region');
    expect(live.textContent).toContain('bundle saved');
  });

  it('save-run button announces disabled hint when bundle unchecked', () => {
    vi.useFakeTimers();
    studio.__studio.renderTrainView();
    const bundle = document.getElementById('train-bundle');
    bundle.checked = false;
    document.getElementById('train-save-run').onclick();
    vi.advanceTimersByTime(250);
    const live = document.getElementById('live-region');
    expect(live.textContent).toContain('bundle disabled');
  });
});

describe('trainStart - happy path streaming', () => {
  it('opens an EventSource with model/steps/bundle params and toggles buttons', () => {
    const sel = document.getElementById('train-model');
    sel.value = 'mlp';
    document.getElementById('train-steps').value = '50';
    document.getElementById('train-bundle').checked = true;

    studio.__studio.trainStart();

    const es = FakeEventSource.last();
    expect(es).toBeTruthy();
    expect(es.url).toContain('/sse/train');
    expect(es.url).toContain('model=mlp');
    expect(es.url).toContain('steps=50');
    expect(es.url).toContain('bundle=1');

    expect(document.getElementById('train-start').disabled).toBe(true);
    expect(document.getElementById('train-cancel').disabled).toBe(false);
    // Reset display: step shows "0 / 50".
    expect(document.getElementById('t-step').textContent).toBe('0 / 50');
  });

  it('clamps steps into [1, 10000] and defaults to 100 when blank', () => {
    document.getElementById('train-steps').value = '';
    studio.__studio.trainStart();
    expect(document.getElementById('t-step').textContent).toBe('0 / 100');

    studio.__studio.trainCancel();
    document.getElementById('train-steps').value = '999999';
    studio.__studio.trainStart();
    expect(document.getElementById('t-step').textContent).toBe('0 / 10000');
  });

  it('updates step / loss / compiler-stat DOM as snapshots arrive', () => {
    document.getElementById('train-steps').value = '10';
    studio.__studio.trainStart();
    const es = FakeEventSource.last();

    es.emit('message', {
      step: 1, max_steps: 10, has_loss: true, loss: 2.5,
      uops_count: 120, kernels_count: 9, fused_count: 4,
    });
    expect(document.getElementById('t-step').textContent).toBe('1 / 10');
    expect(document.getElementById('t-loss').textContent).toBe('2.500000');
    expect(document.getElementById('t-uops').textContent).toBe('120');
    expect(document.getElementById('t-kernels').textContent).toBe('9');
    expect(document.getElementById('t-fused').textContent).toBe('4');

    es.emit('message', { step: 5, max_steps: 10, has_loss: true, loss: 1.0 });
    expect(document.getElementById('t-step').textContent).toBe('5 / 10');
    expect(document.getElementById('t-loss').textContent).toBe('1.000000');
  });

  it('animates the progress bar aria-valuenow + fill width', () => {
    document.getElementById('train-steps').value = '10';
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    const bar = document.getElementById('train-progress-bar');
    const fill = document.getElementById('train-progress-fill');

    es.emit('message', { step: 5, max_steps: 10, has_loss: true, loss: 1.0 });
    expect(bar.getAttribute('aria-valuenow')).toBe('50');
    // jsdom normalizes "50.0%" -> "50%"; assert the parsed numeric value.
    expect(parseFloat(fill.style.width)).toBeCloseTo(50, 5);

    es.emit('message', { step: 10, max_steps: 10, has_loss: true, loss: 0.5 });
    expect(bar.getAttribute('aria-valuenow')).toBe('100');
    expect(parseFloat(fill.style.width)).toBeCloseTo(100, 5);
  });

  it('draws the loss sparkline + updates the SVG desc fallback', () => {
    vi.useFakeTimers(); // rAF is polyfilled via setTimeout in setup.js
    document.getElementById('train-steps').value = '10';
    studio.__studio.trainStart();
    const es = FakeEventSource.last();

    es.emit('message', { step: 1, max_steps: 10, has_loss: true, loss: 3.0 });
    es.emit('message', { step: 2, max_steps: 10, has_loss: true, loss: 2.0 });
    es.emit('message', { step: 3, max_steps: 10, has_loss: true, loss: 1.0 });
    vi.runAllTimers(); // flush the coalesced rAF that calls drawSparkline()

    const path = document.getElementById('loss-path');
    expect(path.getAttribute('d')).toMatch(/^M/);
    const desc = document.getElementById('loss-svg-desc');
    expect(desc.textContent).toContain('decreased');
    expect(desc.textContent).toContain('3 samples');
  });

  it('pulses a kernel dot when a snapshot carries last_kernel_id', () => {
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.emit('message', { step: 1, max_steps: 5, last_kernel_id: 'k:matmul/0' });
    const svg = document.getElementById('kernel-svg');
    const dots = svg.querySelectorAll('.kernel-dot');
    expect(dots.length).toBe(1);
    expect(dots[0].classList.contains('dispatched')).toBe(true);
  });

  it('a malformed JSON frame is ignored (parse guard)', () => {
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.emit('message', { step: 2, max_steps: 5, has_loss: true, loss: 1.0 });
    es.emitRaw('message', '{not valid json');
    // Still showing the last good frame, no throw.
    expect(document.getElementById('t-step').textContent).toBe('2 / 5');
  });

  it('a phase:error snapshot announces and finishes the run', () => {
    vi.useFakeTimers();
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.emit('message', { step: 1, max_steps: 5, phase: 'error', error: 'oom' });
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('oom');
    // finishTrain re-enables start, disables cancel.
    expect(document.getElementById('train-start').disabled).toBe(false);
    expect(document.getElementById('train-cancel').disabled).toBe(true);
  });

  it('the "done" SSE event finishes the run and re-enables actions', () => {
    document.getElementById('train-bundle').checked = true;
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.emit('message', { step: 5, max_steps: 5, has_loss: true, loss: 0.1 });
    es.emit('done');
    expect(es.closed).toBe(true);
    expect(document.getElementById('train-start').disabled).toBe(false);
    expect(document.getElementById('train-cancel').disabled).toBe(true);
    // save-run is enabled because bundle was requested.
    expect(document.getElementById('train-save-run').disabled).toBe(false);
    expect(document.getElementById('train-open-viz').disabled).toBe(false);
  });

  it('a double press cancels the previous stream before opening a new one', () => {
    studio.__studio.trainStart();
    const first = FakeEventSource.last();
    studio.__studio.trainStart();
    const second = FakeEventSource.last();
    expect(first).not.toBe(second);
    expect(first.closed).toBe(true);
    expect(second.closed).toBe(false);
  });

  it('a phase:done snapshot finishes the run', () => {
    document.getElementById('train-bundle').checked = false;
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.emit('message', { step: 5, max_steps: 5, phase: 'done' });
    expect(es.closed).toBe(true);
    expect(document.getElementById('train-start').disabled).toBe(false);
    // bundle was off -> save-run stays disabled.
    expect(document.getElementById('train-save-run').disabled).toBe(true);
  });

  it('a network error (readyState OPEN) announces and finishes', () => {
    vi.useFakeTimers();
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.readyState = 1; // OPEN -> network failure branch
    es.emit('error');
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('stream error');
    expect(document.getElementById('train-start').disabled).toBe(false);
  });

  it('an error after close (readyState CLOSED) silently finishes', () => {
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    es.close();          // readyState -> CLOSED (2)
    es.emit('error');    // CLOSED branch: no announce, just finish
    expect(document.getElementById('train-start').disabled).toBe(false);
    expect(document.getElementById('train-cancel').disabled).toBe(true);
  });

  it('the open-in-viz button navigates to the visualize view', () => {
    studio.__studio.renderTrainView();
    const openViz = document.getElementById('train-open-viz');
    openViz.onclick();
    expect(window.location.pathname).toBe('/v/conv');
    expect(document.getElementById('view-visualize').classList.contains('active')).toBe(true);
  });
});

describe('trainCancel', () => {
  it('closes the active stream and restores the idle button state', () => {
    vi.useFakeTimers();
    studio.__studio.trainStart();
    const es = FakeEventSource.last();
    expect(document.getElementById('train-cancel').disabled).toBe(false);

    studio.__studio.trainCancel();
    expect(es.closed).toBe(true);
    expect(document.getElementById('train-start').disabled).toBe(false);
    expect(document.getElementById('train-cancel').disabled).toBe(true);
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('cancelled');
  });

  it('is a no-op when no stream is open', () => {
    // No active es; should not throw and leaves start enabled.
    expect(() => studio.__studio.trainCancel()).not.toThrow();
    expect(document.getElementById('train-start').disabled).toBe(false);
  });
});

describe('trainStart - no EventSource / error branch', () => {
  it('falls back to idle UI when EventSource construction throws', () => {
    vi.useFakeTimers();
    // Replace EventSource with one whose constructor throws (simulates an
    // environment with no SSE support).
    function ThrowingES() { throw new Error('SSE unavailable'); }
    vi.stubGlobal('EventSource', ThrowingES);
    studio.__studio.trainStart();
    // The catch path announces and calls trainCancel(): start re-enabled,
    // cancel disabled, no live stream recorded.
    vi.advanceTimersByTime(250);
    expect(document.getElementById('train-start').disabled).toBe(false);
    expect(document.getElementById('train-cancel').disabled).toBe(true);
    expect(document.getElementById('live-region').textContent).toContain('cannot open SSE');
  });
});
