// Tests for the generate view (W7) in web/studio.js.
//
// Like the train view, generate is SSE-driven: generateStart() opens an
// EventSource against /sse/generate and appends each TokenSnapshot as a
// focusable <span class="tok"> in #gen-tokens-out, updating the last-token
// logit panel and a click-through href. We stub EventSource with a
// synchronous FakeEventSource and emit frames directly so the real DOM /
// state transitions are observable.
//
// Covered seam entries: __studio.renderGenerateView, generateStart,
// generateCancel, modelFromGenPath. Assertions check URL/control sync, the
// prompt-required guard, token span emission + last-token panel, the
// click-through href, the ref-match (compare) glyphs, batched announcement,
// the done/error lifecycle, cancel, and the no-EventSource branch.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { loadStudio } from './harness.js';

class FakeEventSource {
  static instances = [];
  static last() { return FakeEventSource.instances[FakeEventSource.instances.length - 1]; }
  static reset() { FakeEventSource.instances = []; }

  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.listeners = {};
    this.closed = false;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  emit(type, data) {
    const ev = { data: data == null ? undefined : JSON.stringify(data) };
    (this.listeners[type] || []).forEach((fn) => fn(ev));
  }
  emitRaw(type, raw) {
    (this.listeners[type] || []).forEach((fn) => fn({ data: raw }));
  }
  close() {
    this.closed = true;
    this.readyState = 2;
  }
}
FakeEventSource.CLOSED = 2;

let studio;

beforeEach(async () => {
  FakeEventSource.reset();
  vi.stubGlobal('EventSource', FakeEventSource);
  studio = await loadStudio({ path: '/g/nanogpt' });
});

afterEach(() => {
  try { studio.__studio.generateCancel(); } catch (_) {}
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('modelFromGenPath', () => {
  it('parses the model slug from /g/<model>', () => {
    expect(studio.__studio.modelFromGenPath('/g/nanogpt')).toBe('nanogpt');
    expect(studio.__studio.modelFromGenPath('/g/gpt2')).toBe('gpt2');
  });

  it('decodes percent-encoded slugs and ignores query/hash', () => {
    expect(studio.__studio.modelFromGenPath('/g/my%20gpt')).toBe('my gpt');
    expect(studio.__studio.modelFromGenPath('/g/gpt2?prompt=hi')).toBe('gpt2');
    expect(studio.__studio.modelFromGenPath('/g/gpt2#x')).toBe('gpt2');
  });

  it('defaults to gpt2 for non-matching paths', () => {
    expect(studio.__studio.modelFromGenPath('/')).toBe('gpt2');
    expect(studio.__studio.modelFromGenPath('/t/mlp')).toBe('gpt2');
    expect(studio.__studio.modelFromGenPath('/g/')).toBe('gpt2');
  });
});

describe('renderGenerateView', () => {
  it('syncs the model <select> from the URL deep link', () => {
    // Booted at /g/nanogpt.
    expect(document.getElementById('gen-model').value).toBe('nanogpt');
  });

  it('applies prompt/tokens/compare/bundle from URL query params', () => {
    window.history.replaceState({}, '', '/g/gpt2?prompt=Once%20upon&tokens=64&compare=1&bundle=0');
    studio.__studio.renderGenerateView();
    expect(document.getElementById('gen-model').value).toBe('gpt2');
    expect(document.getElementById('gen-prompt').value).toBe('Once upon');
    expect(document.getElementById('gen-tokens').value).toBe('64');
    expect(document.getElementById('gen-compare').checked).toBe(true);
    expect(document.getElementById('gen-bundle').checked).toBe(false);
  });

  it('clamps a too-large tokens query param to 256', () => {
    window.history.replaceState({}, '', '/g/gpt2?tokens=99999');
    studio.__studio.renderGenerateView();
    expect(document.getElementById('gen-tokens').value).toBe('256');
  });

  it('wires start/cancel click handlers', () => {
    studio.__studio.renderGenerateView();
    expect(typeof document.getElementById('gen-start').onclick).toBe('function');
    expect(typeof document.getElementById('gen-cancel').onclick).toBe('function');
  });
});

describe('generateStart — prompt guard', () => {
  it('refuses to start with an empty/whitespace prompt and announces', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = '   ';
    studio.__studio.generateStart();
    expect(FakeEventSource.instances.length).toBe(0);
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('prompt is required');
    // Buttons untouched: start stays enabled.
    expect(document.getElementById('gen-start').disabled).toBe(false);
  });
});

describe('generateStart — happy path streaming', () => {
  it('opens an EventSource with model/prompt/tokens/compare/bundle params', () => {
    document.getElementById('gen-model').value = 'gpt2';
    document.getElementById('gen-prompt').value = 'Hello world';
    document.getElementById('gen-tokens').value = '16';
    document.getElementById('gen-compare').checked = true;
    document.getElementById('gen-bundle').checked = false;

    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    expect(es.url).toContain('/sse/generate');
    expect(es.url).toContain('model=gpt2');
    expect(es.url).toContain('prompt=Hello+world');
    expect(es.url).toContain('tokens=16');
    expect(es.url).toContain('compare=1');
    expect(es.url).toContain('bundle=0');

    expect(document.getElementById('gen-start').disabled).toBe(true);
    expect(document.getElementById('gen-cancel').disabled).toBe(false);
    // Prompt echo + status reset.
    expect(document.getElementById('gen-prompt-echo').textContent).toBe('Hello world');
    expect(document.getElementById('gen-status').textContent).toBe('starting…');
    expect(document.getElementById('gen-warming').hidden).toBe(false);
  });

  it('the first frame hides the warming hint and sets streaming status', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'init' });
    expect(document.getElementById('gen-warming').hidden).toBe(true);
    expect(document.getElementById('gen-status').textContent).toContain('warming up done');
  });

  it('appends a focusable token span per generating frame and updates the panel', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();

    es.emit('message', { phase: 'generating', token_text: ' the', token_id: 262, logit_summary: 'max 8.1', last_kernel_id: 'k:attn/3' });
    const out = document.getElementById('gen-tokens-out');
    const spans = out.querySelectorAll('span.tok');
    expect(spans.length).toBe(1);
    expect(spans[0].textContent).toBe(' the');
    expect(spans[0].getAttribute('tabindex')).toBe('0');
    expect(spans[0].dataset.tokenId).toBe('262');
    expect(spans[0].dataset.kernelId).toBe('k:attn/3');
    expect(spans[0].classList.contains('fresh')).toBe(true);

    // Last-token panel.
    expect(document.getElementById('gen-last-text').textContent).toBe('" the"');
    expect(document.getElementById('gen-last-id').textContent).toBe('262');
    expect(document.getElementById('gen-last-logit').textContent).toBe('max 8.1');

    // Second token: only the newest carries .fresh.
    es.emit('message', { phase: 'generating', token_text: ' cat', token_id: 3797, last_kernel_id: 'k:mlp/7' });
    const spans2 = out.querySelectorAll('span.tok');
    expect(spans2.length).toBe(2);
    expect(spans2[0].classList.contains('fresh')).toBe(false);
    expect(spans2[1].classList.contains('fresh')).toBe(true);
  });

  it('updates the click-through href to the producing kernel', () => {
    document.getElementById('gen-model').value = 'gpt2';
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'x', token_id: 1, last_kernel_id: 'k:logits/0' });
    const link = document.getElementById('gen-click-through');
    expect(link.getAttribute('href')).toBe('/k/gpt2?kernel=k%3Alogits%2F0');
  });

  it('renders ref-match glyphs and panel text when compare frames arrive', () => {
    document.getElementById('gen-prompt').value = 'hi';
    document.getElementById('gen-compare').checked = true;
    studio.__studio.generateStart();
    const es = FakeEventSource.last();

    es.emit('message', { phase: 'generating', token_text: 'a', token_id: 1, ref_match: true });
    let span = document.getElementById('gen-tokens-out').querySelector('span.tok');
    expect(span.classList.contains('refmatch-yes')).toBe(true);
    expect(document.getElementById('gen-last-ref').textContent).toContain('match');

    es.emit('message', { phase: 'generating', token_text: 'b', token_id: 2, ref_match: false });
    const spans = document.getElementById('gen-tokens-out').querySelectorAll('span.tok');
    expect(spans[1].classList.contains('refmatch-no')).toBe(true);
    expect(document.getElementById('gen-last-ref').textContent).toContain('no match');
  });

  it('click-through on a token span navigates to the kernels view', () => {
    document.getElementById('gen-model').value = 'gpt2';
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'x', token_id: 1, last_kernel_id: 'k:attn/2' });
    const span = document.getElementById('gen-tokens-out').querySelector('span.tok');
    span.dispatchEvent(new window.Event('click'));
    expect(window.location.pathname).toBe('/k/gpt2');
    expect(window.location.search).toBe('?kernel=k%3Aattn%2F2');
    // kernels view becomes active.
    expect(document.getElementById('view-kernels').classList.contains('active')).toBe(true);
  });

  it('Enter key on a token span also activates the click-through', () => {
    document.getElementById('gen-model').value = 'gpt2';
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'x', token_id: 1, last_kernel_id: 'k:zz/1' });
    const span = document.getElementById('gen-tokens-out').querySelector('span.tok');
    span.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter' }));
    expect(window.location.pathname).toBe('/k/gpt2');
    expect(window.location.search).toContain('kernel=');
  });

  it('a malformed JSON frame is ignored (parse guard)', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'ok', token_id: 1 });
    es.emitRaw('message', 'not json{');
    expect(document.getElementById('gen-tokens-out').querySelectorAll('span.tok').length).toBe(1);
  });

  it('batches the announcement after 5 tokens', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    for (let i = 0; i < 5; i++) {
      es.emit('message', { phase: 'generating', token_text: 't' + i, token_id: i });
    }
    // Flush hit at 5 tokens -> announce() scheduled via setTimeout(200).
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('generated 5 tokens');
  });

  it('a phase:error frame announces and finishes', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'error', error: 'cuda fault' });
    vi.advanceTimersByTime(250);
    expect(document.getElementById('gen-status').textContent).toContain('cuda fault');
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
  });

  it('the "done" SSE event sets the summary status and finishes', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'a', token_id: 1 });
    es.emit('done', { total_tokens: 7, wall_ms: 1234 });
    expect(es.closed).toBe(true);
    expect(document.getElementById('gen-status').textContent).toBe('done · 7 tokens · 1234 ms');
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
  });

  it('a double press cancels the previous stream before opening a new one', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const first = FakeEventSource.last();
    studio.__studio.generateStart();
    const second = FakeEventSource.last();
    expect(first).not.toBe(second);
    expect(first.closed).toBe(true);
    expect(second.closed).toBe(false);
  });

  it('restarting clears a pending batched-announce timer', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    // One token (< 5) schedules the 500ms flush timer.
    es.emit('message', { phase: 'generating', token_text: 'a', token_id: 1 });
    const cleared = vi.spyOn(global, 'clearTimeout');
    // Restart: the reset block must clearTimeout the pending announce timer.
    studio.__studio.generateStart();
    expect(cleared).toHaveBeenCalled();
  });

  it('a phase:done frame finishes the run', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.emit('message', { phase: 'generating', token_text: 'a', token_id: 1 });
    es.emit('message', { phase: 'done' });
    expect(es.closed).toBe(true);
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
  });

  it('compare mode shows "pending" until a ref_match flag arrives', () => {
    document.getElementById('gen-prompt').value = 'hi';
    document.getElementById('gen-compare').checked = true;
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    // Frame without ref_match while compare=on -> panel shows "pending".
    es.emit('message', { phase: 'generating', token_text: 'a', token_id: 1 });
    expect(document.getElementById('gen-last-ref').textContent).toBe('pending');
  });

  it('a network error (readyState OPEN) announces and finishes', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.readyState = 1;
    es.emit('error');
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('stream error');
    expect(document.getElementById('gen-start').disabled).toBe(false);
  });

  it('an error after close (readyState CLOSED) silently finishes', () => {
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    es.close();
    es.emit('error');
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
  });
});

describe('generateCancel', () => {
  it('closes the active stream and restores the idle button state', () => {
    vi.useFakeTimers();
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    const es = FakeEventSource.last();
    studio.__studio.generateCancel();
    expect(es.closed).toBe(true);
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
    vi.advanceTimersByTime(250);
    expect(document.getElementById('live-region').textContent).toContain('cancelled');
  });

  it('is a no-op when no stream is open', () => {
    expect(() => studio.__studio.generateCancel()).not.toThrow();
    expect(document.getElementById('gen-start').disabled).toBe(false);
  });
});

describe('generateStart — no EventSource / error branch', () => {
  it('falls back to idle UI when EventSource construction throws', () => {
    vi.useFakeTimers();
    function ThrowingES() { throw new Error('SSE unavailable'); }
    vi.stubGlobal('EventSource', ThrowingES);
    document.getElementById('gen-prompt').value = 'hi';
    studio.__studio.generateStart();
    vi.advanceTimersByTime(250);
    expect(document.getElementById('gen-start').disabled).toBe(false);
    expect(document.getElementById('gen-cancel').disabled).toBe(true);
    expect(document.getElementById('live-region').textContent).toContain('cannot open SSE');
  });
});
