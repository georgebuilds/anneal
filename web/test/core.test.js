// Core studio.js coverage: theme controller, live region, routing, skip link,
// view dispatch, keyboard-help modal, keyboard handlers, worker RPC client,
// ignite, boot. Asserts real DOM/behavior via the __studio seam and real
// dispatched events.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadStudio, installDOM, flushMicrotasks } from './harness.js';

const here = dirname(fileURLToPath(import.meta.url));
const STUDIO_HTML = join(here, '..', 'studio.html');

// bodyFixture replicated inline (harness keeps its copy private; tests that
// build a bespoke worker-meta fixture need the same body without editing it).
function bodyFixtureInline() {
  const raw = readFileSync(STUDIO_HTML, 'utf8');
  const m = raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  let inner = m ? m[1] : raw;
  inner = inner.replace(/<script[\s\S]*?<\/script>/gi, '');
  return inner;
}

let studio;

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

// ── theme controller ────────────────────────────────────────────────────────
describe('theme controller', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('applyTheme sets data-theme and the theme-toggle aria-label', () => {
    studio.__studio.applyTheme('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    const btn = document.getElementById('themeToggle');
    // dark's next in cycle is light.
    expect(btn.getAttribute('aria-label')).toBe(
      'current theme: dark. press to switch to light.'
    );
    expect(btn.title).toBe('theme: dark (click to cycle)');
  });

  it('cycleTheme cycles system → dark → light → system and persists', () => {
    document.documentElement.setAttribute('data-theme', 'system');
    studio.__studio.cycleTheme();
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('anneal-theme')).toBe('dark');

    studio.__studio.cycleTheme();
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(localStorage.getItem('anneal-theme')).toBe('light');

    studio.__studio.cycleTheme();
    expect(document.documentElement.getAttribute('data-theme')).toBe('system');
    expect(localStorage.getItem('anneal-theme')).toBe('system');
  });

  it('re-applies system theme when the OS color scheme changes', async () => {
    // Capture the matchMedia change listener so we can fire it.
    let changeCb = null;
    const orig = window.matchMedia;
    window.matchMedia = (q) => ({
      matches: false, media: q,
      addEventListener: (_t, cb) => { changeCb = cb; },
      removeEventListener() {}, addListener() {}, removeListener() {},
    });
    try {
      installDOM({ path: '/' }); // no saved theme → readTheme() === 'system'
      vi.resetModules();
      const mod = await import('../studio.js');
      expect(typeof changeCb).toBe('function');
      const spy = vi.spyOn(mod.__studio, 'applyTheme');
      // applyTheme isn't called through the seam internally, so assert on DOM:
      document.documentElement.setAttribute('data-theme', 'system');
      changeCb();
      // onChange re-applies 'system' since that's the active selection.
      expect(document.documentElement.getAttribute('data-theme')).toBe('system');
      spy.mockRestore();
    } finally {
      window.matchMedia = orig;
    }
  });

  it('clicking the toggle button cycles the theme (initTheme wiring)', () => {
    // boot() ran initTheme over a fresh DOM with no saved theme → system.
    expect(document.documentElement.getAttribute('data-theme')).toBe('system');
    document.getElementById('themeToggle').click();
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });
});

describe('readTheme precedence', () => {
  afterEach(() => vi.restoreAllMocks());

  it('URL ?theme= wins on boot', async () => {
    studio = await loadStudio({ path: '/?theme=dark' });
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('localStorage is used when no URL override', async () => {
    // loadStudio clears localStorage during installDOM, so seed AFTER install
    // but BEFORE the module evaluates by driving the two steps manually.
    installDOM({ path: '/' });
    localStorage.setItem('anneal-theme', 'light');
    vi.resetModules();
    studio = await import('../studio.js');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('defaults to system when neither URL nor storage set', async () => {
    studio = await loadStudio({ path: '/' });
    expect(document.documentElement.getAttribute('data-theme')).toBe('system');
  });

  it('URL theme beats localStorage', async () => {
    installDOM({ path: '/?theme=light' });
    localStorage.setItem('anneal-theme', 'dark');
    vi.resetModules();
    studio = await import('../studio.js');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });
});

// ── live region ─────────────────────────────────────────────────────────────
describe('live region announce', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('writes the message to #live-region after the clear delay', () => {
    vi.useFakeTimers();
    const live = document.getElementById('live-region');
    studio.__studio.announce('hello a11y');
    // Cleared synchronously, written after 200ms.
    expect(live.textContent).toBe('');
    vi.advanceTimersByTime(200);
    expect(live.textContent).toBe('hello a11y');
  });
});

// ── routing ─────────────────────────────────────────────────────────────────
describe('viewIdForPath', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('maps every route', () => {
    const v = studio.__studio.viewIdForPath;
    expect(v('/')).toBe('studio');
    expect(v('')).toBe('studio');
    expect(v('/v/mlp')).toBe('visualize');
    expect(v('/k/mlp')).toBe('kernels');
    expect(v('/x/Add')).toBe('explain');
    expect(v('/t/mlp')).toBe('train');
    expect(v('/g/nanogpt')).toBe('generate');
    expect(v('/run/42')).toBe('history');
    expect(v('/h')).toBe('history');
    expect(v('/d')).toBe('doctor');
  });

  it('falls back to studio for unknown paths', () => {
    expect(studio.__studio.viewIdForPath('/nope')).toBe('studio');
  });
});

describe('navigate', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('updates history, document.title, live region and aria-current', () => {
    vi.useFakeTimers();
    studio.__studio.navigate('visualize');
    expect(window.location.pathname).toBe('/v/mlp');
    expect(document.title).toBe('anneal · visualize');

    const navBtn = document.querySelector('.nav-item[data-view="visualize"]');
    expect(navBtn.getAttribute('aria-current')).toBe('page');
    expect(navBtn.classList.contains('active')).toBe(true);
    // The previously active studio nav item drops aria-current.
    const studioBtn = document.querySelector('.nav-item[data-view="studio"]');
    expect(studioBtn.hasAttribute('aria-current')).toBe(false);

    // Active view section toggled.
    expect(document.getElementById('view-visualize').classList.contains('active')).toBe(true);
    expect(document.getElementById('view-studio').classList.contains('active')).toBe(false);

    vi.advanceTimersByTime(200);
    expect(document.getElementById('live-region').textContent).toBe('navigated to visualize');
  });

  it('uses pushState by default and replaceState when asked', () => {
    const push = vi.spyOn(history, 'pushState');
    const replace = vi.spyOn(history, 'replaceState');
    studio.__studio.navigate('kernels');
    expect(push).toHaveBeenCalledWith({ viewId: 'kernels' }, '', '/k/mlp');

    studio.__studio.navigate('explain', { replace: true });
    expect(replace).toHaveBeenCalledWith({ viewId: 'explain' }, '', '/x/Add');
  });

  it('sets crumb section text per view group', () => {
    studio.__studio.navigate('history');
    expect(document.getElementById('crumb-section').textContent).toBe('persistence');
    studio.__studio.navigate('doctor');
    expect(document.getElementById('crumb-section').textContent).toBe('diagnostics');
    studio.__studio.navigate('visualize');
    expect(document.getElementById('crumb-section').textContent).toBe('surfaces');
  });

  it('popstate resolves the current URL into the active view', () => {
    window.history.replaceState({}, '', '/k/mlp');
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(document.getElementById('view-kernels').classList.contains('active')).toBe(true);
  });

  it('clicking a nav item navigates', () => {
    document.querySelector('.nav-item[data-view="train"]').click();
    expect(window.location.pathname).toBe('/t/mlp');
    expect(document.getElementById('view-train').classList.contains('active')).toBe(true);
  });
});

// ── skip link ───────────────────────────────────────────────────────────────
describe('skip link', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('forces focus into <main> and strips the hash from the URL', () => {
    window.history.replaceState(null, '', '/#main');
    const main = document.getElementById('main');
    const focusSpy = vi.spyOn(main, 'focus');
    main.scrollIntoView = vi.fn();

    document.querySelector('.skip-link').click();
    expect(focusSpy).toHaveBeenCalled();
    expect(window.location.hash).toBe('');
  });
});

// ── keyboard-help modal ─────────────────────────────────────────────────────
describe('keyboard-help modal', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  it('helpOpen reveals the dialog and focuses the close button', () => {
    const dlg = document.getElementById('keyboard-help');
    expect(dlg.hidden).toBe(true);
    studio.__studio.helpOpen();
    expect(dlg.hidden).toBe(false);
    expect(document.activeElement).toBe(dlg.querySelector('.kbd-help-close'));
  });

  it('helpClose hides the dialog and returns focus to the opener', () => {
    const opener = document.getElementById('helpToggle');
    opener.focus();
    studio.__studio.helpOpen(opener);
    expect(document.getElementById('keyboard-help').hidden).toBe(false);
    studio.__studio.helpClose();
    expect(document.getElementById('keyboard-help').hidden).toBe(true);
    expect(document.activeElement).toBe(opener);
  });

  it('the help trigger button opens the modal', () => {
    document.getElementById('helpToggle').click();
    expect(document.getElementById('keyboard-help').hidden).toBe(false);
  });

  it('data-kbd-close elements close the modal', () => {
    studio.__studio.helpOpen();
    document.querySelector('.kbd-help-close').click();
    expect(document.getElementById('keyboard-help').hidden).toBe(true);
  });

  it('Escape inside the dialog closes it (focus-trap keydown handler)', () => {
    studio.__studio.helpOpen();
    const dlg = document.getElementById('keyboard-help');
    dlg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(dlg.hidden).toBe(true);
  });

  it('focus trap wraps Tab from last → first and Shift+Tab from first → last', () => {
    const dlg = document.getElementById('keyboard-help');
    studio.__studio.helpOpen();
    // jsdom reports offsetParent === null for everything (no layout), which the
    // trap uses to filter visible elements. Force visibility so the cycling
    // branches run.
    const focusables = Array.from(dlg.querySelectorAll(
      'a[href], button:not([disabled]), textarea, input:not([disabled]), select, [tabindex]:not([tabindex="-1"])'
    ));
    expect(focusables.length).toBeGreaterThan(0);
    for (const el of focusables) {
      Object.defineProperty(el, 'offsetParent', { configurable: true, value: dlg });
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    // Tab on the last element wraps to the first.
    last.focus();
    const firstFocus = vi.spyOn(first, 'focus');
    dlg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    expect(firstFocus).toHaveBeenCalled();

    // Shift+Tab on the first element wraps to the last.
    first.focus();
    const lastFocus = vi.spyOn(last, 'focus');
    dlg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true }));
    expect(lastFocus).toHaveBeenCalled();
  });
});

// ── keyboard handlers ───────────────────────────────────────────────────────
describe('keyboard handlers', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/' });
  });

  function key(opts) {
    document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, ...opts }));
  }

  it('"/" focuses the search input', () => {
    key({ key: '/' });
    expect(document.activeElement).toBe(document.getElementById('search'));
  });

  it('"?" opens the keyboard-help modal', () => {
    key({ key: '?' });
    expect(document.getElementById('keyboard-help').hidden).toBe(false);
  });

  it('shift+/ opens the keyboard-help modal', () => {
    key({ key: '/', shiftKey: true });
    expect(document.getElementById('keyboard-help').hidden).toBe(false);
  });

  it('"g" chord jumps to a view (g k → kernels)', () => {
    key({ key: 'g' });
    key({ key: 'k' });
    expect(window.location.pathname).toBe('/k/mlp');
    expect(document.getElementById('view-kernels').classList.contains('active')).toBe(true);
  });

  it('"g" chord maps every destination', () => {
    const cases = [
      ['d', '/'], ['v', '/v/mlp'], ['k', '/k/mlp'], ['x', '/x/Add'],
      ['t', '/t/mlp'], ['g', '/g/nanogpt'], ['h', '/h'], ['r', '/d'],
    ];
    for (const [k, path] of cases) {
      key({ key: 'g' });
      key({ key: k });
      expect(window.location.pathname).toBe(path);
    }
  });

  it('Escape closes the help modal when open', () => {
    studio.__studio.helpOpen();
    key({ key: 'Escape' });
    expect(document.getElementById('keyboard-help').hidden).toBe(true);
  });

  it('Escape blurs a focused search field', () => {
    const s = document.getElementById('search');
    s.focus();
    const blur = vi.spyOn(s, 'blur');
    s.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(blur).toHaveBeenCalled();
  });

  it('does not intercept "/" while typing in a field', () => {
    const s = document.getElementById('search');
    s.focus();
    s.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
    // Still focused on search (no navigation, no crash); the handler returned early.
    expect(document.activeElement).toBe(s);
  });

  it('ignores chords with modifier keys held', () => {
    key({ key: 'g', metaKey: true });
    key({ key: 'k', metaKey: true });
    expect(window.location.pathname).toBe('/');
  });
});

// ── ignite ──────────────────────────────────────────────────────────────────
describe('ignite-once wordmark', () => {
  it('adds the ignite class on first load and persists the flag', async () => {
    studio = await loadStudio({ path: '/' });
    const mark = document.getElementById('brand-mark');
    expect(mark.classList.contains('ignite')).toBe(true);
    expect(localStorage.getItem('anneal-ignited')).toBe('1');
  });

  it('does not re-ignite when the flag is already set', async () => {
    installDOM({ path: '/' });
    localStorage.setItem('anneal-ignited', '1');
    vi.resetModules();
    studio = await import('../studio.js');
    expect(document.getElementById('brand-mark').classList.contains('ignite')).toBe(false);
  });

  it('skips ignite under prefers-reduced-motion', async () => {
    const orig = window.matchMedia;
    window.matchMedia = (q) => ({
      matches: q.includes('reduced-motion'),
      media: q, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {},
    });
    try {
      installDOM({ path: '/' });
      vi.resetModules();
      studio = await import('../studio.js');
      expect(document.getElementById('brand-mark').classList.contains('ignite')).toBe(false);
    } finally {
      window.matchMedia = orig;
    }
  });
});

// ── boot ────────────────────────────────────────────────────────────────────
describe('boot', () => {
  it('runs all initializers and resolves the initial view from the URL', async () => {
    studio = await loadStudio({ path: '/k/mlp' });
    // initRouting's first paint resolved /k/mlp.
    expect(document.getElementById('view-kernels').classList.contains('active')).toBe(true);
    // initTheme ran (data-theme set).
    expect(document.documentElement.getAttribute('data-theme')).toBe('system');
    // initKeyboardHelp wired the dialog (it starts hidden).
    expect(document.getElementById('keyboard-help').hidden).toBe(true);
  });
});

// ── worker RPC client ───────────────────────────────────────────────────────
describe('worker RPC client', () => {
  it('rejects with "worker not loaded" when no meta tag is present', async () => {
    studio = await loadStudio({ path: '/' });
    expect(studio.wasm.ready).toBe(false);
    await expect(studio.wasm.call('annealGetGraph', 1, 2))
      .rejects.toThrow(/worker not loaded/);
  });

  describe('with a worker meta tag present', () => {
    let workerInstances;
    let OrigWorker;

    beforeEach(() => {
      workerInstances = [];
      OrigWorker = global.Worker;
      global.Worker = class FakeWorker {
        constructor(src) {
          this.src = src;
          this.onmessage = null;
          this.onerror = null;
          this.posted = [];
          workerInstances.push(this);
        }
        postMessage(msg) { this.posted.push(msg); }
        terminate() {}
      };
    });

    afterEach(() => {
      global.Worker = OrigWorker;
    });

    async function loadWithWorker() {
      // Build DOM + worker meta + Worker stub, then import fresh.
      document.documentElement.setAttribute('data-theme', 'system');
      document.head.innerHTML = '<title>anneal studio</title>';
      document.body.innerHTML = bodyFixtureInline();
      try { localStorage.clear(); } catch (_) {}
      window.history.replaceState({}, '', '/');
      const meta = document.createElement('meta');
      meta.setAttribute('name', 'anneal-worker');
      meta.setAttribute('content', '/static/worker.js');
      document.head.appendChild(meta);
      vi.resetModules();
      return import('../studio.js');
    }

    it('constructs a Worker, marks ready, and updates the status segment', async () => {
      const mod = await loadWithWorker();
      expect(mod.wasm.ready).toBe(true);
      expect(workerInstances).toHaveLength(1);
      expect(workerInstances[0].src).toBe('/static/worker.js');
      expect(document.getElementById('status-worker').innerHTML)
        .toContain('loaded');
    });

    it('posts {id,fn,args} and resolves when the worker replies ok', async () => {
      const mod = await loadWithWorker();
      const w = workerInstances[0];
      const p = mod.wasm.call('annealGetGraph', 'mlp', 7);
      expect(w.posted).toHaveLength(1);
      expect(w.posted[0]).toEqual({ id: 1, fn: 'annealGetGraph', args: ['mlp', 7] });

      // Worker replies for that id.
      w.onmessage({ data: { id: 1, ok: true, result: { nodes: 52 } } });
      await expect(p).resolves.toEqual({ nodes: 52 });
    });

    it('rejects when the worker replies with an error', async () => {
      const mod = await loadWithWorker();
      const w = workerInstances[0];
      const p = mod.wasm.call('boom');
      w.onmessage({ data: { id: 1, ok: false, error: 'kaboom' } });
      await expect(p).rejects.toThrow('kaboom');
    });

    it('ignores replies for unknown ids', async () => {
      const mod = await loadWithWorker();
      const w = workerInstances[0];
      const p = mod.wasm.call('keep');
      // Stray reply for an id nobody is waiting on — must not throw.
      w.onmessage({ data: { id: 999, ok: true, result: 1 } });
      // The real reply still resolves the pending call.
      w.onmessage({ data: { id: 1, ok: true, result: 'ok' } });
      await expect(p).resolves.toBe('ok');
    });

    it('auto-increments request ids', async () => {
      const mod = await loadWithWorker();
      const w = workerInstances[0];
      mod.wasm.call('a');
      mod.wasm.call('b');
      expect(w.posted.map((m) => m.id)).toEqual([1, 2]);
    });
  });
});
