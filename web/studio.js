// anneal studio — main thread ES module.
//
// Responsibilities (DESIGN.md §3, §5, §7, §11; spec §10; web/A11Y.md):
//   1. History API routing for every studio view (no hash routing).
//   2. Theme controller: system | dark | light cycle, persisted, with a live
//      matchMedia listener so OS theme changes apply without a page reload.
//   3. View renderer dispatch — stubs in W0, real renderers land in W1+.
//   4. Keyboard handlers: `/` focuses search; `g <dest>` chord jumps view;
//      `?` opens the keyboard-help modal; `Esc` closes it.
//   5. Worker RPC client — gated behind a <meta name="anneal-worker"> tag so
//      W0 (no anneal.wasm shipped yet) does not 404. Once a real WASM build
//      lands, flip the meta tag on and `wasm.call(fn, ...args)` lights up.
//   6. Ignite-once wordmark animation on first page load only.
//   7. Accessibility plumbing (the load-bearing pieces; see web/A11Y.md):
//      - skip link forces focus into <main>
//      - document.title updates on route change
//      - polite live region announces "navigated to {view}"
//      - aria-current="page" on the active nav button
//      - dynamic aria-label on the theme toggle (current state described)
//      - focus trap inside the keyboard-help modal
//
// This file is intentionally small and greppable. Each section under one
// header. No third-party imports. No bundler. Pure browser APIs.

'use strict';

// ── theme controller (DESIGN.md §5) ───────────────────────────────────────
// Cycle order: system → dark → light → system. Default: system. Persisted in
// localStorage['anneal-theme']. When `system` is selected, a matchMedia
// listener re-renders on OS theme change without a reload (spec §10 fix).
//
// Accessibility: the toggle's aria-label describes the CURRENT theme and the
// NEXT one a press will switch to (so screen-reader users hear "press to
// switch to dark" instead of an opaque "◑" character).

const THEMES = ['system', 'dark', 'light'];

function readTheme() {
  // URL ?theme=… wins (screenshot harness convention; same as viz).
  try {
    const u = new URL(window.location.href);
    const t = u.searchParams.get('theme');
    if (t && THEMES.includes(t)) return t;
  } catch (_) { /* URL parsing on opaque origins */ }
  try {
    const saved = localStorage.getItem('anneal-theme');
    if (saved && THEMES.includes(saved)) return saved;
  } catch (_) { /* private mode */ }
  return 'system';
}

function nextTheme(cur) {
  const idx = THEMES.indexOf(cur);
  return THEMES[(idx + 1) % THEMES.length];
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const btn = document.getElementById('themeToggle');
  if (btn) {
    const next = nextTheme(theme);
    btn.title = 'theme: ' + theme + ' (click to cycle)';
    // aria-label update on theme cycle — describes current state AND the
    // next state (so a screen-reader user always knows what a press does).
    btn.setAttribute(
      'aria-label',
      'current theme: ' + theme + '. press to switch to ' + next + '.'
    );
  }
}

function cycleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'system';
  const next = nextTheme(cur);
  applyTheme(next);
  try { localStorage.setItem('anneal-theme', next); } catch (_) {}
}

function initTheme() {
  applyTheme(readTheme());
  const btn = document.getElementById('themeToggle');
  if (btn) btn.addEventListener('click', cycleTheme);

  // Live OS theme change while `system` is selected (DESIGN.md §5; spec §10).
  // The matchMedia listener forces a repaint of the `:root[data-theme="system"]`
  // tokens by toggling the data-theme attribute through the same value.
  if (window.matchMedia) {
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      if (readTheme() === 'system') applyTheme('system');
    };
    if (mql.addEventListener) mql.addEventListener('change', onChange);
    else if (mql.addListener) mql.addListener(onChange); // Safari < 14
  }
}

// ── live region (web/A11Y.md) ─────────────────────────────────────────────
// Announces SPA navigation and (later) SSE-driven content updates to screen
// readers without stealing focus. aria-live="polite" + aria-atomic="false"
// means only newly inserted text is read. We clear-then-write with a tiny
// delay because some screen readers ignore writes when the region is empty.

function announce(message) {
  const live = document.getElementById('live-region');
  if (!live) return;
  // Clear first so identical consecutive messages still announce.
  live.textContent = '';
  setTimeout(() => { live.textContent = message; }, 200);
}

// ── routing (DESIGN.md §7) ────────────────────────────────────────────────
// History API only. Patterns:
//   /                       → studio
//   /v/:model               → visualize  (q: stage, node)
//   /k/:model               → kernels    (q: kernel)
//   /x/:op                  → explain
//   /t/:model               → train
//   /g/:model               → generate
//   /run/:id                → history (single run)
//   /h                      → history (table)
//   /d                      → doctor
//
// Each entry resolves to a view id; the renderer table dispatches by id.

const ROUTES = [
  { test: (p) => p === '/' || p === '',                view: 'studio'    },
  { test: (p) => p.startsWith('/v/'),                  view: 'visualize' },
  { test: (p) => p.startsWith('/k/'),                  view: 'kernels'   },
  { test: (p) => p.startsWith('/x/'),                  view: 'explain'   },
  { test: (p) => p.startsWith('/t/'),                  view: 'train'     },
  { test: (p) => p.startsWith('/g/'),                  view: 'generate'  },
  { test: (p) => p.startsWith('/run/') || p === '/h', view: 'history'   },
  { test: (p) => p === '/d',                           view: 'doctor'    },
];

const VIEW_TO_PATH = {
  studio:    '/',
  visualize: '/v/mlp',
  kernels:   '/k/mlp',
  explain:   '/x/matmul',
  train:     '/t/mlp',
  generate:  '/g/nanogpt',
  history:   '/h',
  doctor:    '/d',
};

const VIEW_TITLE = {
  studio:    'studio',
  visualize: 'visualize',
  kernels:   'kernels',
  explain:   'explain',
  train:     'train',
  generate:  'generate',
  history:   'history',
  doctor:    'doctor',
};

function viewIdForPath(pathname) {
  for (const r of ROUTES) if (r.test(pathname)) return r.view;
  return 'studio';
}

function setActiveView(viewId) {
  document.querySelectorAll('.view').forEach((el) => {
    el.classList.toggle('active', el.id === 'view-' + viewId);
  });
  // Active nav button: visual class AND aria-current="page" so screen readers
  // know which view is current. Inactive items get aria-current removed.
  document.querySelectorAll('.nav-item').forEach((el) => {
    const isActive = el.dataset.view === viewId;
    el.classList.toggle('active', isActive);
    if (isActive) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  });
  const crumb = document.getElementById('crumb-current');
  if (crumb) crumb.textContent = viewId === 'studio' ? '' : '· ' + viewId;
  const section = document.getElementById('crumb-section');
  if (section) {
    section.textContent =
      viewId === 'history' ? 'persistence'
      : viewId === 'doctor' ? 'diagnostics'
      : 'surfaces';
  }
  // document.title updates on route change so the tab label tracks the SPA
  // view (a11y + browser history clarity).
  const label = VIEW_TITLE[viewId] || viewId;
  document.title = 'anneal · ' + label;
  // Polite live-region announcement for SPA navigation. Some screen-reader
  // users miss History-API transitions otherwise.
  announce('navigated to ' + label);
  // Renderer dispatch. W0: each view has a stub mount already in HTML.
  // W1+: the renderer hydrates the mount with live content.
  const renderer = RENDERERS[viewId];
  if (renderer) {
    try { renderer(); } catch (e) {
      console.error('renderer error for view', viewId, e);
    }
  }
}

function navigate(viewId, { replace = false } = {}) {
  const path = VIEW_TO_PATH[viewId] || '/';
  if (window.location.pathname !== path) {
    if (replace) history.replaceState({ viewId }, '', path);
    else         history.pushState  ({ viewId }, '', path);
  }
  setActiveView(viewId);
}

function initRouting() {
  // Wire nav items to navigate().
  document.querySelectorAll('.nav-item').forEach((el) => {
    el.addEventListener('click', () => navigate(el.dataset.view));
  });
  // popstate handles back/forward.
  window.addEventListener('popstate', () => {
    setActiveView(viewIdForPath(window.location.pathname));
  });
  // First paint: resolve current URL.
  setActiveView(viewIdForPath(window.location.pathname));
}

// ── skip link (web/A11Y.md) ───────────────────────────────────────────────
// The anchor `<a href="#main">` already works without JS. The handler below
// is a small enhancement that forces keyboard focus into <main> (the anchor
// alone would only scroll, not move focus, in some older browsers).

function initSkipLink() {
  const skip = document.querySelector('.skip-link');
  const main = document.getElementById('main');
  if (!skip || !main) return;
  skip.addEventListener('click', (e) => {
    e.preventDefault();
    main.focus();
    main.scrollIntoView({ block: 'start' });
    // Restore history so the URL hash doesn't linger.
    if (window.location.hash) {
      history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  });
}

// ── view renderer dispatch ────────────────────────────────────────────────
// Each renderer hydrates one view's mount point. W0 renderers are no-ops;
// they exist so the dispatch machinery is real and the W1+ renderers can
// drop in by assignment.

const RENDERERS = {
  studio:    function renderStudio()    { /* W1+ */ },
  visualize: function renderVisualize() { /* W2 */ },
  kernels:   function renderKernels()   { renderKernelsView(); },
  explain:   function renderExplain()   { /* W4 */ },
  train:     function renderTrain()     { /* W5 */ },
  generate:  function renderGenerate()  { /* W6 */ },
  history:   function renderHistory()   { /* W8 */ },
  doctor:    function renderDoctor()    { /* W7 */ },
};

// ── keyboard-help modal (web/A11Y.md) ─────────────────────────────────────
// Opens on `?` (or shift+/), the help button click, or programmatic open.
// Implements:
//   - focus trap (Tab cycles inside)
//   - Escape closes, return focus to the opener
//   - aria-modal=true on the dialog (set in HTML)
//   - background scroll is NOT locked (modal is small enough)

const FOCUSABLE = [
  'a[href]', 'button:not([disabled])', 'textarea', 'input:not([disabled])',
  'select', '[tabindex]:not([tabindex="-1"])',
].join(',');

let helpOpener = null;

function helpOpen(opener) {
  const dlg = document.getElementById('keyboard-help');
  if (!dlg) return;
  helpOpener = opener || document.activeElement;
  dlg.hidden = false;
  // Move focus to the close button (first interactive thing inside).
  const close = dlg.querySelector('.kbd-help-close');
  if (close) close.focus();
}

function helpClose() {
  const dlg = document.getElementById('keyboard-help');
  if (!dlg) return;
  dlg.hidden = true;
  // Return focus to the element that opened it (WCAG 2.4.3 / 2.4.11).
  if (helpOpener && typeof helpOpener.focus === 'function') {
    helpOpener.focus();
  }
  helpOpener = null;
}

function initKeyboardHelp() {
  const dlg = document.getElementById('keyboard-help');
  const trigger = document.getElementById('helpToggle');
  if (!dlg) return;

  if (trigger) trigger.addEventListener('click', () => helpOpen(trigger));

  // Close affordances: any element marked data-kbd-close.
  dlg.querySelectorAll('[data-kbd-close]').forEach((el) => {
    el.addEventListener('click', helpClose);
  });

  // Focus trap. While the dialog is open, Tab / Shift-Tab cycle within it.
  dlg.addEventListener('keydown', (e) => {
    if (dlg.hidden) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      helpClose();
      return;
    }
    if (e.key !== 'Tab') return;
    const focusables = Array.from(dlg.querySelectorAll(FOCUSABLE))
      .filter((el) => !el.hasAttribute('disabled') && el.offsetParent !== null);
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  });
}

// ── keyboard (DESIGN.md §7; spec §10) ─────────────────────────────────────
// `/` focuses the search input. `g <key>` chord jumps to a view within 1s.
// `?` opens the keyboard-help modal.
// Chord destinations:
//   g d → studio (home / "dashboard")
//   g v → visualize
//   g k → kernels
//   g x → explain
//   g t → train
//   g g → generate
//   g h → history
//   g r → doctor (think "diagnose / repair"; d is taken by the home view)
//
// The chord destinations are also the cycle sequence: system → dark → light
// is the theme cycle, surfaced as a literal comment so tests can pin the
// contract (spec §10 test hook).

const KEYMAP = {
  d: 'studio',
  v: 'visualize',
  k: 'kernels',
  x: 'explain',
  t: 'train',
  g: 'generate',
  h: 'history',
  r: 'doctor',
};

function initKeyboard() {
  let leader = false;
  let leaderTimer = null;

  const armLeader = () => {
    leader = true;
    if (leaderTimer) clearTimeout(leaderTimer);
    leaderTimer = setTimeout(() => { leader = false; }, 1000);
  };
  const disarmLeader = () => {
    leader = false;
    if (leaderTimer) clearTimeout(leaderTimer);
  };

  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    const inField = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

    // Escape always: blur search, close help modal.
    if (e.key === 'Escape') {
      const dlg = document.getElementById('keyboard-help');
      if (dlg && !dlg.hidden) { e.preventDefault(); helpClose(); return; }
      if (inField && e.target && typeof e.target.blur === 'function') {
        e.target.blur();
      }
      return;
    }

    // Never intercept while the user is typing (other than Escape above).
    if (inField) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    // `?` (shift+/) opens the keyboard-help modal.
    if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
      e.preventDefault();
      helpOpen(document.activeElement);
      return;
    }

    if (e.key === '/') {
      e.preventDefault();
      const s = document.getElementById('search');
      if (s) s.focus();
      return;
    }
    if (e.key === 'g' && !leader) { armLeader(); return; }
    if (leader) {
      const dest = KEYMAP[e.key];
      if (dest) { e.preventDefault(); navigate(dest); }
      disarmLeader();
    }
  });
}

// ── worker RPC client (spec §10 / §3 of anneal_web_spec) ──────────────────
// Gated behind <meta name="anneal-worker" content="/static/worker.js">.
// When the meta tag is absent (W0), no Worker is constructed and
// wasm.call() rejects with a clear "worker not loaded" message.
//
// Protocol: { id, fn, args } → { id, ok, result|error }. Promise-based.
// Auto-incrementing id; pending map cleared on each response.

function makeWorkerRPC() {
  const meta = document.querySelector('meta[name="anneal-worker"]');
  const src = meta && meta.getAttribute('content');

  if (!src) {
    // No worker configured for this build. wasm.call() always rejects.
    return {
      ready: false,
      call: function call(fn, ..._args) {
        return Promise.reject(new Error(
          'wasm worker not loaded: add <meta name="anneal-worker" ' +
          'content="/static/worker.js"> to studio.html when anneal.wasm is shipped'
        ));
      },
    };
  }

  const worker = new Worker(src);
  const pending = new Map();
  let nextId = 1;

  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) entry.resolve(msg.result);
    else entry.reject(new Error(msg.error || 'worker error'));
  };
  worker.onerror = (ev) => {
    console.error('worker fatal:', ev.message);
    // Surface but do not reject pending; the message handler still owns them.
  };

  const statusEl = document.getElementById('status-worker');
  if (statusEl) statusEl.innerHTML = 'worker: <span class="v">loaded</span>';

  return {
    ready: true,
    call: function call(fn, ...args) {
      return new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, fn, args });
      });
    },
  };
}

export const wasm = makeWorkerRPC();

// ── ignite-once wordmark (DESIGN.md §3.4 item 1) ──────────────────────────
// First page load per browser session animates the brand mark; subsequent
// loads (including refresh in the same tab) do not re-ignite. Persisted in
// localStorage under 'anneal-ignited'.
//
// Respects prefers-reduced-motion: when reduced motion is requested, we
// skip the ignite class entirely (CSS would already neutralise the animation
// but the JS skip is a belt-and-braces guarantee).

function initIgnite() {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }
  let already = false;
  try { already = localStorage.getItem('anneal-ignited') === '1'; } catch (_) {}
  const mark = document.getElementById('brand-mark');
  if (!mark) return;
  if (!already) {
    mark.classList.add('ignite');
    try { localStorage.setItem('anneal-ignited', '1'); } catch (_) {}
  }
}

// ── WGSL tokenizer (W2) ───────────────────────────────────────────────────
// Single-pass hand-rolled scanner. Walks the WGSL text once and emits a list
// of {start, end, kind} spans the kernels-view renderer wraps into <span>
// elements. No regex backtracking, no JS deps.
//
// Token kinds emitted:
//   keyword   — control flow + storage classes (fn, var, let, for, if, …)
//   type      — primitive scalar + vector + matrix types
//   builtin   — main, gid, lid, wid, the @builtin identifiers
//   attribute — @compute, @workgroup_size, @binding, @group, @builtin, …
//   number    — int, float, hex literals (123, 12.5, 0xCAFE, 64u, 1.0f, ...)
//   string    — "..." literals (rare in WGSL but tokenized for safety)
//   comment   — // line comments AND /* block comments */
//   ident     — everything else that starts with a letter/underscore
//   punct     — every other single-char token (operators, braces, semi, …)
//
// Reduced-motion + forced-colors are CSS concerns; the tokenizer is pure
// structure. Token boundaries are byte-offset into the input string.

const WGSL_KEYWORDS = new Set([
  'fn', 'var', 'let', 'for', 'while', 'if', 'else', 'return', 'break',
  'continue', 'struct', 'enable', 'override', 'const', 'switch', 'case',
  'default', 'discard', 'loop', 'true', 'false', 'storage', 'uniform',
  'read', 'read_write', 'workgroup', 'function', 'private',
]);

const WGSL_TYPES = new Set([
  'f32', 'i32', 'u32', 'f16', 'bool',
  'vec2', 'vec3', 'vec4',
  'mat2x2', 'mat2x3', 'mat2x4',
  'mat3x2', 'mat3x3', 'mat3x4',
  'mat4x2', 'mat4x3', 'mat4x4',
  'array', 'atomic', 'ptr',
  'texture_1d', 'texture_2d', 'texture_3d', 'texture_cube',
  'sampler', 'sampler_comparison',
]);

// Identifiers WGSL builds in (@builtin parameter names + the kernel entry
// point). Stays small; everything else is treated as a regular identifier.
const WGSL_BUILTINS = new Set([
  'main', 'gid', 'lid', 'wid',
  'global_invocation_id', 'local_invocation_id', 'workgroup_id',
  'num_workgroups', 'local_invocation_index', 'vertex_index',
  'instance_index', 'position', 'front_facing', 'frag_depth',
  'sample_index', 'sample_mask',
  'bitcast', 'select', 'min', 'max', 'abs', 'clamp',
  'workgroupBarrier', 'storageBarrier',
]);

function tokenizeWGSL(text) {
  const spans = [];
  const N = text.length;
  let i = 0;

  const isLetter = (c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_';
  const isDigit  = (c) => c >= '0' && c <= '9';
  const isAlnum  = (c) => isLetter(c) || isDigit(c);
  const isHex    = (c) => isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');

  while (i < N) {
    const c = text[i];

    // Whitespace — skipped (not emitted as a token).
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      i++;
      continue;
    }

    // Line comment.
    if (c === '/' && text[i + 1] === '/') {
      const start = i;
      while (i < N && text[i] !== '\n') i++;
      spans.push({ start, end: i, kind: 'comment' });
      continue;
    }
    // Block comment.
    if (c === '/' && text[i + 1] === '*') {
      const start = i;
      i += 2;
      while (i < N - 1 && !(text[i] === '*' && text[i + 1] === '/')) i++;
      if (i < N) i += 2;
      spans.push({ start, end: i, kind: 'comment' });
      continue;
    }
    // String.
    if (c === '"') {
      const start = i;
      i++;
      while (i < N && text[i] !== '"') {
        if (text[i] === '\\' && i + 1 < N) i++;
        i++;
      }
      if (i < N) i++;
      spans.push({ start, end: i, kind: 'string' });
      continue;
    }
    // Attribute (@compute, @binding, @group, @builtin, @workgroup_size).
    if (c === '@') {
      const start = i;
      i++;
      while (i < N && (isAlnum(text[i]) || text[i] === '_')) i++;
      spans.push({ start, end: i, kind: 'attribute' });
      continue;
    }
    // Number. Handles decimal, hex (0x...), suffix u/f.
    if (isDigit(c) || (c === '.' && isDigit(text[i + 1]))) {
      const start = i;
      if (c === '0' && (text[i + 1] === 'x' || text[i + 1] === 'X')) {
        i += 2;
        while (i < N && isHex(text[i])) i++;
      } else {
        while (i < N && isDigit(text[i])) i++;
        if (i < N && text[i] === '.') {
          i++;
          while (i < N && isDigit(text[i])) i++;
        }
        if (i < N && (text[i] === 'e' || text[i] === 'E')) {
          i++;
          if (i < N && (text[i] === '+' || text[i] === '-')) i++;
          while (i < N && isDigit(text[i])) i++;
        }
      }
      // Numeric suffix (u, f, h, i).
      if (i < N && (text[i] === 'u' || text[i] === 'f' || text[i] === 'h' || text[i] === 'i')) i++;
      spans.push({ start, end: i, kind: 'number' });
      continue;
    }
    // Identifier or keyword.
    if (isLetter(c)) {
      const start = i;
      while (i < N && isAlnum(text[i])) i++;
      const word = text.slice(start, i);
      let kind = 'ident';
      if (WGSL_KEYWORDS.has(word))      kind = 'keyword';
      else if (WGSL_TYPES.has(word))    kind = 'type';
      else if (WGSL_BUILTINS.has(word)) kind = 'builtin';
      spans.push({ start, end: i, kind });
      continue;
    }
    // Single-char punctuation (operators, braces, commas, semis, …).
    spans.push({ start: i, end: i + 1, kind: 'punct' });
    i++;
  }
  return spans;
}

// ── kernels view (W2) ─────────────────────────────────────────────────────
// State held outside the renderer so navigation back to the kernels view
// preserves the selection (per spec deep-link contract).

const _kernelsState = {
  model: null,        // currently-loaded model name
  set: null,          // KernelSet JSON (the annealGetKernels payload)
  selectedIdx: -1,    // index into set.kernels
  loading: false,
  error: null,
};

// modelFromPath extracts the model name from /k/<model>. Defaults to 'mlp'.
function modelFromPath(pathname) {
  const m = /^\/k\/([^/?#]+)/.exec(pathname);
  return m ? decodeURIComponent(m[1]) : 'mlp';
}

// kernelIdFromQuery reads ?kernel=K3 off the URL. Returns null if absent.
function kernelIdFromQuery() {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get('kernel');
  } catch (_) { return null; }
}

async function renderKernelsView() {
  const listEl = document.getElementById('kernel-list-items');
  if (!listEl) return;

  const model = modelFromPath(window.location.pathname);
  const wantKernel = kernelIdFromQuery();

  // If we already have data for this model, just (re)render and pick
  // the kernel from the URL (deep-link case).
  if (_kernelsState.model === model && _kernelsState.set) {
    let idx = _kernelsState.selectedIdx;
    if (wantKernel) {
      const found = _kernelsState.set.kernels.findIndex((k) => k.id === wantKernel);
      if (found >= 0) idx = found;
    }
    drawKernelList(_kernelsState.set, idx);
    drawKernelDetail(_kernelsState.set, idx);
    return;
  }

  // Fresh load. Show a placeholder while we fetch from the worker.
  _kernelsState.model = model;
  _kernelsState.set = null;
  _kernelsState.selectedIdx = -1;
  _kernelsState.loading = true;
  _kernelsState.error = null;
  listEl.innerHTML = '<li class="kernel-list-loading" aria-live="polite">loading kernels…</li>';

  let raw;
  try {
    raw = await wasm.call('annealGetKernels', model);
  } catch (e) {
    _kernelsState.loading = false;
    _kernelsState.error = (e && e.message) || String(e);
    listEl.innerHTML = '';
    const li = document.createElement('li');
    li.className = 'kernel-list-error';
    li.textContent = 'wasm not loaded — build anneal.wasm to populate this view';
    listEl.appendChild(li);
    announce('kernels view: wasm not loaded');
    return;
  }

  let set;
  try {
    set = JSON.parse(raw);
  } catch (e) {
    _kernelsState.loading = false;
    _kernelsState.error = 'invalid JSON from annealGetKernels';
    listEl.innerHTML = '<li class="kernel-list-error">kernels view: invalid JSON from compiler</li>';
    return;
  }
  if (set && set.error) {
    _kernelsState.loading = false;
    _kernelsState.error = set.error;
    listEl.innerHTML = '';
    const li = document.createElement('li');
    li.className = 'kernel-list-error';
    li.textContent = 'compiler error: ' + set.error;
    listEl.appendChild(li);
    return;
  }

  _kernelsState.loading = false;
  _kernelsState.set = set;

  let initialIdx = 0;
  if (wantKernel) {
    const found = set.kernels.findIndex((k) => k.id === wantKernel);
    if (found >= 0) initialIdx = found;
  }
  _kernelsState.selectedIdx = initialIdx;
  drawKernelList(set, initialIdx);
  drawKernelDetail(set, initialIdx);
}

function drawKernelList(set, selectedIdx) {
  const listEl = document.getElementById('kernel-list-items');
  if (!listEl) return;
  listEl.innerHTML = '';
  if (!set || !set.kernels || set.kernels.length === 0) {
    const li = document.createElement('li');
    li.className = 'kernel-list-empty';
    li.textContent = 'no kernels';
    listEl.appendChild(li);
    return;
  }
  for (let i = 0; i < set.kernels.length; i++) {
    const k = set.kernels[i];
    const li = document.createElement('li');
    li.className = 'kernel-list-item';
    li.setAttribute('role', 'option');
    li.id = 'k-opt-' + k.id;
    li.dataset.idx = String(i);
    const sel = (i === selectedIdx);
    li.setAttribute('aria-selected', sel ? 'true' : 'false');
    // Two channels (DD1): the active id is bold + has a left border (via
    // CSS [aria-selected="true"]); the id text itself is the label.
    const idSpan = document.createElement('span');
    idSpan.className = 'k-id';
    idSpan.textContent = k.id;
    const metaSpan = document.createElement('span');
    metaSpan.className = 'k-meta';
    const shape = Array.isArray(k.shape) && k.shape.length ? '[' + k.shape.join(',') + ']' : '';
    metaSpan.textContent =
      k.op_count + ' ops · ' + k.buffers_in + ' in / ' + k.buffers_out + ' out' +
      (shape ? ' · ' + shape : '');
    li.appendChild(idSpan);
    li.appendChild(metaSpan);
    li.addEventListener('click', () => selectKernel(i, { focus: false, fromKeyboard: false }));
    listEl.appendChild(li);
  }
  // aria-activedescendant tracks the keyboard-active item without moving
  // focus from the listbox container.
  if (selectedIdx >= 0 && selectedIdx < set.kernels.length) {
    listEl.setAttribute('aria-activedescendant', 'k-opt-' + set.kernels[selectedIdx].id);
  }
}

function drawKernelDetail(set, idx) {
  const idEl     = document.getElementById('k-id');
  const shapeEl  = document.getElementById('k-shape');
  const countsEl = document.getElementById('k-counts');
  const preEl    = document.getElementById('k-wgsl');
  if (!idEl || !shapeEl || !countsEl || !preEl) return;
  if (!set || !set.kernels || idx < 0 || idx >= set.kernels.length) {
    idEl.textContent = '';
    shapeEl.textContent = '';
    countsEl.textContent = '';
    preEl.innerHTML = '';
    return;
  }
  const k = set.kernels[idx];
  idEl.textContent = k.id;
  shapeEl.textContent = (Array.isArray(k.shape) && k.shape.length)
    ? '[' + k.shape.join(',') + ']' : '';
  countsEl.textContent = k.op_count + ' ops · ' + k.buffers_in + ' in / ' + k.buffers_out + ' out';
  preEl.innerHTML = '';
  preEl.appendChild(renderWGSL(k.wgsl, k.fusion_spans));
}

// renderWGSL produces a DocumentFragment that the WGSL <pre> consumes. Each
// source line is wrapped in a .wgsl-line span; the leading edge of each
// fusion span carries a gutter label (fwd / bwd / fused) coloured per
// DESIGN.md §1 (teal / ember / gold).
function renderWGSL(text, fusionSpans) {
  const frag = document.createDocumentFragment();
  if (!text) return frag;
  const tokens = tokenizeWGSL(text);

  // Split text into physical lines for the gutter mapping (1-based).
  const lines = text.split('\n');
  // Token-offset → which (1-based) line it belongs to.
  // We compute line breakpoints once for the per-token line lookup.
  const lineStarts = [0];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') lineStarts.push(i + 1);
  }
  function lineOf(off) {
    // Binary search; lineStarts[k] <= off < lineStarts[k+1].
    let lo = 0, hi = lineStarts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (lineStarts[mid] <= off) lo = mid; else hi = mid - 1;
    }
    return lo + 1; // 1-based
  }

  // Build the per-line label map up-front (line → label or null).
  const lineLabel = new Array(lines.length + 1).fill(null);
  if (Array.isArray(fusionSpans)) {
    for (const sp of fusionSpans) {
      // Stamp the gutter on the first line of every span.
      if (sp.start_line >= 1 && sp.start_line <= lines.length) {
        lineLabel[sp.start_line] = sp.label;
      }
    }
  }

  // Group tokens by line.
  const tokensByLine = {};
  for (const tok of tokens) {
    const ln = lineOf(tok.start);
    if (!tokensByLine[ln]) tokensByLine[ln] = [];
    tokensByLine[ln].push(tok);
  }

  for (let ln = 1; ln <= lines.length; ln++) {
    const line = lines[ln - 1];
    const lineEl = document.createElement('span');
    lineEl.className = 'wgsl-line';

    // Gutter (fusion-span label), if this line is the start of a span.
    if (lineLabel[ln]) {
      const g = document.createElement('span');
      g.className = 'gutter ' + lineLabel[ln];
      g.textContent = lineLabel[ln];
      g.setAttribute('aria-hidden', 'true');
      lineEl.appendChild(g);
    }

    // Token spans within the line. We translate token (start,end) into the
    // line-local substring; non-tokenized gaps are emitted as plain text.
    const startOfLine = lineStarts[ln - 1];
    const endOfLine = (ln < lineStarts.length ? lineStarts[ln] - 1 : text.length);
    const ts = tokensByLine[ln] || [];
    let cursor = startOfLine;
    for (const tok of ts) {
      if (tok.start > cursor) {
        lineEl.appendChild(document.createTextNode(text.slice(cursor, tok.start)));
      }
      const sp = document.createElement('span');
      sp.className = 'tk-' + tok.kind;
      sp.textContent = text.slice(tok.start, Math.min(tok.end, endOfLine));
      lineEl.appendChild(sp);
      cursor = tok.end;
    }
    if (cursor < endOfLine) {
      lineEl.appendChild(document.createTextNode(text.slice(cursor, endOfLine)));
    }
    lineEl.appendChild(document.createTextNode('\n'));
    frag.appendChild(lineEl);
  }
  return frag;
}

// selectKernel updates state + URL + DOM. fromKeyboard=true scrolls the new
// active item into view (so arrow-key navigation in long kernel lists is
// usable without a mouse).
function selectKernel(idx, { focus = false, fromKeyboard = false } = {}) {
  const set = _kernelsState.set;
  if (!set || idx < 0 || idx >= set.kernels.length) return;
  _kernelsState.selectedIdx = idx;
  const k = set.kernels[idx];

  // Push a new history entry so the deep-link reflects the active kernel.
  // Use pushState so back/forward navigates between kernels.
  try {
    const url = '/k/' + encodeURIComponent(_kernelsState.model) +
                '?kernel=' + encodeURIComponent(k.id);
    if (window.location.pathname + window.location.search !== url) {
      history.pushState({ viewId: 'kernels', kernel: k.id }, '', url);
    }
  } catch (_) {}

  drawKernelList(set, idx);
  drawKernelDetail(set, idx);
  announce('kernel ' + k.id + ' selected');

  if (fromKeyboard) {
    const el = document.getElementById('k-opt-' + k.id);
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ block: 'nearest' });
    }
  }
  if (focus) {
    const el = document.getElementById('k-opt-' + k.id);
    if (el && typeof el.focus === 'function') el.focus();
  }
}

// Keyboard navigation on the kernel list. ArrowUp/Down move selection;
// Home/End jump to the ends; Enter activates (re-announces the current
// kernel). Escape moves focus back to the nav rail (per a11y plan).
function initKernelsKeyboard() {
  const listEl = document.getElementById('kernel-list-items');
  if (!listEl) return;
  listEl.addEventListener('keydown', (e) => {
    const set = _kernelsState.set;
    if (!set || !set.kernels) return;
    const n = set.kernels.length;
    const cur = _kernelsState.selectedIdx;
    let next = cur;
    switch (e.key) {
      case 'ArrowDown': next = Math.min(n - 1, cur + 1); break;
      case 'ArrowUp':   next = Math.max(0, cur - 1);     break;
      case 'Home':      next = 0;                        break;
      case 'End':       next = n - 1;                    break;
      case 'Enter':
      case ' ':
        if (cur >= 0) {
          e.preventDefault();
          announce('kernel ' + set.kernels[cur].id);
        }
        return;
      case 'Escape': {
        e.preventDefault();
        const navItem = document.querySelector('.nav-item[data-view="kernels"]');
        if (navItem && typeof navItem.focus === 'function') navItem.focus();
        return;
      }
      default: return;
    }
    if (next !== cur) {
      e.preventDefault();
      selectKernel(next, { fromKeyboard: true });
    }
  });
}

// initKernelsDiffToggle wires the stub "tuned vs default" button. The
// backend lands in W6+ (POST /api/compile/tuned per spec §7); today the
// button announces a pending message.
function initKernelsDiffToggle() {
  const btn = document.getElementById('diff-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const pressed = btn.getAttribute('aria-pressed') === 'true';
    const next = !pressed;
    btn.setAttribute('aria-pressed', next ? 'true' : 'false');
    announce(next ? 'tuned compile pending — native backend not yet wired'
                  : 'showing default WGSL');
  });
}

// ── boot ──────────────────────────────────────────────────────────────────

function boot() {
  initTheme();
  initSkipLink();
  initRouting();
  initKeyboardHelp();
  initKeyboard();
  initIgnite();
  initKernelsKeyboard();
  initKernelsDiffToggle();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

// Exported for tests / consoles. Not a public API; W1+ may rename.
export const __studio = {
  navigate, applyTheme, cycleTheme, viewIdForPath, announce,
  helpOpen, helpClose,
  // W2 hooks (kernels view) — exported so manual console tests can drive them.
  tokenizeWGSL, renderKernelsView, selectKernel, modelFromPath,
};
