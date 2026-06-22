// anneal studio - main thread ES module.
//
// Responsibilities (DESIGN.md §3, §5, §7, §11; spec §10; web/A11Y.md):
//   1. History API routing for every studio view (no hash routing).
//   2. Theme controller: system | dark | light cycle, persisted, with a live
//      matchMedia listener so OS theme changes apply without a page reload.
//   3. View renderer dispatch - stubs in W0, real renderers land in W1+.
//   4. Keyboard handlers: `/` focuses search; `g <dest>` chord jumps view;
//      `?` opens the keyboard-help modal; `Esc` closes it.
//   5. Worker RPC client - gated behind a <meta name="anneal-worker"> tag so
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
    // aria-label update on theme cycle - describes current state AND the
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
  explain:   '/x/Add',
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
  studio:    function renderStudio()    { renderStudioView(); },
  visualize: function renderVisualize() { renderVisualizeView(); },
  kernels:   function renderKernels()   { renderKernelsView(); },
  explain:   function renderExplain()   { renderExplainView(); },
  train:     function renderTrain()     { renderTrainView(); },
  generate:  function renderGenerate()  { renderGenerateView(); },
  history:   function renderHistory()   { /* W8 */ },
  doctor:    function renderDoctor()    { renderDoctorView(); },
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
//   keyword   - control flow + storage classes (fn, var, let, for, if, …)
//   type      - primitive scalar + vector + matrix types
//   builtin   - main, gid, lid, wid, the @builtin identifiers
//   attribute - @compute, @workgroup_size, @binding, @group, @builtin, …
//   number    - int, float, hex literals (123, 12.5, 0xCAFE, 64u, 1.0f, ...)
//   string    - "..." literals (rare in WGSL but tokenized for safety)
//   comment   - // line comments AND /* block comments */
//   ident     - everything else that starts with a letter/underscore
//   punct     - every other single-char token (operators, braces, semi, …)
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

    // Whitespace - skipped (not emitted as a token).
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
    li.textContent = 'wasm not loaded - build anneal.wasm to populate this view';
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
    announce(next ? 'tuned compile pending - native backend not yet wired'
                  : 'showing default WGSL');
  });
}

// ── train view (W6) ───────────────────────────────────────────────────────
// SSE-driven live dashboard. The renderer is invoked on every navigation to
// /t/<model>; it (re)wires the controls and primes the UI. Starting a run
// opens an EventSource against /sse/train?model=...&steps=...&bundle=...
// and pushes each Snapshot into the DOM + a ring buffer for the sparkline.
//
// a11y (web/A11Y.md §3d):
//   - Every control is labelled (model select, steps input, buttons,
//     checkbox).
//   - The progressbar updates aria-valuenow as it animates.
//   - The stat region is aria-live="polite" - every value change is
//     announced without stealing focus.
//   - The loss SVG's <desc> is rewritten on each step ("loss decreased
//     from X to Y over N steps") as a textual fallback.
//   - The sparkline updates batched into rAF so a fast trainer can't
//     starve the main thread.
//   - Reduced-motion: the kernel-dot gold pulse is replaced by an instant
//     fill flash via the CSS rule already in studio.css.

const _trainState = {
  model: 'mlp',
  es: null,                // active EventSource, null when idle
  buf: new Array(200),     // ring buffer of last 200 loss values
  bufLen: 0,
  bufIdx: 0,
  minLoss: Infinity,
  maxLoss: -Infinity,
  step: 0,
  maxSteps: 100,
  bundle: true,
  pendingFrame: false,     // rAF coalescing flag
};

function modelFromTrainPath(pathname) {
  const m = /^\/t\/([^/?#]+)/.exec(pathname);
  return m ? decodeURIComponent(m[1]) : 'mlp';
}

function renderTrainView() {
  const sel = document.getElementById('train-model');
  const steps = document.getElementById('train-steps');
  const start = document.getElementById('train-start');
  const cancel = document.getElementById('train-cancel');
  const bundle = document.getElementById('train-bundle');
  if (!sel || !steps || !start || !cancel) return;

  // Sync model select from the URL deep link (/t/<model>).
  const model = modelFromTrainPath(window.location.pathname);
  _trainState.model = model;
  if (sel.value !== model) {
    // If the URL specifies an unknown model the <option> won't exist; in
    // that case leave the default selection and let the user pick again.
    const opt = Array.from(sel.options).find((o) => o.value === model);
    if (opt) sel.value = model;
  }

  // Wire controls (idempotent: replace handlers each time).
  start.onclick = trainStart;
  cancel.onclick = trainCancel;
  const openViz = document.getElementById('train-open-viz');
  if (openViz) {
    openViz.onclick = () => {
      navigate('visualize');
      history.pushState({ viewId: 'visualize' }, '', '/v/' + encodeURIComponent(_trainState.model));
    };
  }
  const saveRun = document.getElementById('train-save-run');
  if (saveRun) {
    // Bundle is already written server-side when ?bundle=1 (default).
    // The button is a no-op affordance for screen-reader users; it
    // announces the bundle's status when activated.
    saveRun.onclick = () => {
      announce(bundle && bundle.checked
        ? 'bundle saved to ~/.cache/anneal/runs/'
        : 'bundle disabled; re-run with save bundle checked');
    };
  }
}

function trainStart() {
  // If a stream is already open, cancel it first so a double-press
  // doesn't fork two parallel runs.
  trainCancel();

  const sel = document.getElementById('train-model');
  const steps = document.getElementById('train-steps');
  const bundle = document.getElementById('train-bundle');
  const start = document.getElementById('train-start');
  const cancel = document.getElementById('train-cancel');

  const model = (sel && sel.value) || 'mlp';
  const nSteps = Math.max(1, Math.min(10000, parseInt(steps && steps.value, 10) || 100));
  const wantBundle = !!(bundle && bundle.checked);

  _trainState.model = model;
  _trainState.maxSteps = nSteps;
  _trainState.bundle = wantBundle;
  _trainState.buf = new Array(200);
  _trainState.bufLen = 0;
  _trainState.bufIdx = 0;
  _trainState.minLoss = Infinity;
  _trainState.maxLoss = -Infinity;
  _trainState.step = 0;

  // Update URL deep link so a reload/share resumes at the right model.
  try {
    const url = '/t/' + encodeURIComponent(model);
    if (window.location.pathname !== url) {
      history.replaceState({ viewId: 'train' }, '', url);
    }
  } catch (_) {}

  // Reset display.
  setTrainText('t-step', '0 / ' + nSteps);
  setTrainText('t-loss', '-');
  setTrainText('t-uops', '-');
  setTrainText('t-kernels', '-');
  setTrainText('t-fused', '-');
  setProgress(0, nSteps);
  drawSparkline();

  // Disable start, enable cancel.
  if (start)  start.disabled = true;
  if (cancel) cancel.disabled = false;
  const openViz = document.getElementById('train-open-viz');
  const saveRun = document.getElementById('train-save-run');
  if (openViz) openViz.disabled = true;
  if (saveRun) saveRun.disabled = true;

  const url = '/sse/train?model=' + encodeURIComponent(model)
            + '&steps=' + nSteps
            + '&bundle=' + (wantBundle ? '1' : '0');
  let es;
  try {
    es = new EventSource(url);
  } catch (e) {
    announce('train: cannot open SSE: ' + (e && e.message));
    trainCancel();
    return;
  }
  _trainState.es = es;
  announce('training ' + model + ' started');

  es.addEventListener('message', (ev) => {
    let snap;
    try { snap = JSON.parse(ev.data); }
    catch (e) { return; }
    onSnapshot(snap);
  });
  es.addEventListener('done', () => {
    announce('training complete');
    finishTrain();
  });
  es.addEventListener('error', (_ev) => {
    // EventSource fires `error` on close as well as on network failure;
    // distinguish via readyState.
    if (es.readyState === EventSource.CLOSED) {
      finishTrain();
      return;
    }
    announce('train: stream error');
    finishTrain();
  });
}

function trainCancel() {
  if (_trainState.es) {
    try { _trainState.es.close(); } catch (_) {}
    _trainState.es = null;
    announce('training cancelled');
  }
  const start = document.getElementById('train-start');
  const cancel = document.getElementById('train-cancel');
  if (start)  start.disabled = false;
  if (cancel) cancel.disabled = true;
}

function finishTrain() {
  if (_trainState.es) {
    try { _trainState.es.close(); } catch (_) {}
    _trainState.es = null;
  }
  const start = document.getElementById('train-start');
  const cancel = document.getElementById('train-cancel');
  const openViz = document.getElementById('train-open-viz');
  const saveRun = document.getElementById('train-save-run');
  if (start)  start.disabled = false;
  if (cancel) cancel.disabled = true;
  if (openViz) openViz.disabled = false;
  if (saveRun) saveRun.disabled = !_trainState.bundle;
}

function onSnapshot(snap) {
  if (typeof snap.step === 'number') _trainState.step = snap.step;
  if (typeof snap.max_steps === 'number' && snap.max_steps > 0) {
    _trainState.maxSteps = snap.max_steps;
  }

  setTrainText('t-step', _trainState.step + ' / ' + _trainState.maxSteps);
  if (snap.has_loss) {
    setTrainText('t-loss', formatLoss(snap.loss));
    pushLoss(snap.loss);
  }
  if (typeof snap.uops_count === 'number')    setTrainText('t-uops',    String(snap.uops_count));
  if (typeof snap.kernels_count === 'number') setTrainText('t-kernels', String(snap.kernels_count));
  if (typeof snap.fused_count === 'number')   setTrainText('t-fused',   String(snap.fused_count));

  setProgress(_trainState.step, _trainState.maxSteps);

  if (snap.last_kernel_id) {
    pulseKernelDot(snap.last_kernel_id);
  }

  // Coalesce sparkline redraws into the next animation frame so a fast
  // trainer doesn't repaint per snapshot. Reduced-motion users still get
  // updates; rAF is not motion, it is throttling.
  if (!_trainState.pendingFrame) {
    _trainState.pendingFrame = true;
    requestAnimationFrame(() => {
      _trainState.pendingFrame = false;
      drawSparkline();
    });
  }

  if (snap.phase === 'done') {
    finishTrain();
  } else if (snap.phase === 'error') {
    announce('train error: ' + (snap.error || 'unknown'));
    finishTrain();
  }
}

function setTrainText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setProgress(step, max) {
  const bar = document.getElementById('train-progress-bar');
  const fill = document.getElementById('train-progress-fill');
  if (!bar || !fill) return;
  const pct = max > 0 ? Math.max(0, Math.min(100, (step / max) * 100)) : 0;
  fill.style.width = pct.toFixed(1) + '%';
  bar.setAttribute('aria-valuenow', String(Math.round(pct)));
}

function pushLoss(loss) {
  if (typeof loss !== 'number' || !isFinite(loss)) return;
  const i = _trainState.bufIdx % _trainState.buf.length;
  _trainState.buf[i] = loss;
  _trainState.bufIdx++;
  _trainState.bufLen = Math.min(_trainState.bufLen + 1, _trainState.buf.length);
  if (loss < _trainState.minLoss) _trainState.minLoss = loss;
  if (loss > _trainState.maxLoss) _trainState.maxLoss = loss;
}

function getLossWindow() {
  const N = _trainState.bufLen;
  const cap = _trainState.buf.length;
  const startIdx = _trainState.bufIdx - N;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = _trainState.buf[((startIdx + i) % cap + cap) % cap];
  }
  return out;
}

function formatLoss(loss) {
  if (typeof loss !== 'number' || !isFinite(loss)) return '-';
  const a = Math.abs(loss);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return loss.toExponential(3);
  return loss.toFixed(6);
}

function drawSparkline() {
  const svg = document.getElementById('loss-svg');
  const path = document.getElementById('loss-path');
  const desc = document.getElementById('loss-svg-desc');
  if (!svg || !path) return;
  const vals = getLossWindow();
  if (vals.length < 2) {
    path.setAttribute('d', '');
    if (desc) desc.textContent = vals.length === 1
      ? 'loss sparkline: one sample at ' + formatLoss(vals[0])
      : 'loss sparkline: no data yet';
    return;
  }
  const w = 400, h = 80, pad = 2;
  const lo = _trainState.minLoss, hi = _trainState.maxLoss;
  const range = hi - lo || 1;
  let d = '';
  for (let i = 0; i < vals.length; i++) {
    const x = pad + ((w - 2 * pad) * i) / Math.max(1, vals.length - 1);
    const y = pad + ((h - 2 * pad) * (1 - (vals[i] - lo) / range));
    d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
  }
  path.setAttribute('d', d);
  if (desc) {
    // Textual fallback for screen-reader users; updated every snapshot
    // so the latest value is reported in the SVG's <desc>.
    const first = formatLoss(vals[0]);
    const last  = formatLoss(vals[vals.length - 1]);
    const dir = vals[0] > vals[vals.length - 1] ? 'decreased' : 'changed';
    desc.textContent = 'loss sparkline: ' + dir + ' from '
                     + first + ' to ' + last
                     + ' over ' + vals.length + ' samples';
  }
}

// pulseKernelDot ensures there is a gold dot for kernelID in the thumb SVG
// and adds the .dispatched class to drive the gold pulse animation (or the
// reduced-motion instant flash).
function pulseKernelDot(kernelID) {
  const svg = document.getElementById('kernel-svg');
  if (!svg) return;
  const dotId = 'kd-' + cssEscapeID(kernelID);
  let dot = svg.querySelector('#' + dotId);
  if (!dot) {
    const dots = svg.querySelectorAll('.kernel-dot');
    const n = dots.length;
    // Lay out dots in a horizontal row across the viewBox (200x80).
    const cx = 12 + n * 16;
    const cy = 40;
    dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('id', dotId);
    dot.setAttribute('class', 'kernel-dot');
    dot.setAttribute('cx', String(cx));
    dot.setAttribute('cy', String(cy));
    dot.setAttribute('r', '5');
    svg.appendChild(dot);
  }
  // Drop the dispatched class from siblings so only the most recent
  // dispatch pulses (the CSS animation is infinite; we want one at a
  // time visually).
  svg.querySelectorAll('.kernel-dot.dispatched').forEach((el) => {
    if (el !== dot) el.classList.remove('dispatched');
  });
  dot.classList.add('dispatched');
}

function cssEscapeID(s) {
  // Conservative: keep [a-zA-Z0-9-_], replace everything else with '-'.
  return String(s).replace(/[^a-zA-Z0-9_-]/g, '-');
}

// ── generate view (W7) ───────────────────────────────────────────────────
// SSE-driven token stream. The renderer is invoked on every navigation to
// /g/<model>; it (re)wires the controls and primes the UI. Starting a run
// opens an EventSource against /sse/generate?model=...&prompt=...&tokens=...
// and pushes each TokenSnapshot into the DOM. Every emitted token is a
// focusable <span class="tok"> the user can click (or Enter on while
// focused) to navigate to /k/<model>?kernel=<lastKernelID> - the
// click-through to the producing fused kernel per spec §5.6.
//
// a11y (web/A11Y.md §3e):
//   - Every control is labelled (model select, prompt input, tokens input,
//     buttons, both checkboxes).
//   - The prompt input has maxlength + aria-describedby for the cap rule.
//   - The token stream is aria-live="polite" + aria-atomic="false";
//     announcements are batched every 5 tokens OR 500ms (whichever first)
//     so a fast generation does not flood the SR.
//   - The last-token panel is its own role="region" so SR users can
//     navigate to it via region jump and read the structured detail.
//   - Reduced-motion: the gold pulse on the fresh token is replaced by
//     an instant fill via the CSS rule already in studio.css.
//   - Each .tok span is tabindex="0" and Enter activates click-through.
//   - The warming hint is announced once on first activation.

const _genState = {
  model: 'gpt2',
  es: null,                // active EventSource, null when idle
  promptText: '',
  maxTokens: 32,
  compare: false,
  bundle: true,
  warmed: false,           // first SSE frame has arrived?
  tokenCount: 0,
  lastKernelID: '',
  // Batched announcement: collect token texts, flush every 5 OR after
  // 500ms - whichever comes first - to avoid screen-reader spam.
  pendingAnnounce: [],
  announceTimer: null,
};

function modelFromGenPath(pathname) {
  const m = /^\/g\/([^/?#]+)/.exec(pathname);
  return m ? decodeURIComponent(m[1]) : 'gpt2';
}

function renderGenerateView() {
  const sel = document.getElementById('gen-model');
  const prompt = document.getElementById('gen-prompt');
  const tokens = document.getElementById('gen-tokens');
  const start = document.getElementById('gen-start');
  const cancel = document.getElementById('gen-cancel');
  const compare = document.getElementById('gen-compare');
  const bundle = document.getElementById('gen-bundle');
  if (!sel || !prompt || !tokens || !start || !cancel) return;

  // Sync model select from the URL deep link (/g/<model>).
  const model = modelFromGenPath(window.location.pathname);
  _genState.model = model;
  const opt = Array.from(sel.options).find((o) => o.value === model);
  if (opt && sel.value !== model) sel.value = model;

  // URL params override defaults for prompt / tokens / compare.
  try {
    const u = new URL(window.location.href);
    const pp = u.searchParams.get('prompt');
    if (pp != null && pp.length > 0) prompt.value = pp;
    const tt = u.searchParams.get('tokens');
    if (tt != null && Number(tt) > 0) tokens.value = String(Math.min(256, Number(tt)));
    if (u.searchParams.get('compare') === '1' && compare) compare.checked = true;
    if (u.searchParams.get('bundle') === '0' && bundle) bundle.checked = false;
  } catch (_) {}

  start.onclick = generateStart;
  cancel.onclick = generateCancel;
}

function generateStart() {
  // If a stream is already open, cancel it first.
  generateCancel();

  const sel = document.getElementById('gen-model');
  const promptEl = document.getElementById('gen-prompt');
  const tokensEl = document.getElementById('gen-tokens');
  const compareEl = document.getElementById('gen-compare');
  const bundleEl = document.getElementById('gen-bundle');
  const start = document.getElementById('gen-start');
  const cancel = document.getElementById('gen-cancel');
  const status = document.getElementById('gen-status');
  const warming = document.getElementById('gen-warming');
  const out = document.getElementById('gen-tokens-out');
  const echo = document.getElementById('gen-prompt-echo');
  const lastText = document.getElementById('gen-last-text');
  const lastId = document.getElementById('gen-last-id');
  const lastLogit = document.getElementById('gen-last-logit');
  const lastRef = document.getElementById('gen-last-ref');

  const model = (sel && sel.value) || 'gpt2';
  const promptText = (promptEl && promptEl.value) || '';
  if (!promptText.trim()) {
    announce('generate: prompt is required');
    return;
  }
  const nTokens = Math.max(1, Math.min(256, parseInt(tokensEl && tokensEl.value, 10) || 32));
  const wantCompare = !!(compareEl && compareEl.checked);
  const wantBundle = !!(bundleEl && bundleEl.checked);

  _genState.model = model;
  _genState.promptText = promptText;
  _genState.maxTokens = nTokens;
  _genState.compare = wantCompare;
  _genState.bundle = wantBundle;
  _genState.warmed = false;
  _genState.tokenCount = 0;
  _genState.lastKernelID = '';
  _genState.pendingAnnounce = [];
  if (_genState.announceTimer) {
    clearTimeout(_genState.announceTimer);
    _genState.announceTimer = null;
  }

  // Reset UI.
  if (out) out.textContent = '';
  if (echo) echo.textContent = promptText;
  if (lastText) lastText.textContent = '-';
  if (lastId) lastId.textContent = '-';
  if (lastLogit) lastLogit.textContent = '-';
  if (lastRef) lastRef.textContent = wantCompare ? 'pending' : '-';
  if (status) status.textContent = 'starting…';
  if (warming) warming.hidden = false;

  // Update URL deep link.
  try {
    const params = new URLSearchParams();
    params.set('prompt', promptText);
    if (wantCompare) params.set('compare', '1');
    if (!wantBundle) params.set('bundle', '0');
    const url = '/g/' + encodeURIComponent(model) + '?' + params.toString();
    if (window.location.pathname + window.location.search !== url) {
      history.replaceState({ viewId: 'generate' }, '', url);
    }
  } catch (_) {}

  if (start)  start.disabled = true;
  if (cancel) cancel.disabled = false;

  // Build SSE URL.
  const u = new URL('/sse/generate', window.location.origin);
  u.searchParams.set('model', model);
  u.searchParams.set('prompt', promptText);
  u.searchParams.set('tokens', String(nTokens));
  u.searchParams.set('compare', wantCompare ? '1' : '0');
  u.searchParams.set('bundle', wantBundle ? '1' : '0');

  let es;
  try {
    es = new EventSource(u.toString());
  } catch (e) {
    announce('generate: cannot open SSE: ' + (e && e.message));
    generateCancel();
    return;
  }
  _genState.es = es;
  announce('generating with ' + model + ' started');

  es.addEventListener('message', (ev) => {
    let tok;
    try { tok = JSON.parse(ev.data); }
    catch (e) { return; }
    onTokenSnapshot(tok);
  });
  es.addEventListener('done', (ev) => {
    let payload = {};
    try { payload = JSON.parse(ev.data || '{}'); } catch (_) {}
    const total = payload.total_tokens || _genState.tokenCount;
    const wall = payload.wall_ms || 0;
    if (status) status.textContent = 'done · ' + total + ' tokens · ' + wall + ' ms';
    announce('generation complete: ' + total + ' tokens in ' + wall + ' ms');
    finishGenerate();
  });
  es.addEventListener('error', (_ev) => {
    if (es.readyState === EventSource.CLOSED) {
      finishGenerate();
      return;
    }
    announce('generate: stream error');
    finishGenerate();
  });
}

function generateCancel() {
  if (_genState.es) {
    try { _genState.es.close(); } catch (_) {}
    _genState.es = null;
    announce('generation cancelled');
  }
  const start = document.getElementById('gen-start');
  const cancel = document.getElementById('gen-cancel');
  if (start)  start.disabled = false;
  if (cancel) cancel.disabled = true;
}

function finishGenerate() {
  if (_genState.es) {
    try { _genState.es.close(); } catch (_) {}
    _genState.es = null;
  }
  const start = document.getElementById('gen-start');
  const cancel = document.getElementById('gen-cancel');
  if (start)  start.disabled = false;
  if (cancel) cancel.disabled = true;
  // Flush any pending announcement.
  flushGenAnnounce();
}

function onTokenSnapshot(tok) {
  if (!tok || typeof tok !== 'object') return;

  // The first frame (any phase) hides the warming hint.
  if (!_genState.warmed) {
    _genState.warmed = true;
    const warming = document.getElementById('gen-warming');
    if (warming) warming.hidden = true;
    const status = document.getElementById('gen-status');
    if (status && tok.phase === 'init') status.textContent = 'warming up done · streaming';
    else if (status) status.textContent = 'streaming';
  }

  // Phase routing.
  if (tok.phase === 'done') {
    finishGenerate();
    return;
  }
  if (tok.phase === 'error') {
    announce('generate error: ' + (tok.error || 'unknown'));
    const status = document.getElementById('gen-status');
    if (status) status.textContent = 'error: ' + (tok.error || 'unknown');
    finishGenerate();
    return;
  }
  // Lifecycle PhaseInit carries no token; the warming-hint hide above
  // already used it.
  if (tok.phase === 'init') {
    return;
  }

  // PhaseTraining (used as "generating") - append a token span.
  const text = typeof tok.token_text === 'string' ? tok.token_text : '';
  appendTokenSpan(tok, text);
  _genState.tokenCount++;
  _genState.lastKernelID = tok.last_kernel_id || _genState.lastKernelID;

  // Update last-token panel.
  setGenText('gen-last-text', text === '' ? '(empty)' : JSON.stringify(text));
  setGenText('gen-last-id', String(tok.token_id != null ? tok.token_id : '-'));
  setGenText('gen-last-logit', tok.logit_summary || '-');
  if (typeof tok.ref_match === 'boolean') {
    setGenText('gen-last-ref', tok.ref_match ? '✓ match' : '✗ no match');
  } else if (_genState.compare) {
    setGenText('gen-last-ref', 'pending');
  } else {
    setGenText('gen-last-ref', '-');
  }

  // Update click-through href to the producing kernel.
  const link = document.getElementById('gen-click-through');
  if (link) {
    const kid = _genState.lastKernelID;
    if (kid) {
      link.href = '/k/' + encodeURIComponent(_genState.model)
                + '?kernel=' + encodeURIComponent(kid);
    } else {
      link.href = '/k/' + encodeURIComponent(_genState.model);
    }
  }

  // Compiler pulse: the train view's kernel-thumb is reused if visible
  // (best-effort - typically not present in the generate DOM, so guard).
  if (tok.last_kernel_id) {
    try { pulseKernelDot(tok.last_kernel_id); } catch (_) {}
  }

  // Batched announce: every 5 tokens OR 500ms.
  _genState.pendingAnnounce.push(text || '');
  if (_genState.pendingAnnounce.length >= 5) {
    flushGenAnnounce();
  } else if (!_genState.announceTimer) {
    _genState.announceTimer = setTimeout(flushGenAnnounce, 500);
  }
}

function appendTokenSpan(tok, text) {
  const out = document.getElementById('gen-tokens-out');
  if (!out) return;

  // Drop the .fresh class from any previously freshly-emitted token so
  // only the newest one glows.
  const prev = out.querySelector('.tok.fresh');
  if (prev) prev.classList.remove('fresh');

  const span = document.createElement('span');
  span.className = 'tok fresh';
  span.setAttribute('tabindex', '0');
  span.textContent = text;
  span.dataset.tokenId = String(tok.token_id != null ? tok.token_id : '');
  span.dataset.kernelId = String(tok.last_kernel_id || '');
  // Ref-match marker: when ?compare=1 each token carries a yes/no glyph.
  if (typeof tok.ref_match === 'boolean') {
    span.classList.add(tok.ref_match ? 'refmatch-yes' : 'refmatch-no');
  }
  // Click + Enter both activate the click-through.
  span.addEventListener('click', (e) => {
    e.preventDefault();
    openKernelForToken(span);
  });
  span.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openKernelForToken(span);
    }
  });
  out.appendChild(span);

  // Auto-scroll the stream so the latest token stays in view.
  const stream = out.parentElement;
  if (stream && stream.scrollTop != null) {
    stream.scrollTop = stream.scrollHeight;
  }
}

// openKernelForToken navigates to /k/<model>?kernel=<id> so the user
// lands in the producing fused kernel for the clicked token. This is
// the click-through promised by spec §5.6.
function openKernelForToken(span) {
  const kid = span.dataset.kernelId || _genState.lastKernelID;
  const model = _genState.model || 'gpt2';
  const url = kid
    ? '/k/' + encodeURIComponent(model) + '?kernel=' + encodeURIComponent(kid)
    : '/k/' + encodeURIComponent(model);
  history.pushState({ viewId: 'kernels' }, '', url);
  setActiveView('kernels');
}

function setGenText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function flushGenAnnounce() {
  if (_genState.announceTimer) {
    clearTimeout(_genState.announceTimer);
    _genState.announceTimer = null;
  }
  if (_genState.pendingAnnounce.length === 0) return;
  const last = _genState.pendingAnnounce[_genState.pendingAnnounce.length - 1];
  const msg = 'generated ' + _genState.tokenCount + ' tokens, last token: '
            + JSON.stringify(last);
  announce(msg);
  _genState.pendingAnnounce = [];
}

// ── visualize view (W4) ───────────────────────────────────────────────────
// Spec: notes/anneal_web_spec.md §5.2.
// Two pieces share this section:
//   1. renderVisualizeView()  - runs every time we navigate to /v/<model>;
//      reads the URL, sets the iframe src to /visualize/embed (or sends an
//      inbound nodeSelected for a pre-opened deep link), and ensures the
//      message + Escape handlers are armed exactly once.
//   2. The drawer state machine - open / close / focus management. The
//      drawer is `role="region"` + `aria-label`; opening it captures focus,
//      Escape closes it and returns focus to the iframe (per A11Y.md §3c).

const _vizState = {
  // The element that had focus before the drawer opened; we restore here.
  previousActiveElement: null,
  // Last node we showed; suppresses redundant work when the iframe sends
  // the same nodeClick (e.g. user double-clicks).
  currentNodeId: null,
  // Has the iframe sent embedReady? Until then, parent → iframe
  // nodeSelected messages have nothing to highlight; we queue the most
  // recent intent and resend on ready.
  embedReady: false,
  pendingNodeId: null,
  // armed === true once the wireVisualizeListeners() one-shot has run.
  armed: false,
};

function modelFromVizPath(pathname) {
  const m = /^\/v\/([^/?#]+)/.exec(pathname);
  return m ? decodeURIComponent(m[1]) : 'mlp';
}

function nodeIdFromQuery() {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get('node');
  } catch (_) { return null; }
}

function renderVisualizeView() {
  wireVisualizeListeners();

  // If the URL carries ?node=<id>, pre-open the drawer with that node
  // selected once the iframe says it is ready.
  const wantNode = nodeIdFromQuery();
  if (wantNode) {
    _vizState.pendingNodeId = wantNode;
    if (_vizState.embedReady) {
      sendNodeSelectedToIframe(wantNode);
      fetchAndShowNodeDetail(modelFromVizPath(window.location.pathname), wantNode);
    }
  }
}

// wireVisualizeListeners arms the postMessage and Escape handlers exactly
// once per page load. We listen on window for the iframe messages and on
// document for Escape; the close button has its own click handler.
function wireVisualizeListeners() {
  if (_vizState.armed) return;
  _vizState.armed = true;

  window.addEventListener('message', onVizMessage);

  const closeBtn = document.getElementById('node-inspector-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => closeNodeInspector());
  }

  // Escape on the visualize view closes the drawer if open. We attach to
  // document so the iframe focus does not block the key path - the iframe
  // itself runs in a sandbox and will not capture Escape from the parent.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const drawer = document.getElementById('node-inspector');
    if (!drawer || drawer.hidden) return;
    // Only on the visualize view.
    const view = document.getElementById('view-visualize');
    if (!view || !view.classList.contains('active')) return;
    e.preventDefault();
    closeNodeInspector();
  });
}

function onVizMessage(ev) {
  const m = ev && ev.data;
  if (!m || typeof m !== 'object') return;
  switch (m.type) {
    case 'embedReady':
      _vizState.embedReady = true;
      if (_vizState.pendingNodeId) {
        sendNodeSelectedToIframe(_vizState.pendingNodeId);
        fetchAndShowNodeDetail(
          modelFromVizPath(window.location.pathname),
          _vizState.pendingNodeId
        );
      }
      return;
    case 'nodeClick': {
      const graphId = String(m.graphId || modelFromVizPath(window.location.pathname));
      const nodeId  = String(m.nodeId || '');
      if (!nodeId) return;
      fetchAndShowNodeDetail(graphId, nodeId);
      // Reflect the selection in the URL so deep-link copy/paste preserves it.
      try {
        const url = '/v/' + encodeURIComponent(graphId) + '?node=' + encodeURIComponent(nodeId);
        if (window.location.pathname + window.location.search !== url) {
          history.pushState({ viewId: 'visualize', node: nodeId }, '', url);
        }
      } catch (_) {}
      return;
    }
    default:
      return;
  }
}

function sendNodeSelectedToIframe(nodeId) {
  const iframe = document.getElementById('viz-iframe');
  if (!iframe || !iframe.contentWindow) return;
  iframe.contentWindow.postMessage({ type: 'nodeSelected', nodeId }, '*');
}

async function fetchAndShowNodeDetail(graphId, nodeId) {
  const drawer = document.getElementById('node-inspector');
  if (!drawer) return;

  // The drawer's first interactive element is the close button; capture
  // the previously-focused element BEFORE we open so Escape restores
  // it correctly.
  if (drawer.hidden) {
    _vizState.previousActiveElement = document.activeElement;
  }

  // Open + show a loading placeholder so the screen-reader user hears
  // "inspecting node X" promptly even if the WASM call takes a beat.
  drawer.hidden = false;
  setNodeInspectorContent({
    op: nodeId,
    dtype: 'loading…',
    shape: [],
    phase: '',
    arg: '',
    source_file: '',
    parents: [],
    children: [],
  });
  announce('inspecting node ' + nodeId);

  let raw;
  try {
    raw = await wasm.call('annealNodeDetail', graphId, nodeId);
  } catch (e) {
    setNodeInspectorError((e && e.message) || String(e));
    focusDrawerClose();
    return;
  }
  let nd;
  try { nd = JSON.parse(raw); }
  catch (e) {
    setNodeInspectorError('invalid JSON from annealNodeDetail');
    focusDrawerClose();
    return;
  }
  if (nd && nd.error) {
    setNodeInspectorError(nd.error);
    focusDrawerClose();
    return;
  }
  _vizState.currentNodeId = nodeId;
  setNodeInspectorContent(nd);
  focusDrawerClose();
}

function focusDrawerClose() {
  const btn = document.getElementById('node-inspector-close');
  if (btn && typeof btn.focus === 'function') btn.focus();
}

function setNodeInspectorContent(nd) {
  setText('node-inspector-op', nd.op || '');
  setText('ni-dtype', nd.dtype || '');
  setText('ni-shape', Array.isArray(nd.shape) && nd.shape.length
    ? '[' + nd.shape.join(',') + ']'
    : '');
  setText('ni-phase', nd.phase || '');
  setText('ni-arg', nd.arg || '');
  setText('ni-source', (nd.source_file && nd.source_line)
    ? (nd.source_file + ':' + nd.source_line)
    : '');
  renderRelationList('ni-parents', nd.parents || []);
  renderRelationList('ni-children', nd.children || []);
}

function setNodeInspectorError(msg) {
  setText('node-inspector-op', 'error');
  setText('ni-dtype', msg);
  setText('ni-shape', '');
  setText('ni-phase', '');
  setText('ni-arg', '');
  setText('ni-source', '');
  renderRelationList('ni-parents', []);
  renderRelationList('ni-children', []);
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(txt);
}

function renderRelationList(id, items) {
  const ul = document.getElementById(id);
  if (!ul) return;
  ul.innerHTML = '';
  for (const it of items) {
    const li = document.createElement('li');
    li.textContent = (it.op || '?') + ' · ' + (it.id || '');
    if (it.label && it.label !== it.op) {
      const small = document.createElement('span');
      small.className = 'ni-label';
      small.textContent = ' (' + it.label + ')';
      li.appendChild(small);
    }
    ul.appendChild(li);
  }
}

function closeNodeInspector() {
  const drawer = document.getElementById('node-inspector');
  if (!drawer) return;
  drawer.hidden = true;
  _vizState.currentNodeId = null;

  // Restore focus to the previously-active element (WCAG 2.4.3 / 2.4.11).
  // Falls back to the iframe so keyboard users land somewhere usable.
  const prev = _vizState.previousActiveElement;
  if (prev && typeof prev.focus === 'function' &&
      document.body.contains(prev)) {
    prev.focus();
  } else {
    const iframe = document.getElementById('viz-iframe');
    if (iframe && typeof iframe.focus === 'function') iframe.focus();
  }
  _vizState.previousActiveElement = null;
}

// ── explain view (W3) ─────────────────────────────────────────────────────
// Driven by the annealExplainOp WASM bridge. State lives outside the
// renderer so navigating back to /x/<op> preserves filter + selection.
//
// The op list is the master list (parsed once from the WASM bridge's first
// payload, or fallback to a hard-coded set for the wasm-not-loaded path);
// search input filters it client-side with a 100ms debounce so the keyboard
// stays responsive. Detail panel renders rules + gradient + mini-graph.
//
// a11y plan (web/A11Y.md §3b):
//   - Focus order: search input → op list → detail (description → rules →
//     gradient pre → play button).
//   - aria-live="polite" on the detail article announces selection changes.
//   - The play-rewrite button announces "rule X fired" during the animation.
//   - prefers-reduced-motion replaces the animation with an instant swap.

// Fallback op list when WASM has not loaded. Sourced from uop/ops.go opNames;
// the studio still renders a usable picker so the user can browse op names
// without the compiler being available.
const FALLBACK_OPS = [
  'Add', 'And', 'Bind', 'Bitcast', 'Buffer', 'Cast', 'CmpEq', 'CmpLt', 'CmpNe',
  'Const', 'DefineVar', 'Erf', 'Exp2', 'Expand', 'FDiv', 'Flip', 'Gather',
  'IDiv', 'Log2', 'Max', 'Min', 'Mod', 'Mul', 'MulAcc', 'Neg', 'Or', 'Pad',
  'Permute', 'Pow', 'Reciprocal', 'ReduceAxis', 'Reshape', 'Shl', 'Shrink',
  'Shr', 'Sin', 'Sink', 'Sqrt', 'Sub', 'ThreeFry', 'Trunc', 'Where', 'Xor',
];

const _explainState = {
  ops: FALLBACK_OPS.slice(),  // visible op list (full set, pre-filter)
  filter: '',
  selectedOp: null,
  detail: null,           // last successful annealExplainOp payload
  loading: false,
  error: null,
};

let _explainDebounce = null;

function opFromExplainPath(pathname) {
  const m = /^\/x\/([^/?#]+)/.exec(pathname);
  return m ? decodeURIComponent(m[1]) : 'Add';
}

async function renderExplainView() {
  const wantOp = opFromExplainPath(window.location.pathname);
  _explainState.selectedOp = wantOp;

  drawOpList();
  await loadExplain(wantOp);
}

// loadExplain calls annealExplainOp and renders the detail panel. Falls
// through to an empty-state render if the bridge rejects (no WASM loaded).
async function loadExplain(opName) {
  _explainState.loading = true;
  _explainState.error = null;
  drawExplainDetail();

  let raw;
  try {
    raw = await wasm.call('annealExplainOp', opName);
  } catch (e) {
    _explainState.loading = false;
    _explainState.error = (e && e.message) || String(e);
    drawExplainDetail();
    announce('explain view: wasm not loaded');
    return;
  }
  let payload;
  try { payload = JSON.parse(raw); }
  catch (_) {
    _explainState.loading = false;
    _explainState.error = 'invalid JSON from annealExplainOp';
    drawExplainDetail();
    return;
  }
  if (payload && payload.error) {
    _explainState.loading = false;
    _explainState.error = payload.error;
    drawExplainDetail();
    announce('explain: ' + payload.error);
    return;
  }
  _explainState.loading = false;
  _explainState.detail = payload;
  drawExplainDetail();
  announce('explain: ' + payload.op + ' selected');
}

// drawOpList renders the filterable op list. Items are real <li role="option">
// so the listbox role on the parent UL is honoured.
function drawOpList() {
  const ul = document.getElementById('op-list-items');
  if (!ul) return;
  const q = _explainState.filter.toLowerCase();
  const visible = _explainState.ops.filter((op) =>
    !q || op.toLowerCase().includes(q));
  ul.innerHTML = '';
  if (visible.length === 0) {
    const li = document.createElement('li');
    li.className = 'op-list-empty';
    li.textContent = 'no ops match';
    ul.appendChild(li);
    return;
  }
  for (const op of visible) {
    const li = document.createElement('li');
    li.className = 'op-list-item';
    li.setAttribute('role', 'option');
    li.id = 'op-opt-' + op;
    li.textContent = op;
    const sel = (op === _explainState.selectedOp);
    li.setAttribute('aria-selected', sel ? 'true' : 'false');
    li.addEventListener('click', () => selectOp(op));
    ul.appendChild(li);
  }
  if (_explainState.selectedOp) {
    ul.setAttribute('aria-activedescendant', 'op-opt-' + _explainState.selectedOp);
  }
}

// drawExplainDetail renders the right pane: name, description, rules list,
// gradient pre, and the mini-graph SVG. Defensive against null/error states
// so the view always has SOMETHING legible on screen.
function drawExplainDetail() {
  const nameEl  = document.getElementById('exp-op-name');
  const descEl  = document.getElementById('exp-desc');
  const rulesEl = document.getElementById('exp-rules');
  const gradEl  = document.getElementById('exp-grad');
  const miniEl  = document.getElementById('exp-mini');
  if (!nameEl || !descEl || !rulesEl || !gradEl || !miniEl) return;

  if (_explainState.error) {
    nameEl.textContent = _explainState.selectedOp || '';
    descEl.textContent = 'wasm not loaded - build anneal.wasm to populate this view';
    rulesEl.innerHTML = '';
    gradEl.textContent = '';
    const svg = miniEl.querySelector('svg');
    if (svg) svg.innerHTML = '';
    return;
  }
  if (_explainState.loading || !_explainState.detail) {
    nameEl.textContent = _explainState.selectedOp || '';
    descEl.textContent = 'loading…';
    return;
  }
  const d = _explainState.detail;
  nameEl.textContent = d.op;
  descEl.textContent = d.description || '';
  // Rules list. Each row pairs a monospace pattern + arrow + rewrite. The
  // notes line is rendered as a small caption so colour is not the only
  // channel of meaning (DD1).
  rulesEl.innerHTML = '';
  if (!d.symbolic_rules || d.symbolic_rules.length === 0) {
    const li = document.createElement('li');
    li.className = 'rules-list-empty';
    li.textContent = 'no symbolic rewrite rules registered for this op';
    rulesEl.appendChild(li);
  } else {
    for (const r of d.symbolic_rules) {
      const li = document.createElement('li');
      li.className = 'rules-list-item';
      const code = document.createElement('code');
      code.className = 'rule-pattern';
      code.textContent = r.pattern + '  →  ' + (r.rewrite || '');
      li.appendChild(code);
      if (r.notes) {
        const note = document.createElement('span');
        note.className = 'rule-note';
        note.textContent = r.notes;
        li.appendChild(note);
      }
      const src = document.createElement('span');
      src.className = 'rule-source';
      src.textContent = r.source;
      li.appendChild(src);
      rulesEl.appendChild(li);
    }
  }
  // Gradient pre.
  if (d.gradient_rule) {
    gradEl.textContent = d.gradient_rule.pattern + '\n\nsource: ' + d.gradient_rule.source;
  } else {
    gradEl.textContent = 'no gradient registered (this op is non-differentiable or has a derived gradient via primitives).';
  }
  // Mini-graph SVG.
  drawMiniGraph(d.mini_graph, 'before');
}

// drawMiniGraph renders the before/after node trees into the SVG. side is
// 'before' or 'after' (the play-rewrite button animates from before to
// after; the JS just calls drawMiniGraph again with the other side).
function drawMiniGraph(mg, side) {
  const miniEl = document.getElementById('exp-mini');
  if (!miniEl) return;
  const svg = miniEl.querySelector('svg');
  if (!svg) return;
  svg.innerHTML = '';
  if (!mg) return;
  const nodes = (side === 'after' ? mg.after : mg.before) || [];
  const edges = (side === 'after' ? [] : mg.edges) || [];

  // Layout: simple top-down with the operation node at the bottom. We anchor
  // the operation node (the last entry that has incoming edges) at the
  // bottom-centre; leaves spread along the top.
  const W = 360, H = 140;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('width', '100%');
  // A descriptive title for screen readers; the SVG role="img" is set in HTML.
  const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'title');
  titleEl.textContent = 'rewrite ' + side + ' state';
  svg.appendChild(titleEl);

  // Distinguish "operation" node (has incoming edges) from "leaf" (no
  // incoming). Simple in-degree count.
  const inDeg = {};
  for (const n of nodes) inDeg[n.id] = 0;
  for (const e of edges) {
    if (inDeg[e.to] !== undefined) inDeg[e.to] += 1;
  }
  const op = nodes.find((n) => inDeg[n.id] > 0);
  const leaves = nodes.filter((n) => inDeg[n.id] === 0);

  const yLeaf = 28, yOp = H - 28;
  const positions = {};
  if (leaves.length > 0) {
    const step = W / (leaves.length + 1);
    leaves.forEach((n, i) => {
      positions[n.id] = { x: step * (i + 1), y: yLeaf };
    });
  }
  if (op) {
    positions[op.id] = { x: W / 2, y: yOp };
  } else if (nodes.length === 1) {
    // Single-node side (the "After" of a typical identity rewrite).
    positions[nodes[0].id] = { x: W / 2, y: H / 2 };
  }

  // Edges first so they sit beneath the nodes.
  for (const e of edges) {
    const from = positions[e.from];
    const to   = positions[e.to];
    if (!from || !to) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
    line.setAttribute('x2', to.x);   line.setAttribute('y2', to.y);
    line.setAttribute('class', 'mini-edge');
    line.setAttribute('stroke-linecap', 'round');
    svg.appendChild(line);
  }

  // Nodes.
  for (const n of nodes) {
    const p = positions[n.id];
    if (!p) continue;
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'mini-node mini-node-' + nodeShapeClass(n.op));
    g.setAttribute('id', 'mini-' + n.id);
    g.setAttribute('transform', 'translate(' + p.x + ',' + p.y + ')');

    // Shape per node kind (DD1: shape carries meaning, not just colour).
    const shape = nodeShape(n.op);
    svgAppendShape(g, shape);

    // Inner title for assistive tech.
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = n.op + ': ' + n.label;
    g.appendChild(t);

    // Label text.
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'central');
    text.setAttribute('class', 'mini-label');
    text.textContent = n.label;
    g.appendChild(text);
    svg.appendChild(g);
  }
}

// nodeShape returns the SVG primitive description for a given op kind. Each
// shape is distinct so a screen reader without colour can still tell a Const
// (square) from a Var (circle) from an ALU node (hexagon).
function nodeShape(op) {
  if (op === 'Const')     return { type: 'rect', w: 38, h: 22 };
  if (op === 'DefineVar') return { type: 'circle', r: 14 };
  if (op === 'Var')       return { type: 'circle', r: 14 };
  // Default: rounded rect (ALU op).
  return { type: 'rounded', w: 44, h: 22, rx: 6 };
}
function nodeShapeClass(op) {
  if (op === 'Const')     return 'const';
  if (op === 'DefineVar' || op === 'Var') return 'leaf';
  return 'alu';
}
function svgAppendShape(g, s) {
  if (s.type === 'rect') {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', -s.w / 2); r.setAttribute('y', -s.h / 2);
    r.setAttribute('width', s.w); r.setAttribute('height', s.h);
    r.setAttribute('class', 'mini-shape');
    g.appendChild(r);
  } else if (s.type === 'circle') {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('r', s.r);
    c.setAttribute('class', 'mini-shape');
    g.appendChild(c);
  } else {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', -s.w / 2); r.setAttribute('y', -s.h / 2);
    r.setAttribute('width', s.w); r.setAttribute('height', s.h);
    r.setAttribute('rx', s.rx);
    r.setAttribute('class', 'mini-shape');
    g.appendChild(r);
  }
}

function selectOp(op) {
  _explainState.selectedOp = op;
  // Push deep link.
  try {
    const url = '/x/' + encodeURIComponent(op);
    if (window.location.pathname !== url) {
      history.pushState({ viewId: 'explain', op }, '', url);
    }
  } catch (_) {}
  drawOpList();
  loadExplain(op);
}

function initExplainSearch() {
  const input = document.getElementById('op-search');
  if (!input) return;
  input.addEventListener('input', (e) => {
    const v = e.target.value || '';
    if (_explainDebounce) clearTimeout(_explainDebounce);
    _explainDebounce = setTimeout(() => {
      _explainState.filter = v;
      drawOpList();
    }, 100);
  });
}

function initExplainKeyboard() {
  const listEl = document.getElementById('op-list-items');
  if (!listEl) return;
  listEl.addEventListener('keydown', (e) => {
    // Compute the currently-visible filtered ops so ArrowDown/Up navigate the
    // SAME set the user sees.
    const q = _explainState.filter.toLowerCase();
    const visible = _explainState.ops.filter((op) =>
      !q || op.toLowerCase().includes(q));
    if (visible.length === 0) return;
    const cur = visible.indexOf(_explainState.selectedOp);
    let next = cur;
    switch (e.key) {
      case 'ArrowDown': next = Math.min(visible.length - 1, cur + 1); break;
      case 'ArrowUp':   next = Math.max(0, cur - 1);                  break;
      case 'Home':      next = 0;                                     break;
      case 'End':       next = visible.length - 1;                    break;
      case 'Enter':
      case ' ':
        if (_explainState.selectedOp) {
          e.preventDefault();
          announce('op ' + _explainState.selectedOp);
        }
        return;
      case 'Escape': {
        e.preventDefault();
        const navItem = document.querySelector('.nav-item[data-view="explain"]');
        if (navItem && typeof navItem.focus === 'function') navItem.focus();
        return;
      }
      default: return;
    }
    if (next !== cur && next >= 0) {
      e.preventDefault();
      selectOp(visible[next]);
      const el = document.getElementById('op-opt-' + visible[next]);
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'nearest' });
      }
    }
  });
}

function initExplainPlayButton() {
  const btn = document.getElementById('play-rewrite');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const detail = _explainState.detail;
    if (!detail || !detail.mini_graph) return;
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    // Announce the rule that's about to fire (first symbolic rule when
    // present; otherwise the gradient rule). The polite live region picks
    // it up without stealing focus.
    let ruleName = 'identity';
    if (detail.symbolic_rules && detail.symbolic_rules.length > 0) {
      ruleName = detail.symbolic_rules[0].name;
    }
    announce('rule ' + ruleName + ' fired');
    if (reduced) {
      drawMiniGraph(detail.mini_graph, 'after');
      // After a beat, restore the before so the user can replay.
      setTimeout(() => drawMiniGraph(detail.mini_graph, 'before'), 1200);
      return;
    }
    // Animated path: a CSS class on the SVG pulses the op node before swap.
    const miniEl = document.getElementById('exp-mini');
    const svg = miniEl && miniEl.querySelector('svg');
    if (svg) {
      svg.classList.add('rewriting');
      // Find the op (in-degree > 0) node and pulse it.
      const opNode = svg.querySelector('.mini-node-alu');
      if (opNode) {
        const shape = opNode.querySelector('.mini-shape');
        if (shape) {
          shape.classList.add('rule-just-fired');
          setTimeout(() => shape.classList.remove('rule-just-fired'), 700);
        }
      }
    }
    setTimeout(() => {
      drawMiniGraph(detail.mini_graph, 'after');
      if (svg) svg.classList.remove('rewriting');
      // Re-render the before after a short pause so the user can replay.
      setTimeout(() => drawMiniGraph(detail.mini_graph, 'before'), 1400);
    }, 600);
  });
}

// ── studio home view (W9): tensor-inspect dropzone ───────────────────────
// Spec: notes/anneal_web_spec.md §5.1.
//
// The dropzone reads a dropped .npy / .npz / .safetensors file into a
// Uint8Array, dispatches it to the WASM bridge (annealInspectTensor), and
// renders the InspectResult JSON into the table. Bytes never leave the tab;
// there is no server endpoint for tensor inspection (the privacy property
// that falls out of WASM-tier inspection per spec §1).
//
// a11y notes:
//   - dropzone is `role="region"` with an aria-label and aria-describedby
//   - file input sits inside the dropzone for pointer + AT users
//   - result section is aria-live="polite" so a screen reader announces it
//   - result table uses real <th scope="col"> headers
//   - keyboard: focus the dropzone and press Enter or Space to open the
//     file picker

const _inspectState = {
  armed: false,
};

function renderStudioView() {
  initInspectDropzone();
  initStudioDropzone();
}

function initInspectDropzone() {
  if (_inspectState.armed) return;
  _inspectState.armed = true;

  const zone = document.getElementById('tensor-dropzone');
  const picker = document.getElementById('tensor-picker');
  if (!zone || !picker) return;

  // Click anywhere in the zone opens the picker (so pointer users don't
  // have to find the small input). The inner input click is allowed
  // through; stopping the recursion is what the `e.target === picker`
  // guard does.
  zone.addEventListener('click', (e) => {
    if (e.target === picker) return;
    picker.click();
  });
  // Keyboard activation: Enter / Space on the dropzone opens the picker.
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      picker.click();
    }
  });

  // Drag and drop events. dragover with preventDefault is required to make
  // a drop target functional; dragleave clears the visual highlight.
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) inspectFile(file);
  });

  picker.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) inspectFile(file);
  });
}

function detectInspectFormat(name) {
  const n = String(name || '').toLowerCase();
  if (n.endsWith('.npy'))         return 'npy';
  if (n.endsWith('.npz'))         return 'npz';
  if (n.endsWith('.safetensors')) return 'safetensors';
  return '';
}

async function inspectFile(file) {
  const format = detectInspectFormat(file.name);
  const resultEl = document.getElementById('tensor-result');
  const nameEl = document.getElementById('tensor-file-name');
  const metaEl = document.getElementById('tensor-result-meta');
  const tbody = document.getElementById('tensor-rows');
  if (!resultEl || !nameEl || !metaEl || !tbody) return;

  nameEl.textContent = file.name;
  resultEl.hidden = false;
  tbody.innerHTML = '';

  if (!format) {
    metaEl.textContent = 'unknown extension - expected .npy, .npz, or .safetensors';
    announce('tensor inspect: unknown extension');
    return;
  }

  metaEl.textContent = 'reading ' + file.size.toLocaleString() + ' bytes…';
  let bytes;
  try {
    const buf = await file.arrayBuffer();
    bytes = new Uint8Array(buf);
  } catch (e) {
    metaEl.textContent = 'read error: ' + ((e && e.message) || String(e));
    return;
  }

  let raw;
  try {
    raw = await wasm.call('annealInspectTensor', bytes, format);
  } catch (e) {
    metaEl.textContent = 'wasm not loaded - build anneal.wasm to use the inspector';
    announce('tensor inspect: wasm not loaded');
    return;
  }
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (e) {
    metaEl.textContent = 'invalid JSON from annealInspectTensor';
    return;
  }
  if (payload && payload.error) {
    metaEl.textContent = 'parse error: ' + payload.error;
    return;
  }
  renderInspectRows(payload);
  announce('tensor inspect: ' + (payload.tensors ? payload.tensors.length : 0) + ' tensor(s) read');
}

function renderInspectRows(payload) {
  const metaEl = document.getElementById('tensor-result-meta');
  const tbody = document.getElementById('tensor-rows');
  if (!metaEl || !tbody) return;
  const tensors = (payload && payload.tensors) || [];
  metaEl.textContent = 'format: ' + (payload.format || '?') +
    ' · tensors: ' + tensors.length;
  tbody.innerHTML = '';
  for (const t of tensors) {
    const tr = document.createElement('tr');
    appendCell(tr, t.name || '');
    appendCell(tr, t.dtype || '');
    appendCell(tr, '[' + ((t.shape || []).join(',')) + ']');
    appendCell(tr, String(t.numel || 0));
    const previewCell = appendCell(tr, '');
    previewCell.className = 'tensor-preview';
    previewCell.textContent = formatPreview(t.preview || []);
    tbody.appendChild(tr);
  }
}

function appendCell(tr, text) {
  const td = document.createElement('td');
  td.textContent = text;
  tr.appendChild(td);
  return td;
}

function formatPreview(values) {
  if (!Array.isArray(values) || values.length === 0) return '[]';
  const formatted = values.map((v) => {
    if (typeof v !== 'number' || !isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(3);
    return v.toFixed(4);
  });
  return '[' + formatted.join(', ') + ']';
}

// ── studio home view (W8): ONNX dropzone ─────────────────────────────────
// Spec: notes/anneal_web_spec.md §5.1, §8.
//
// The studio reads a dropped .onnx file into a Uint8Array, dispatches it to
// the WASM bridge (annealImportONNX) which runs the importer in
// structure-only mode, and renders the topology immediately.
//
// Privacy contract (spec §1.3): model bytes never leave the tab. There is
// NO server endpoint. The entire import path is WASM-tier.
//
// The imported summary's graph_id is stashed in sessionStorage so the
// "visualize" and "kernels" deep links resolve on a subsequent page load:
//   sessionStorage["anneal-imported-<graphID>"] = JSON.stringify(summary)
// W9+ will lift these into a real /v/imported-<id> route renderer.
//
// a11y notes:
//   - dropzone is `role="region"` with aria-label + aria-describedby
//   - keyboard: focus the dropzone, press Enter or Space, or use the pick
//     button (a real <button>) for AT users
//   - result section is aria-live="polite" so the import announces
//     incrementally without stealing focus
//   - unsupported-op list uses ember accent + ● glyph + dashed underline
//     (DD1: colour never alone)
//   - errors (malformed protobuf, unsupported dtype, etc.) surface in the
//     summary line and announce as "import error: …"

const _onnxState = {
  armed: false,
};

function initStudioDropzone() {
  if (_onnxState.armed) return;
  _onnxState.armed = true;

  const zone = document.getElementById('onnx-dropzone');
  const picker = document.getElementById('onnx-picker');
  const pickerBtn = document.getElementById('onnx-picker-btn');
  if (!zone || !picker) return;

  if (pickerBtn) {
    pickerBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      picker.click();
    });
  }

  // Click the zone (anywhere except the picker / button) opens the picker.
  zone.addEventListener('click', (e) => {
    if (e.target === picker) return;
    if (e.target === pickerBtn) return;
    picker.click();
  });
  // Keyboard activation: Enter / Space on the dropzone opens the picker.
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      // Don't intercept space when the focused element is a button (the
      // pick button consumes it natively).
      if (e.target === pickerBtn || e.target === picker) return;
      e.preventDefault();
      picker.click();
    }
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) importONNXFile(file);
  });

  picker.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) importONNXFile(file);
  });
}

async function importONNXFile(file) {
  const resultEl = document.getElementById('onnx-result');
  const nameEl = document.getElementById('onnx-model-name');
  const sumEl = document.getElementById('onnx-summary');
  const unsupSec = document.getElementById('onnx-unsupported-section');
  const unsupList = document.getElementById('onnx-unsupported-list');
  const visualizeLink = document.getElementById('onnx-visualize-link');
  const kernelsLink = document.getElementById('onnx-kernels-link');
  if (!resultEl || !nameEl || !sumEl) return;

  resultEl.hidden = false;
  // Strip the extension for display.
  const displayName = String(file.name || 'model').replace(/\.onnx$/i, '');
  nameEl.textContent = displayName;
  sumEl.textContent = 'reading ' + file.size.toLocaleString() + ' bytes…';
  if (unsupSec) unsupSec.hidden = true;
  if (unsupList) unsupList.innerHTML = '';

  let bytes;
  try {
    const buf = await file.arrayBuffer();
    bytes = new Uint8Array(buf);
  } catch (e) {
    sumEl.textContent = 'read error: ' + ((e && e.message) || String(e));
    announce('import error: read failed');
    return;
  }

  let raw;
  try {
    raw = await wasm.call('annealImportONNX', bytes);
  } catch (e) {
    sumEl.textContent = 'wasm not loaded - build anneal.wasm to import ONNX models';
    announce('import error: wasm not loaded');
    return;
  }
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (_e) {
    sumEl.textContent = 'invalid JSON from annealImportONNX';
    announce('import error: invalid JSON');
    return;
  }
  if (payload && payload.error) {
    sumEl.textContent = 'parse error: ' + payload.error;
    announce('import error: ' + payload.error);
    return;
  }
  renderONNXSummary(payload, displayName);
}

function renderONNXSummary(payload, displayName) {
  const sumEl = document.getElementById('onnx-summary');
  const unsupSec = document.getElementById('onnx-unsupported-section');
  const unsupList = document.getElementById('onnx-unsupported-list');
  const visualizeLink = document.getElementById('onnx-visualize-link');
  const kernelsLink = document.getElementById('onnx-kernels-link');
  if (!sumEl || !payload) return;

  const nodeCount = payload.node_count || 0;
  const initCount = payload.initializer_count || 0;
  const unsupOps = (payload.unsupported_ops || []);
  const unsupCount = unsupOps.length;

  let summary = nodeCount + ' nodes · ' + initCount + ' initializers';
  if (payload.opset) summary += ' · opset ' + payload.opset;
  if (unsupCount > 0) {
    summary += ' · ' + unsupCount + ' unsupported op';
    if (unsupCount !== 1) summary += 's';
  }
  if (payload.note) summary += ' · ' + payload.note;
  sumEl.textContent = summary;

  if (unsupSec && unsupList) {
    if (unsupCount > 0) {
      unsupSec.hidden = false;
      unsupList.innerHTML = '';
      for (const u of unsupOps) {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.className = 'op-name';
        name.textContent = u.op_type + (u.count > 1 ? ' (' + u.count + ')' : '');
        const reason = document.createElement('span');
        reason.className = 'op-reason';
        reason.textContent = ' - ' + (u.reason || 'no handler registered');
        li.appendChild(name);
        li.appendChild(reason);
        unsupList.appendChild(li);
      }
    } else {
      unsupSec.hidden = true;
    }
  }

  // Stash the imported summary in sessionStorage so the deep links can
  // resolve graph_id on a subsequent page load.
  const gid = payload.graph_id;
  if (gid) {
    try {
      sessionStorage.setItem('anneal-imported-' + gid, JSON.stringify({
        ...payload,
        display_name: displayName,
      }));
    } catch (_e) { /* private mode / quota */ }
    if (visualizeLink) visualizeLink.setAttribute('href', '/v/' + gid + '?stage=forward');
    if (kernelsLink) kernelsLink.setAttribute('href', '/k/' + gid);
  }

  let msg = 'imported ' + nodeCount + ' nodes';
  if (unsupCount > 0) {
    msg += ', ' + unsupCount + ' unsupported op';
    if (unsupCount !== 1) msg += 's';
  }
  announce(msg);
}

// ── doctor view (W9) ─────────────────────────────────────────────────────
// Spec: notes/anneal_web_spec.md §5.8.
//
// Two cards: the native binary's adapter via GET /api/device, and the
// browser's own navigator.gpu.requestAdapter() probe run in-page. The
// browser card carries a binding caveat - the two adapters are independent
// enumerations on the same machine; anneal kernels do NOT run in the
// browser's WebGPU. The caveat is a real <p>, not aria-hidden, because it
// is important context.
//
// navigator.gpu is NOT available in Web Workers (it is a Window-only API
// in all current browsers); the probe runs on the main thread here. If
// the browser does not expose navigator.gpu (older browsers, restrictive
// contexts), the renderer fills in a friendly fallback line.

async function renderDoctorView() {
  fillNativeCard();
  fillBrowserCard();
}

async function fillNativeCard() {
  const dl = document.getElementById('native-info');
  if (!dl) return;
  dl.innerHTML = '<dt>loading…</dt><dd></dd>';
  try {
    const resp = await fetch('/api/device', { headers: { 'Accept': 'application/json' } });
    const data = await resp.json();
    const rows = [
      ['adapter',     data.adapter_name || '?'],
      ['backend',     data.backend || '?'],
      ['os / arch',   (data.os || '?') + ' / ' + (data.arch || '?')],
      ['anneal',      data.anneal_version || '?'],
      ['shader-f16',  data.shader_f16 ? 'yes' : 'no'],
      ['max storage', formatBytes(data.max_storage_buffer_binding_size)],
    ];
    if (data.error) {
      rows.push(['error', data.error]);
    }
    dl.innerHTML = '';
    for (const [k, v] of rows) {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = String(v);
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
  } catch (e) {
    dl.innerHTML = '';
    const dt = document.createElement('dt');
    dt.textContent = 'error';
    const dd = document.createElement('dd');
    dd.textContent = (e && e.message) || String(e);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
}

async function fillBrowserCard() {
  const dl = document.getElementById('browser-info');
  if (!dl) return;
  dl.innerHTML = '<dt>requesting…</dt><dd></dd>';
  if (!('gpu' in navigator)) {
    dl.innerHTML = '';
    const dt = document.createElement('dt');
    dt.textContent = 'webgpu';
    const dd = document.createElement('dd');
    dd.textContent = 'not available in this browser';
    dl.appendChild(dt);
    dl.appendChild(dd);
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      dl.innerHTML = '';
      const dt = document.createElement('dt');
      dt.textContent = 'adapter';
      const dd = document.createElement('dd');
      dd.textContent = 'navigator.gpu present but no adapter granted';
      dl.appendChild(dt);
      dl.appendChild(dd);
      return;
    }
    // Adapter info: name, architecture, vendor (where exposed). Newer
    // browsers expose adapter.info; older ones expose requestAdapterInfo().
    let info = adapter.info;
    if (!info && typeof adapter.requestAdapterInfo === 'function') {
      try { info = await adapter.requestAdapterInfo(); } catch (_) {}
    }
    const features = [];
    if (adapter.features && adapter.features.forEach) {
      adapter.features.forEach((f) => features.push(f));
    }
    const rows = [
      ['vendor',       (info && info.vendor) || '?'],
      ['architecture', (info && info.architecture) || '?'],
      ['device',       (info && info.device) || '?'],
      ['description',  (info && info.description) || '?'],
      ['shader-f16',   features.indexOf('shader-f16') >= 0 ? 'yes' : 'no'],
    ];
    if (adapter.limits) {
      rows.push(['max storage', formatBytes(adapter.limits.maxStorageBufferBindingSize)]);
    }
    dl.innerHTML = '';
    for (const [k, v] of rows) {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = String(v);
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
  } catch (e) {
    dl.innerHTML = '';
    const dt = document.createElement('dt');
    dt.textContent = 'error';
    const dd = document.createElement('dd');
    dd.textContent = (e && e.message) || String(e);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
}

function formatBytes(n) {
  if (typeof n !== 'number' || !isFinite(n) || n <= 0) return '?';
  if (n >= 1024 * 1024 * 1024) return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GiB';
  if (n >= 1024 * 1024)        return (n / (1024 * 1024)).toFixed(1)        + ' MiB';
  if (n >= 1024)               return (n / 1024).toFixed(1)                 + ' KiB';
  return n + ' B';
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
  initExplainSearch();
  initExplainKeyboard();
  initExplainPlayButton();
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
  // W2 hooks (kernels view) - exported so manual console tests can drive them.
  tokenizeWGSL, renderKernelsView, selectKernel, modelFromPath,
  // W6 hooks (train view).
  renderTrainView, trainStart, trainCancel, modelFromTrainPath,
  // W7 hooks (generate view).
  renderGenerateView, generateStart, generateCancel, modelFromGenPath,
  // W4 hooks (visualize view).
  renderVisualizeView, closeNodeInspector, modelFromVizPath, nodeIdFromQuery,
  // W3 hooks (explain view).
  renderExplainView, selectOp, opFromExplainPath, drawMiniGraph,
  // W9 hooks (tensor inspect dropzone + doctor view).
  renderStudioView, inspectFile, detectInspectFormat, formatPreview,
  renderDoctorView, fillNativeCard, fillBrowserCard, formatBytes,
  // W8 hooks (ONNX dropzone).
  initStudioDropzone, importONNXFile, renderONNXSummary,
};
