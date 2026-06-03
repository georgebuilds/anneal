// anneal studio — main thread ES module.
//
// Responsibilities (DESIGN.md §3, §5, §7; spec §10):
//   1. History API routing for every studio view (no hash routing).
//   2. Theme controller: system | dark | light cycle, persisted, with a live
//      matchMedia listener so OS theme changes apply without a page reload.
//   3. View renderer dispatch — stubs in W0, real renderers land in W1+.
//   4. Keyboard handlers: `/` focuses search; `g <dest>` chord jumps view.
//   5. Worker RPC client — gated behind a <meta name="anneal-worker"> tag so
//      W0 (no anneal.wasm shipped yet) does not 404. Once a real WASM build
//      lands, flip the meta tag on and `wasm.call(fn, ...args)` lights up.
//   6. Ignite-once wordmark animation on first page load only.
//
// This file is intentionally small and greppable. Each section under one
// header. No third-party imports. No bundler. Pure browser APIs.

'use strict';

// ── theme controller (DESIGN.md §5) ───────────────────────────────────────
// Cycle order: system → dark → light → system. Default: system. Persisted in
// localStorage['anneal-theme']. When `system` is selected, a matchMedia
// listener re-renders on OS theme change without a reload (spec §10 fix).

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

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const btn = document.getElementById('themeToggle');
  if (btn) {
    btn.title = 'theme: ' + theme + ' (click to cycle)';
    btn.setAttribute('aria-label', 'theme is ' + theme + '; click to cycle');
  }
}

function cycleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'system';
  const idx = THEMES.indexOf(cur);
  const next = THEMES[(idx + 1) % THEMES.length];
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

function viewIdForPath(pathname) {
  for (const r of ROUTES) if (r.test(pathname)) return r.view;
  return 'studio';
}

function setActiveView(viewId) {
  document.querySelectorAll('.view').forEach((el) => {
    el.classList.toggle('active', el.id === 'view-' + viewId);
  });
  document.querySelectorAll('.nav-item').forEach((el) => {
    el.classList.toggle('active', el.dataset.view === viewId);
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

// ── view renderer dispatch ────────────────────────────────────────────────
// Each renderer hydrates one view's mount point. W0 renderers are no-ops;
// they exist so the dispatch machinery is real and the W1+ renderers can
// drop in by assignment.

const RENDERERS = {
  studio:    function renderStudio()    { /* W1+ */ },
  visualize: function renderVisualize() { /* W2 */ },
  kernels:   function renderKernels()   { /* W3 */ },
  explain:   function renderExplain()   { /* W4 */ },
  train:     function renderTrain()     { /* W5 */ },
  generate:  function renderGenerate()  { /* W6 */ },
  history:   function renderHistory()   { /* W8 */ },
  doctor:    function renderDoctor()    { /* W7 */ },
};

// ── keyboard (DESIGN.md §7; spec §10) ─────────────────────────────────────
// `/` focuses the search input. `g <key>` chord jumps to a view within 1s.
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
    // Never intercept while the user is typing.
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

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

function initIgnite() {
  let already = false;
  try { already = localStorage.getItem('anneal-ignited') === '1'; } catch (_) {}
  const mark = document.getElementById('brand-mark');
  if (!mark) return;
  if (!already) {
    mark.classList.add('ignite');
    try { localStorage.setItem('anneal-ignited', '1'); } catch (_) {}
  }
}

// ── boot ──────────────────────────────────────────────────────────────────

function boot() {
  initTheme();
  initRouting();
  initKeyboard();
  initIgnite();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

// Exported for tests / consoles. Not a public API; W1+ may rename.
export const __studio = { navigate, applyTheme, cycleTheme, viewIdForPath };
