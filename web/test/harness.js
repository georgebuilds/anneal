// Shared test harness for the anneal studio frontend.
//
// Importing studio.js runs boot() against the live DOM, so every test must
// install the studio.html body BEFORE importing the module. Use loadStudio()
// for that ordering; it returns the freshly evaluated module namespace
// (including the __studio test seam).
//
// Read-only for subagent test files: import these helpers, do not edit this
// file (keeps parallel test authoring conflict-free). If you need a bespoke
// fixture, build it inline in your own *.test.js.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { vi } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));
const STUDIO_HTML = join(here, '..', 'studio.html');

let bodyHTML = null;

// bodyFixture returns the inner HTML of <body> from the real studio.html, with
// the module <script> stripped (innerHTML never executes it, but dropping it
// keeps the fixture honest). Cached after first read.
export function bodyFixture() {
  if (bodyHTML !== null) return bodyHTML;
  const raw = readFileSync(STUDIO_HTML, 'utf8');
  const m = raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  let inner = m ? m[1] : raw;
  inner = inner.replace(/<script[\s\S]*?<\/script>/gi, '');
  bodyHTML = inner;
  return bodyHTML;
}

// installDOM resets document/body/head to the studio.html skeleton and clears
// localStorage. Pass { theme } to seed the data-theme attribute, or { path }
// to seed window.location via history.replaceState.
export function installDOM(opts = {}) {
  document.documentElement.setAttribute('data-theme', opts.theme || 'system');
  document.head.innerHTML = '<title>anneal studio</title>';
  document.body.innerHTML = bodyFixture();
  try {
    localStorage.clear();
  } catch (_) { /* private mode shim */ }
  if (opts.path) {
    window.history.replaceState({}, '', opts.path);
  } else {
    window.history.replaceState({}, '', '/');
  }
}

// loadStudio installs the DOM fixture, drops the module cache so studio.js
// re-evaluates (re-running boot() against the fresh DOM), and returns the
// module namespace. Always await it inside beforeEach for per-test isolation.
export async function loadStudio(opts = {}) {
  installDOM(opts);
  vi.resetModules();
  return import('../studio.js');
}

// flushMicrotasks resolves pending promise jobs (worker RPC mocks, etc.).
export function flushMicrotasks() {
  return new Promise((r) => setTimeout(r, 0));
}
