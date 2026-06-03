// anneal studio — Web Worker scaffold.
//
// Loads wasm_exec.js, instantiates anneal.wasm, listens for {id, fn, args}
// messages from the main thread, and dispatches to globalThis[fn](...args).
// Responds with {id, ok, result|error}.
//
// All RPC functions are registered Go-side via syscall/js
// (e.g. js.Global().Set("annealGetGraph", ...)). This file does not own the
// handler table; it only forwards. See web/README-DEV.md and
// notes/anneal_web_spec.md §4 for the exported function contract.
//
// W0 ships without anneal.wasm. studio.js only instantiates this worker when
// <meta name="anneal-worker"> is present; otherwise no fetch happens and no
// 404 appears in the console.

'use strict';

self.importScripts('/static/wasm_exec.js');

// ── boot WASM ─────────────────────────────────────────────────────────────

let wasmReady = false;
let wasmReadyResolve;
const wasmReadyPromise = new Promise((res) => { wasmReadyResolve = res; });

(async function bootWasm() {
  try {
    const go = new Go(); // eslint-disable-line no-undef
    const result = await WebAssembly.instantiateStreaming(
      fetch('/static/anneal.wasm'),
      go.importObject
    );
    go.run(result.instance); // returns once the Go program exits; long-lived for js/wasm.
    wasmReady = true;
    wasmReadyResolve();
    // Notify the main thread that handlers are live. id=0 is reserved for
    // unsolicited worker → main events.
    self.postMessage({ id: 0, ok: true, result: { event: 'ready' } });
  } catch (e) {
    // Surface the boot failure to the main thread. Subsequent calls will
    // reject with the same error.
    wasmReadyResolve();
    self.postMessage({
      id: 0,
      ok: false,
      error: 'wasm boot failed: ' + (e && e.message ? e.message : String(e)),
    });
  }
})();

// ── RPC dispatch ──────────────────────────────────────────────────────────

self.onmessage = async (ev) => {
  const { id, fn, args } = ev.data || {};
  if (id == null || typeof fn !== 'string') {
    self.postMessage({ id: id ?? 0, ok: false, error: 'malformed RPC request' });
    return;
  }
  // Wait for WASM boot before dispatching the first call.
  if (!wasmReady) await wasmReadyPromise;
  try {
    const handler = self[fn];
    if (typeof handler !== 'function') {
      throw new Error('unknown RPC function: ' + fn);
    }
    const argArray = Array.isArray(args) ? args : [];
    const result = handler.apply(null, argArray);
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({
      id,
      ok: false,
      error: e && e.message ? e.message : String(e),
    });
  }
};
