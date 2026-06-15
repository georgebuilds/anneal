// web/worker.js coverage: WASM boot IIFE (ready + boot-failure), and the
// self.onmessage RPC dispatch (malformed, unknown fn, known fn, throwing
// handler). worker.js reads self/Go/WebAssembly/fetch as globals at import
// time, so every global is stubbed BEFORE importing, and a fresh module +
// fresh fake self is built per test via vi.resetModules().
import { describe, it, expect, afterEach, vi } from 'vitest';

const flush = () => new Promise((r) => setTimeout(r, 0));

// Saved real globals to restore after each test (worker.js mutates self.onmessage
// and we replace self/Go/WebAssembly/fetch wholesale).
let saved;

function snapshotGlobals() {
  saved = {
    self: globalThis.self,
    Go: globalThis.Go,
    WebAssembly: globalThis.WebAssembly,
    fetch: globalThis.fetch,
    hasSelf: Object.prototype.hasOwnProperty.call(globalThis, 'self'),
    hasGo: Object.prototype.hasOwnProperty.call(globalThis, 'Go'),
    hasFetch: Object.prototype.hasOwnProperty.call(globalThis, 'fetch'),
  };
}

function makeFakeSelf() {
  const fakeSelf = {
    importScripts: vi.fn(),
    postMessage: vi.fn(),
    onmessage: null,
  };
  globalThis.self = fakeSelf;
  return fakeSelf;
}

// installEnv wires the global stubs worker.js needs at import time. wasmOk:
// when false, instantiateStreaming rejects to drive the boot-failure branch.
function installEnv({ wasmOk = true, runImpl } = {}) {
  const fakeSelf = makeFakeSelf();

  const goRun = vi.fn(runImpl || (() => {}));
  globalThis.Go = class FakeGo {
    constructor() { this.importObject = { env: {} }; }
    run(instance) { return goRun(instance); }
  };

  globalThis.fetch = vi.fn(() => Promise.resolve({ /* fake Response */ }));

  globalThis.WebAssembly = {
    instantiateStreaming: vi.fn(() =>
      wasmOk
        ? Promise.resolve({ instance: {} })
        : Promise.reject(new Error('bad magic word'))
    ),
  };

  return { fakeSelf, goRun };
}

async function importWorker() {
  vi.resetModules();
  await import('../worker.js');
}

afterEach(() => {
  vi.restoreAllMocks();
  if (saved) {
    if (saved.hasSelf) globalThis.self = saved.self; else delete globalThis.self;
    if (saved.hasGo) globalThis.Go = saved.Go; else delete globalThis.Go;
    globalThis.WebAssembly = saved.WebAssembly;
    if (saved.hasFetch) globalThis.fetch = saved.fetch; else delete globalThis.fetch;
    saved = null;
  }
});

describe('worker.js boot', () => {
  it('importScripts wasm_exec.js and posts the ready event', async () => {
    snapshotGlobals();
    const { fakeSelf, goRun } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();

    expect(fakeSelf.importScripts).toHaveBeenCalledWith('/static/wasm_exec.js');
    expect(globalThis.WebAssembly.instantiateStreaming).toHaveBeenCalled();
    expect(goRun).toHaveBeenCalledWith({});
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 0, ok: true, result: { event: 'ready' },
    });
  });

  it('posts a "wasm boot failed" error when instantiateStreaming rejects', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: false });
    await importWorker();
    await flush();

    const calls = fakeSelf.postMessage.mock.calls.map((c) => c[0]);
    const fail = calls.find((m) => m.ok === false);
    expect(fail).toBeTruthy();
    expect(fail.id).toBe(0);
    expect(fail.error).toMatch(/^wasm boot failed: bad magic word/);
    // The ready event must NOT have been posted on the failure path.
    expect(calls.some((m) => m.ok === true)).toBe(false);
  });
});

describe('worker.js RPC dispatch', () => {
  it('rejects a malformed request (no fn) with "malformed RPC request"', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    await fakeSelf.onmessage({ data: { id: 5 } }); // missing fn
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 5, ok: false, error: 'malformed RPC request',
    });
  });

  it('uses id 0 when a malformed request omits id too', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    await fakeSelf.onmessage({ data: {} });
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 0, ok: false, error: 'malformed RPC request',
    });
  });

  it('rejects an unknown RPC function', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    await fakeSelf.onmessage({ data: { id: 9, fn: 'doesNotExist', args: [] } });
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 9, ok: false, error: 'unknown RPC function: doesNotExist',
    });
  });

  it('dispatches a known fn and posts its result', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    // Handlers are registered on self by the Go runtime; emulate that.
    fakeSelf.add = (a, b) => a + b;
    await fakeSelf.onmessage({ data: { id: 3, fn: 'add', args: [40, 2] } });
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 3, ok: true, result: 42,
    });
  });

  it('defaults args to [] when not an array', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    fakeSelf.argc = (...xs) => xs.length;
    await fakeSelf.onmessage({ data: { id: 4, fn: 'argc' } }); // no args field
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 4, ok: true, result: 0,
    });
  });

  it('posts the error message when a handler throws', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    fakeSelf.kaboom = () => { throw new Error('handler exploded'); };
    await fakeSelf.onmessage({ data: { id: 7, fn: 'kaboom', args: [] } });
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 7, ok: false, error: 'handler exploded',
    });
  });

  it('handles a null ev.data without crashing (malformed branch)', async () => {
    snapshotGlobals();
    const { fakeSelf } = installEnv({ wasmOk: true });
    await importWorker();
    await flush();
    fakeSelf.postMessage.mockClear();

    await fakeSelf.onmessage({ data: null });
    expect(fakeSelf.postMessage).toHaveBeenCalledWith({
      id: 0, ok: false, error: 'malformed RPC request',
    });
  });
});
