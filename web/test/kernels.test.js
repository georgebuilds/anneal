// Tests for the kernels view (W2) in web/studio.js.
//
// renderKernelsView() fetches a KernelSet via `await wasm.call('annealGetKernels',
// model)` (which returns a JSON *string*), parses it, then draws the list +
// detail. studio.js reads wasm.call by property at call time and `wasm` is an
// exported mutable object, so after loadStudio we replace wasm.call with a mock
// and assert the resulting DOM.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

// Accurate KernelSet fixture, inferred from drawKernelList / drawKernelDetail:
//   set.kernels[i] = { id, shape:[..], op_count, buffers_in, buffers_out,
//                      wgsl, fusion_spans:[{start_line,label}] }
//   set.error (optional) is the compiler-error channel.
function makeSet() {
  return {
    kernels: [
      {
        id: 'K0',
        shape: [16, 32],
        op_count: 5,
        buffers_in: 2,
        buffers_out: 1,
        wgsl: '@compute @workgroup_size(64)\nfn main() {\n  let x: f32 = 1.0f;\n}',
        fusion_spans: [{ start_line: 2, label: 'fwd' }],
      },
      {
        id: 'K1',
        shape: [8],
        op_count: 3,
        buffers_in: 1,
        buffers_out: 1,
        wgsl: 'fn k1() { return; }',
        fusion_spans: [],
      },
      {
        id: 'K2',
        shape: [],
        op_count: 9,
        buffers_in: 3,
        buffers_out: 2,
        wgsl: '// k2\nvar a = 0u;',
        fusion_spans: [{ start_line: 1, label: 'bwd' }],
      },
    ],
  };
}

// renderKernelsView reads the model off window.location.pathname and short-
// circuits when _kernelsState.model already matches. loadStudio gives a fresh
// module each time, so state is clean per test.
async function loadAt(path) {
  const studio = await loadStudio({ path });
  return studio;
}

describe('modelFromPath', () => {
  let modelFromPath;
  beforeEach(async () => {
    const studio = await loadStudio();
    modelFromPath = studio.__studio.modelFromPath;
  });

  it('extracts the model from /k/<model>', () => {
    expect(modelFromPath('/k/resnet')).toBe('resnet');
  });

  it('stops at a query string', () => {
    expect(modelFromPath('/k/gpt2?kernel=K3')).toBe('gpt2');
  });

  it('url-decodes the model segment', () => {
    expect(modelFromPath('/k/my%20model')).toBe('my model');
  });

  it('defaults to mlp when the path is not a /k/ route', () => {
    expect(modelFromPath('/')).toBe('mlp');
    expect(modelFromPath('/t/mlp')).toBe('mlp');
  });
});

describe('renderKernelsView - success path', () => {
  let studio;
  beforeEach(async () => {
    studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify(makeSet()));
  });

  it('calls annealGetKernels with the model from the path', async () => {
    await studio.__studio.renderKernelsView();
    expect(studio.wasm.call).toHaveBeenCalledWith('annealGetKernels', 'mlp');
  });

  it('renders one list item per kernel with id + meta', async () => {
    await studio.__studio.renderKernelsView();
    const items = document.getElementById('kernel-list-items').children;
    expect(items).toHaveLength(3);

    const first = items[0];
    expect(first.id).toBe('k-opt-K0');
    expect(first.getAttribute('role')).toBe('option');
    expect(first.querySelector('.k-id').textContent).toBe('K0');
    expect(first.querySelector('.k-meta').textContent)
      .toBe('5 ops · 2 in / 1 out · [16,32]');
  });

  it('omits the shape clause when shape is empty', async () => {
    await studio.__studio.renderKernelsView();
    const third = document.getElementById('kernel-list-items').children[2];
    expect(third.querySelector('.k-meta').textContent).toBe('9 ops · 3 in / 2 out');
  });

  it('selects the first kernel by default and fills the detail pane', async () => {
    await studio.__studio.renderKernelsView();
    const items = document.getElementById('kernel-list-items').children;
    expect(items[0].getAttribute('aria-selected')).toBe('true');
    expect(items[1].getAttribute('aria-selected')).toBe('false');
    expect(document.getElementById('kernel-list-items')
      .getAttribute('aria-activedescendant')).toBe('k-opt-K0');

    expect(document.getElementById('k-id').textContent).toBe('K0');
    expect(document.getElementById('k-shape').textContent).toBe('[16,32]');
    expect(document.getElementById('k-counts').textContent).toBe('5 ops · 2 in / 1 out');
  });

  it('renders highlighted WGSL into the <pre> with token spans + gutter', async () => {
    await studio.__studio.renderKernelsView();
    const pre = document.getElementById('k-wgsl');
    // One .wgsl-line span per physical line of the K0 source (4 lines).
    const lines = pre.querySelectorAll('.wgsl-line');
    expect(lines).toHaveLength(4);
    // Token classes are wrapped as tk-<kind>.
    expect(pre.querySelector('.tk-keyword')).not.toBeNull();
    expect(pre.querySelector('.tk-attribute').textContent).toBe('@compute');
    // The fusion span on start_line 2 stamps a gutter labeled "fwd".
    const gutter = pre.querySelector('.gutter.fwd');
    expect(gutter).not.toBeNull();
    expect(gutter.textContent).toBe('fwd');
    expect(gutter.getAttribute('aria-hidden')).toBe('true');
  });
});

describe('renderKernelsView - deep-link ?kernel=', () => {
  it('pre-selects the kernel named in the query string', async () => {
    const studio = await loadAt('/k/mlp?kernel=K2');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify(makeSet()));
    await studio.__studio.renderKernelsView();

    const items = document.getElementById('kernel-list-items').children;
    expect(items[2].getAttribute('aria-selected')).toBe('true');
    expect(items[0].getAttribute('aria-selected')).toBe('false');
    expect(document.getElementById('k-id').textContent).toBe('K2');
    expect(document.getElementById('k-counts').textContent).toBe('9 ops · 3 in / 2 out');
  });

  it('falls back to the first kernel when ?kernel= is unknown', async () => {
    const studio = await loadAt('/k/mlp?kernel=NOPE');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify(makeSet()));
    await studio.__studio.renderKernelsView();
    expect(document.getElementById('k-id').textContent).toBe('K0');
  });
});

describe('selectKernel - switching the detail pane', () => {
  let studio;
  beforeEach(async () => {
    studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify(makeSet()));
    await studio.__studio.renderKernelsView();
  });

  it('updates aria-selected, the detail pane, and the URL', () => {
    studio.__studio.selectKernel(1, {});
    const items = document.getElementById('kernel-list-items').children;
    expect(items[1].getAttribute('aria-selected')).toBe('true');
    expect(items[0].getAttribute('aria-selected')).toBe('false');

    expect(document.getElementById('k-id').textContent).toBe('K1');
    expect(document.getElementById('k-shape').textContent).toBe('[8]');
    expect(document.getElementById('k-counts').textContent).toBe('3 ops · 1 in / 1 out');

    expect(window.location.pathname).toBe('/k/mlp');
    expect(window.location.search).toBe('?kernel=K1');
  });

  it('a click on a list item selects that kernel', () => {
    const items = document.getElementById('kernel-list-items').children;
    items[2].click();
    expect(document.getElementById('k-id').textContent).toBe('K2');
    expect(items[2].getAttribute('aria-selected')).toBe('true');
  });

  it('ignores out-of-range indices', () => {
    studio.__studio.selectKernel(99, {});
    // Still showing the default first kernel.
    expect(document.getElementById('k-id').textContent).toBe('K0');
  });
});

describe('renderKernelsView - loading placeholder', () => {
  it('shows the loading placeholder before the worker resolves', async () => {
    const studio = await loadAt('/k/mlp');
    let resolveCall;
    studio.wasm.call = vi.fn(() => new Promise((r) => { resolveCall = r; }));

    const p = studio.__studio.renderKernelsView();
    // Synchronously after kicking off the fetch, the placeholder is present.
    const li = document.querySelector('#kernel-list-items .kernel-list-loading');
    expect(li).not.toBeNull();
    expect(li.textContent).toBe('loading kernels…');
    expect(li.getAttribute('aria-live')).toBe('polite');

    resolveCall(JSON.stringify(makeSet()));
    await p;
    // Placeholder is replaced once data arrives.
    expect(document.querySelector('#kernel-list-items .kernel-list-loading')).toBeNull();
    expect(document.getElementById('kernel-list-items').children).toHaveLength(3);
  });
});

describe('renderKernelsView - error branches', () => {
  it('renders the wasm-not-loaded message when wasm.call rejects', async () => {
    const studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockRejectedValue(new Error('worker not loaded'));
    await studio.__studio.renderKernelsView();

    const err = document.querySelector('#kernel-list-items .kernel-list-error');
    expect(err).not.toBeNull();
    expect(err.textContent).toBe('wasm not loaded - build anneal.wasm to populate this view');
    // Detail pane is untouched / empty.
    expect(document.getElementById('k-id').textContent).toBe('');
  });

  it('falls back to the rejecting default wasm.call (no worker meta)', async () => {
    // Do not override wasm.call: the real RPC rejects with "worker not loaded".
    const studio = await loadAt('/k/mlp');
    await studio.__studio.renderKernelsView();
    await flushMicrotasks();
    const err = document.querySelector('#kernel-list-items .kernel-list-error');
    expect(err).not.toBeNull();
    expect(err.textContent).toContain('build anneal.wasm');
  });

  it('renders an invalid-JSON message when the payload will not parse', async () => {
    const studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue('{ not json');
    await studio.__studio.renderKernelsView();
    const err = document.querySelector('#kernel-list-items .kernel-list-error');
    expect(err).not.toBeNull();
    expect(err.textContent).toBe('kernels view: invalid JSON from compiler');
  });

  it('renders a compiler-error message when set.error is present', async () => {
    const studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ error: 'boom' }));
    await studio.__studio.renderKernelsView();
    const err = document.querySelector('#kernel-list-items .kernel-list-error');
    expect(err).not.toBeNull();
    expect(err.textContent).toBe('compiler error: boom');
  });

  it('renders the empty state for a kernel set with no kernels', async () => {
    const studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ kernels: [] }));
    await studio.__studio.renderKernelsView();
    const empty = document.querySelector('#kernel-list-items .kernel-list-empty');
    expect(empty).not.toBeNull();
    expect(empty.textContent).toBe('no kernels');
  });
});

describe('renderKernelsView - cached re-render (deep-link reselect)', () => {
  it('reuses cached data and reselects from ?kernel= without refetching', async () => {
    const studio = await loadAt('/k/mlp');
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify(makeSet()));
    await studio.__studio.renderKernelsView();
    expect(studio.wasm.call).toHaveBeenCalledTimes(1);

    // Navigate to a deep link for the same model and re-render.
    window.history.replaceState({}, '', '/k/mlp?kernel=K1');
    await studio.__studio.renderKernelsView();

    // No second worker call (cached path).
    expect(studio.wasm.call).toHaveBeenCalledTimes(1);
    expect(document.getElementById('k-id').textContent).toBe('K1');
    const items = document.getElementById('kernel-list-items').children;
    expect(items[1].getAttribute('aria-selected')).toBe('true');
  });
});
