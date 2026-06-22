// Tests for the visualize view (W4) in web/studio.js.
//
// Surface under test (via the __studio seam):
//   renderVisualizeView, closeNodeInspector, modelFromVizPath, nodeIdFromQuery
//
// The node inspector is populated by fetchAndShowNodeDetail, which calls
// `await wasm.call('annealNodeDetail', graphId, nodeId)` and JSON.parses the
// result string. We drive that path two ways:
//   - deep link: navigate to /v/<model>?node=<id>, mark the iframe ready
//     (embedReady), then call renderVisualizeView so the queued node is
//     fetched + shown.
//   - error branch: let wasm.call reject and assert the error render.
//
// wasm is an exported object; studio.js reads `wasm.call` by property at call
// time, so we overwrite studio.wasm.call AFTER loadStudio (per harness note).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

let studio;

// A node-detail payload shaped the way setNodeInspectorContent consumes it.
// annealNodeDetail returns a JSON *string* (the renderer JSON.parses it).
const NODE_DETAIL = {
  op: 'Mul',
  dtype: 'f32',
  shape: [4, 8],
  phase: 'forward',
  arg: 'alpha=2',
  source_file: 'tensor/ops.go',
  source_line: 412,
  parents: [
    { op: 'Const', id: 'n1', label: '2.0' },
    { op: 'Buffer', id: 'n2', label: 'x' },
  ],
  children: [
    { op: 'Add', id: 'n9', label: 'sum' },
  ],
};

// Mark the embed iframe ready so renderVisualizeView's deep-link branch fires
// the fetch synchronously (otherwise it only queues pendingNodeId).
function markEmbedReady() {
  window.dispatchEvent(new MessageEvent('message', { data: { type: 'embedReady' } }));
}

describe('modelFromVizPath', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/v/mlp' });
  });

  it('parses the model slug from /v/<model>', () => {
    expect(studio.__studio.modelFromVizPath('/v/resnet9')).toBe('resnet9');
  });

  it('decodes URI-encoded slugs', () => {
    expect(studio.__studio.modelFromVizPath('/v/gpt%2D2')).toBe('gpt-2');
  });

  it('ignores query + hash after the slug', () => {
    expect(studio.__studio.modelFromVizPath('/v/mlp?node=n4#x')).toBe('mlp');
  });

  it('defaults to mlp when the path does not match', () => {
    expect(studio.__studio.modelFromVizPath('/k/conv')).toBe('mlp');
  });
});

describe('nodeIdFromQuery', () => {
  it('returns the ?node= param', async () => {
    studio = await loadStudio({ path: '/v/mlp?node=n42' });
    expect(studio.__studio.nodeIdFromQuery()).toBe('n42');
  });

  it('returns null when ?node= is absent', async () => {
    studio = await loadStudio({ path: '/v/mlp' });
    expect(studio.__studio.nodeIdFromQuery()).toBeNull();
  });
});

describe('renderVisualizeView - deep-link success path', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/v/mlp?node=n7' });
    studio.wasm.call = vi
      .fn()
      .mockImplementation((fn, graphId, nodeId) =>
        Promise.resolve(JSON.stringify(NODE_DETAIL))
      );
  });

  it('opens the inspector and renders node detail from annealNodeDetail', async () => {
    // Arm listeners + queue the pending node, then signal the iframe ready so
    // the queued node is fetched and shown.
    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();

    const drawer = document.getElementById('node-inspector');
    expect(drawer.hidden).toBe(false);

    // wasm.call was invoked with the model + node from the URL.
    expect(studio.wasm.call).toHaveBeenCalledWith('annealNodeDetail', 'mlp', 'n7');

    // Detail fields rendered.
    expect(document.getElementById('node-inspector-op').textContent).toBe('Mul');
    expect(document.getElementById('ni-dtype').textContent).toBe('f32');
    expect(document.getElementById('ni-shape').textContent).toBe('[4,8]');
    expect(document.getElementById('ni-phase').textContent).toBe('forward');
    expect(document.getElementById('ni-arg').textContent).toBe('alpha=2');
    expect(document.getElementById('ni-source').textContent).toBe('tensor/ops.go:412');

    // Relation lists.
    const parents = document.getElementById('ni-parents').querySelectorAll('li');
    expect(parents.length).toBe(2);
    expect(parents[0].textContent).toContain('Const');
    expect(parents[0].textContent).toContain('n1');
    expect(parents[0].textContent).toContain('(2.0)');
    const children = document.getElementById('ni-children').querySelectorAll('li');
    expect(children.length).toBe(1);
    expect(children[0].textContent).toContain('Add');
  });

  it('does not fetch until the iframe reports embedReady', async () => {
    studio.__studio.renderVisualizeView();
    await flushMicrotasks();
    // No embedReady yet → nothing fetched, drawer still hidden.
    expect(studio.wasm.call).not.toHaveBeenCalled();
    expect(document.getElementById('node-inspector').hidden).toBe(true);
  });
});

describe('closeNodeInspector', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/v/mlp?node=n7' });
    studio.wasm.call = vi
      .fn()
      .mockResolvedValue(JSON.stringify(NODE_DETAIL));
  });

  it('hides the drawer after it was opened', async () => {
    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();
    const drawer = document.getElementById('node-inspector');
    expect(drawer.hidden).toBe(false);

    studio.__studio.closeNodeInspector();
    expect(drawer.hidden).toBe(true);
  });

  it('is a no-op safe call when the drawer is already closed', () => {
    const drawer = document.getElementById('node-inspector');
    expect(drawer.hidden).toBe(true);
    expect(() => studio.__studio.closeNodeInspector()).not.toThrow();
    expect(drawer.hidden).toBe(true);
  });

  it('close button click closes the drawer (wired via renderVisualizeView)', async () => {
    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();
    expect(document.getElementById('node-inspector').hidden).toBe(false);

    document.getElementById('node-inspector-close').click();
    expect(document.getElementById('node-inspector').hidden).toBe(true);
  });
});

describe('renderVisualizeView - error branch', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/v/mlp?node=n3' });
  });

  it('renders the error state when wasm.call rejects (no worker)', async () => {
    studio.wasm.call = vi
      .fn()
      .mockRejectedValue(new Error('wasm worker not loaded'));

    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();

    // Drawer is opened (loading placeholder) then flipped to the error render.
    expect(document.getElementById('node-inspector').hidden).toBe(false);
    expect(document.getElementById('node-inspector-op').textContent).toBe('error');
    expect(document.getElementById('ni-dtype').textContent).toBe('wasm worker not loaded');
    // Error render clears the relation lists.
    expect(document.getElementById('ni-parents').querySelectorAll('li').length).toBe(0);
    expect(document.getElementById('ni-children').querySelectorAll('li').length).toBe(0);
  });

  it('renders an error when annealNodeDetail returns invalid JSON', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue('{not json');

    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();

    expect(document.getElementById('node-inspector-op').textContent).toBe('error');
    expect(document.getElementById('ni-dtype').textContent).toContain('invalid JSON');
  });

  it('renders an error when the payload carries an error field', async () => {
    studio.wasm.call = vi
      .fn()
      .mockResolvedValue(JSON.stringify({ error: 'no such node n3' }));

    studio.__studio.renderVisualizeView();
    markEmbedReady();
    await flushMicrotasks();

    expect(document.getElementById('node-inspector-op').textContent).toBe('error');
    expect(document.getElementById('ni-dtype').textContent).toBe('no such node n3');
  });
});
