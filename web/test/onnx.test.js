// W8 ONNX dropzone tests (studio home view).
//
// Covers importONNXFile (mocked wasm.call RPC) happy + every error branch,
// initStudioDropzone event wiring for the onnx zone, and renderONNXSummary
// (fixture -> summary DOM, unsupported-op list, deep-link hrefs, sessionStorage
// stash).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

let studio;
let S;

function makeFile(name, bytes = new Uint8Array([8, 1, 16, 2]), opts = {}) {
  const file = new File([bytes], name, { type: 'application/octet-stream' });
  if (opts.failRead) {
    file.arrayBuffer = () => Promise.reject(new Error('disk-gone'));
  } else {
    file.arrayBuffer = () => Promise.resolve(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
  return file;
}

beforeEach(async () => {
  studio = await loadStudio({ path: '/' });
  S = studio.__studio;
  try { sessionStorage.clear(); } catch (_) { /* shim */ }
});

describe('initStudioDropzone (onnx) wiring', () => {
  it('dragover/dragleave toggle drag-over on the onnx zone', () => {
    S.initStudioDropzone();
    const zone = document.getElementById('onnx-dropzone');
    const over = new Event('dragover', { bubbles: true, cancelable: true });
    zone.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(true);
    expect(zone.classList.contains('drag-over')).toBe(true);
    zone.dispatchEvent(new Event('dragleave', { bubbles: true }));
    expect(zone.classList.contains('drag-over')).toBe(false);
  });

  it('drop dispatches the first file to importONNXFile', async () => {
    S.initStudioDropzone();
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({
      node_count: 3, initializer_count: 1,
    }));
    const zone = document.getElementById('onnx-dropzone');
    const ev = new Event('drop', { bubbles: true, cancelable: true });
    ev.dataTransfer = { files: [makeFile('net.onnx')] };
    zone.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    await flushMicrotasks();
    expect(studio.wasm.call).toHaveBeenCalledWith('annealImportONNX', expect.any(Uint8Array));
    expect(document.getElementById('onnx-result').hidden).toBe(false);
  });

  it('clicking the zone (not the button/picker) opens the picker', () => {
    S.initStudioDropzone();
    const zone = document.getElementById('onnx-dropzone');
    const picker = document.getElementById('onnx-picker');
    const spy = vi.spyOn(picker, 'click');
    zone.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('Enter on the zone opens the picker but space on the button is left native', () => {
    S.initStudioDropzone();
    const zone = document.getElementById('onnx-dropzone');
    const picker = document.getElementById('onnx-picker');
    const btn = document.getElementById('onnx-picker-btn');
    const spy = vi.spyOn(picker, 'click');
    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
    zone.dispatchEvent(enter);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(enter.defaultPrevented).toBe(true);
    // Space targeted at the button is ignored by the zone handler (native button activation).
    const space = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
    Object.defineProperty(space, 'target', { value: btn });
    zone.dispatchEvent(space);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('picker change event imports the selected file', async () => {
    S.initStudioDropzone();
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ node_count: 1, initializer_count: 0 }));
    const picker = document.getElementById('onnx-picker');
    Object.defineProperty(picker, 'files', { value: [makeFile('p.onnx')], configurable: true });
    picker.dispatchEvent(new Event('change', { bubbles: true }));
    await flushMicrotasks();
    expect(studio.wasm.call).toHaveBeenCalledWith('annealImportONNX', expect.any(Uint8Array));
  });

  it('pick button click opens the picker without triggering the zone handler', () => {
    S.initStudioDropzone();
    const picker = document.getElementById('onnx-picker');
    const spy = vi.spyOn(picker, 'click');
    const btn = document.getElementById('onnx-picker-btn');
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    // Zone listener bails on e.target === pickerBtn; button listener clicks once.
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

describe('importONNXFile', () => {
  it('strips the .onnx extension for display and renders summary', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({
      node_count: 12, initializer_count: 4, opset: 17, graph_id: 'g1',
    }));
    await S.importONNXFile(makeFile('ResNet.onnx'));
    expect(document.getElementById('onnx-model-name').textContent).toBe('ResNet');
    expect(document.getElementById('onnx-summary').textContent)
      .toBe('12 nodes · 4 initializers · opset 17');
    expect(document.getElementById('onnx-result').hidden).toBe(false);
  });

  it('reports read error when arrayBuffer rejects', async () => {
    studio.wasm.call = vi.fn();
    await S.importONNXFile(makeFile('a.onnx', new Uint8Array([1]), { failRead: true }));
    expect(studio.wasm.call).not.toHaveBeenCalled();
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('read error: disk-gone');
  });

  it('reports wasm-not-loaded when wasm.call rejects', async () => {
    studio.wasm.call = vi.fn().mockRejectedValue(new Error('worker not loaded'));
    await S.importONNXFile(makeFile('a.onnx'));
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('wasm not loaded');
  });

  it('reports invalid JSON', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue('}{garbage');
    await S.importONNXFile(makeFile('a.onnx'));
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('invalid JSON');
  });

  it('reports a parse error payload', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ error: 'malformed protobuf' }));
    await S.importONNXFile(makeFile('a.onnx'));
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('parse error: malformed protobuf');
  });
});

describe('renderONNXSummary', () => {
  it('renders a plain summary with zero counts defaulting', () => {
    S.renderONNXSummary({}, 'm');
    expect(document.getElementById('onnx-summary').textContent)
      .toBe('0 nodes · 0 initializers');
    expect(document.getElementById('onnx-unsupported-section').hidden).toBe(true);
  });

  it('appends opset and note when present', () => {
    S.renderONNXSummary({
      node_count: 5, initializer_count: 2, opset: 13, note: 'structure-only',
    }, 'm');
    expect(document.getElementById('onnx-summary').textContent)
      .toBe('5 nodes · 2 initializers · opset 13 · structure-only');
  });

  it('singular vs plural unsupported-op wording', () => {
    S.renderONNXSummary({
      node_count: 1, initializer_count: 0,
      unsupported_ops: [{ op_type: 'Foo', count: 1 }],
    }, 'm');
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('1 unsupported op');
    expect(document.getElementById('onnx-summary').textContent)
      .not.toContain('1 unsupported ops');
  });

  it('renders the unsupported-op list with names, counts and reasons', () => {
    S.renderONNXSummary({
      node_count: 10, initializer_count: 3,
      unsupported_ops: [
        { op_type: 'GridSample', count: 2, reason: 'not implemented' },
        { op_type: 'Loop' }, // count<=1, default reason
      ],
    }, 'm');
    const sec = document.getElementById('onnx-unsupported-section');
    expect(sec.hidden).toBe(false);
    const items = document.querySelectorAll('#onnx-unsupported-list li');
    expect(items.length).toBe(2);
    expect(items[0].querySelector('.op-name').textContent).toBe('GridSample (2)');
    expect(items[0].querySelector('.op-reason').textContent).toBe(' - not implemented');
    expect(items[1].querySelector('.op-name').textContent).toBe('Loop');
    expect(items[1].querySelector('.op-reason').textContent).toBe(' - no handler registered');
    expect(document.getElementById('onnx-summary').textContent)
      .toContain('2 unsupported ops');
  });

  it('sets deep-link hrefs and stashes the summary in sessionStorage', () => {
    S.renderONNXSummary({
      node_count: 7, initializer_count: 1, graph_id: 'abc123',
    }, 'mynet');
    expect(document.getElementById('onnx-visualize-link').getAttribute('href'))
      .toBe('/v/abc123?stage=forward');
    expect(document.getElementById('onnx-kernels-link').getAttribute('href'))
      .toBe('/k/abc123');
    const stash = JSON.parse(sessionStorage.getItem('anneal-imported-abc123'));
    expect(stash.display_name).toBe('mynet');
    expect(stash.node_count).toBe(7);
  });

  it('hides the unsupported section when an earlier import had unsupported ops', () => {
    S.renderONNXSummary({
      node_count: 1, initializer_count: 0,
      unsupported_ops: [{ op_type: 'X', count: 1 }],
    }, 'a');
    expect(document.getElementById('onnx-unsupported-section').hidden).toBe(false);
    S.renderONNXSummary({ node_count: 1, initializer_count: 0 }, 'b');
    expect(document.getElementById('onnx-unsupported-section').hidden).toBe(true);
  });

  it('is a no-op on null payload', () => {
    expect(() => S.renderONNXSummary(null, 'x')).not.toThrow();
  });
});
