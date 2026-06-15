// W9 tensor-inspect dropzone tests (studio home view).
//
// Covers detectInspectFormat (every branch + unknown), formatPreview,
// formatBytes boundaries, renderStudioView wiring, initStudioDropzone /
// initInspectDropzone drag-and-drop event handling, and inspectFile happy +
// error paths. wasm.call and the File API are mocked per-test.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

let studio;
let S;

// makeFile builds a jsdom File whose arrayBuffer() resolves to `bytes`.
// jsdom's File.arrayBuffer is flaky across versions, so we override it.
function makeFile(name, bytes = new Uint8Array([1, 2, 3]), opts = {}) {
  const file = new File([bytes], name, { type: 'application/octet-stream' });
  if (opts.failRead) {
    file.arrayBuffer = () => Promise.reject(new Error('boom-read'));
  } else {
    file.arrayBuffer = () => Promise.resolve(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
  return file;
}

beforeEach(async () => {
  studio = await loadStudio({ path: '/' });
  S = studio.__studio;
});

describe('detectInspectFormat', () => {
  it('detects npy', () => {
    expect(S.detectInspectFormat('weights.npy')).toBe('npy');
  });
  it('detects npz', () => {
    expect(S.detectInspectFormat('bundle.npz')).toBe('npz');
  });
  it('detects safetensors', () => {
    expect(S.detectInspectFormat('model.safetensors')).toBe('safetensors');
  });
  it('is case-insensitive', () => {
    expect(S.detectInspectFormat('MODEL.SAFETENSORS')).toBe('safetensors');
    expect(S.detectInspectFormat('A.NPY')).toBe('npy');
  });
  it('returns empty for unknown extension', () => {
    expect(S.detectInspectFormat('data.bin')).toBe('');
    expect(S.detectInspectFormat('noext')).toBe('');
  });
  it('handles null / undefined name', () => {
    expect(S.detectInspectFormat(null)).toBe('');
    expect(S.detectInspectFormat(undefined)).toBe('');
  });
});

describe('formatPreview', () => {
  it('returns [] for non-array or empty', () => {
    expect(S.formatPreview(null)).toBe('[]');
    expect(S.formatPreview([])).toBe('[]');
    expect(S.formatPreview('nope')).toBe('[]');
  });
  it('fixes mid-range numbers to 4 decimals', () => {
    expect(S.formatPreview([1, 2.5])).toBe('[1.0000, 2.5000]');
  });
  it('uses exponential for tiny magnitudes', () => {
    expect(S.formatPreview([1e-5])).toBe('[1.000e-5]');
  });
  it('uses exponential for large magnitudes', () => {
    expect(S.formatPreview([12345])).toBe('[1.235e+4]');
  });
  it('keeps zero in fixed notation (not exponential)', () => {
    expect(S.formatPreview([0])).toBe('[0.0000]');
  });
  it('stringifies non-finite / non-number values', () => {
    expect(S.formatPreview([NaN, Infinity, 'x'])).toBe('[NaN, Infinity, x]');
  });
});

describe('formatBytes', () => {
  it('returns ? for non-positive or non-number', () => {
    expect(S.formatBytes(0)).toBe('?');
    expect(S.formatBytes(-5)).toBe('?');
    expect(S.formatBytes(NaN)).toBe('?');
    expect(S.formatBytes('100')).toBe('?');
  });
  it('formats bytes below 1 KiB', () => {
    expect(S.formatBytes(512)).toBe('512 B');
  });
  it('formats KiB at the boundary', () => {
    expect(S.formatBytes(1024)).toBe('1.0 KiB');
    expect(S.formatBytes(1536)).toBe('1.5 KiB');
  });
  it('formats MiB at the boundary', () => {
    expect(S.formatBytes(1024 * 1024)).toBe('1.0 MiB');
  });
  it('formats GiB at the boundary', () => {
    expect(S.formatBytes(1024 * 1024 * 1024)).toBe('1.00 GiB');
    expect(S.formatBytes(2 * 1024 * 1024 * 1024)).toBe('2.00 GiB');
  });
});

describe('renderStudioView + dropzone wiring', () => {
  it('arms the dropzones without throwing', () => {
    expect(() => S.renderStudioView()).not.toThrow();
  });

  it('dragover adds drag-over class and is cancelled', () => {
    S.renderStudioView();
    const zone = document.getElementById('tensor-dropzone');
    const ev = new Event('dragover', { bubbles: true, cancelable: true });
    zone.dispatchEvent(ev);
    expect(zone.classList.contains('drag-over')).toBe(true);
    expect(ev.defaultPrevented).toBe(true);
  });

  it('dragleave removes drag-over class', () => {
    S.renderStudioView();
    const zone = document.getElementById('tensor-dropzone');
    zone.classList.add('drag-over');
    zone.dispatchEvent(new Event('dragleave', { bubbles: true }));
    expect(zone.classList.contains('drag-over')).toBe(false);
  });

  it('drop dispatches the first file to inspectFile', async () => {
    S.renderStudioView();
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({
      format: 'npy', tensors: [],
    }));
    const zone = document.getElementById('tensor-dropzone');
    zone.classList.add('drag-over');
    const file = makeFile('a.npy');
    const ev = new Event('drop', { bubbles: true, cancelable: true });
    ev.dataTransfer = { files: [file] };
    zone.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
    expect(zone.classList.contains('drag-over')).toBe(false);
    await flushMicrotasks();
    expect(studio.wasm.call).toHaveBeenCalledWith('annealInspectTensor', expect.any(Uint8Array), 'npy');
    expect(document.getElementById('tensor-result').hidden).toBe(false);
  });

  it('clicking the zone opens the picker', () => {
    S.renderStudioView();
    const zone = document.getElementById('tensor-dropzone');
    const picker = document.getElementById('tensor-picker');
    const spy = vi.spyOn(picker, 'click');
    zone.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('clicking directly on the picker does not recurse', () => {
    S.renderStudioView();
    const picker = document.getElementById('tensor-picker');
    const spy = vi.spyOn(picker, 'click');
    // event.target === picker -> handler returns early
    picker.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(spy).not.toHaveBeenCalled();
  });

  it('Enter and Space on the zone open the picker', () => {
    S.renderStudioView();
    const zone = document.getElementById('tensor-dropzone');
    const picker = document.getElementById('tensor-picker');
    const spy = vi.spyOn(picker, 'click');
    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
    zone.dispatchEvent(enter);
    const space = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
    zone.dispatchEvent(space);
    expect(spy).toHaveBeenCalledTimes(2);
    expect(enter.defaultPrevented).toBe(true);
    expect(space.defaultPrevented).toBe(true);
  });

  it('other keys on the zone do not open the picker', () => {
    S.renderStudioView();
    const zone = document.getElementById('tensor-dropzone');
    const picker = document.getElementById('tensor-picker');
    const spy = vi.spyOn(picker, 'click');
    zone.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true }));
    expect(spy).not.toHaveBeenCalled();
  });

  it('picker change event inspects the selected file', async () => {
    S.renderStudioView();
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ format: 'npy', tensors: [] }));
    const picker = document.getElementById('tensor-picker');
    const file = makeFile('sel.npy');
    Object.defineProperty(picker, 'files', { value: [file], configurable: true });
    picker.dispatchEvent(new Event('change', { bubbles: true }));
    await flushMicrotasks();
    expect(studio.wasm.call).toHaveBeenCalledWith('annealInspectTensor', expect.any(Uint8Array), 'npy');
  });

  it('drop with no files is a no-op', () => {
    S.renderStudioView();
    studio.wasm.call = vi.fn();
    const zone = document.getElementById('tensor-dropzone');
    const ev = new Event('drop', { bubbles: true, cancelable: true });
    ev.dataTransfer = { files: [] };
    zone.dispatchEvent(ev);
    expect(studio.wasm.call).not.toHaveBeenCalled();
  });
});

describe('inspectFile', () => {
  it('renders tensor rows on a happy path', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({
      format: 'safetensors',
      tensors: [
        { name: 'w', dtype: 'f32', shape: [2, 3], numel: 6, preview: [0.5, 1.0] },
        { name: 'b', dtype: 'f32', shape: [3], numel: 3, preview: [] },
      ],
    }));
    const file = makeFile('m.safetensors', new Uint8Array([9, 8, 7]));
    await S.inspectFile(file);
    expect(studio.wasm.call).toHaveBeenCalledWith('annealInspectTensor', expect.any(Uint8Array), 'safetensors');
    expect(document.getElementById('tensor-file-name').textContent).toBe('m.safetensors');
    const meta = document.getElementById('tensor-result-meta').textContent;
    expect(meta).toContain('format: safetensors');
    expect(meta).toContain('tensors: 2');
    const rows = document.querySelectorAll('#tensor-rows tr');
    expect(rows.length).toBe(2);
    const cells = rows[0].querySelectorAll('td');
    expect(cells[0].textContent).toBe('w');
    expect(cells[1].textContent).toBe('f32');
    expect(cells[2].textContent).toBe('[2,3]');
    expect(cells[3].textContent).toBe('6');
    expect(cells[4].textContent).toBe('[0.5000, 1.0000]');
    expect(cells[4].className).toBe('tensor-preview');
  });

  it('reports unknown extension and does not call wasm', async () => {
    studio.wasm.call = vi.fn();
    await S.inspectFile(makeFile('data.bin'));
    expect(studio.wasm.call).not.toHaveBeenCalled();
    expect(document.getElementById('tensor-result-meta').textContent)
      .toContain('unknown extension');
    expect(document.getElementById('tensor-result').hidden).toBe(false);
  });

  it('reports a read error when arrayBuffer rejects', async () => {
    studio.wasm.call = vi.fn();
    const file = makeFile('a.npy', new Uint8Array([1]), { failRead: true });
    await S.inspectFile(file);
    expect(studio.wasm.call).not.toHaveBeenCalled();
    expect(document.getElementById('tensor-result-meta').textContent)
      .toContain('read error: boom-read');
  });

  it('reports wasm-not-loaded when wasm.call rejects', async () => {
    studio.wasm.call = vi.fn().mockRejectedValue(new Error('worker not loaded'));
    await S.inspectFile(makeFile('a.npy'));
    expect(document.getElementById('tensor-result-meta').textContent)
      .toContain('wasm not loaded');
  });

  it('reports invalid JSON from the bridge', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue('not-json{');
    await S.inspectFile(makeFile('a.npy'));
    expect(document.getElementById('tensor-result-meta').textContent)
      .toContain('invalid JSON');
  });

  it('reports a parse error payload', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ error: 'bad magic' }));
    await S.inspectFile(makeFile('a.npy'));
    expect(document.getElementById('tensor-result-meta').textContent)
      .toContain('parse error: bad magic');
  });
});
