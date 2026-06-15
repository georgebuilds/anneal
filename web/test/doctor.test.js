// W9 doctor view tests.
//
// Covers renderDoctorView (drives both cards), fillNativeCard (GET /api/device
// success, error-field, and fetch rejection), and fillBrowserCard
// (no navigator.gpu, no adapter granted, full adapter.info path including
// features + limits, requestAdapterInfo fallback, and a thrown error). fetch
// and navigator.gpu are stubbed per-test.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

let studio;
let S;

// dlPairs returns the [dt,dd] text pairs of a definition list as a plain object.
function dlPairs(id) {
  const dl = document.getElementById(id);
  const out = {};
  const dts = dl.querySelectorAll('dt');
  const dds = dl.querySelectorAll('dd');
  for (let i = 0; i < dts.length; i++) {
    out[dts[i].textContent] = dds[i] ? dds[i].textContent : undefined;
  }
  return out;
}

beforeEach(async () => {
  studio = await loadStudio({ path: '/' });
  S = studio.__studio;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fillNativeCard', () => {
  it('renders the native adapter rows from /api/device', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({
        adapter_name: 'Apple M3',
        backend: 'Metal',
        os: 'darwin',
        arch: 'arm64',
        anneal_version: 'v0.9',
        shader_f16: true,
        max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
      }),
    }));
    await S.fillNativeCard();
    const p = dlPairs('native-info');
    expect(p.adapter).toBe('Apple M3');
    expect(p.backend).toBe('Metal');
    expect(p['os / arch']).toBe('darwin / arm64');
    expect(p.anneal).toBe('v0.9');
    expect(p['shader-f16']).toBe('yes');
    expect(p['max storage']).toBe('2.00 GiB');
    expect(p.error).toBeUndefined();
  });

  it('falls back to ? for missing fields and shows shader-f16 no', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({}),
    }));
    await S.fillNativeCard();
    const p = dlPairs('native-info');
    expect(p.adapter).toBe('?');
    expect(p['os / arch']).toBe('? / ?');
    expect(p['shader-f16']).toBe('no');
    expect(p['max storage']).toBe('?');
  });

  it('appends an error row when the payload carries one', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ adapter_name: 'X', error: 'no device' }),
    }));
    await S.fillNativeCard();
    expect(dlPairs('native-info').error).toBe('no device');
  });

  it('renders an error row when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    await S.fillNativeCard();
    expect(dlPairs('native-info').error).toBe('offline');
  });
});

describe('fillBrowserCard', () => {
  it('reports webgpu not available when navigator.gpu is absent', async () => {
    // jsdom navigator has no gpu by default, but guard explicitly.
    if ('gpu' in navigator) {
      // eslint-disable-next-line no-undef
      delete navigator.gpu;
    }
    await S.fillBrowserCard();
    const p = dlPairs('browser-info');
    expect(p.webgpu).toBe('not available in this browser');
  });

  it('reports no adapter granted', async () => {
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: vi.fn().mockResolvedValue(null) },
    });
    await S.fillBrowserCard();
    expect(dlPairs('browser-info').adapter)
      .toBe('navigator.gpu present but no adapter granted');
  });

  it('renders adapter.info, features and limits', async () => {
    const adapter = {
      info: { vendor: 'apple', architecture: 'metal-3', device: 'M3', description: 'Apple GPU' },
      features: new Set(['shader-f16', 'timestamp-query']),
      limits: { maxStorageBufferBindingSize: 1024 * 1024 },
    };
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: vi.fn().mockResolvedValue(adapter) },
    });
    await S.fillBrowserCard();
    const p = dlPairs('browser-info');
    expect(p.vendor).toBe('apple');
    expect(p.architecture).toBe('metal-3');
    expect(p.device).toBe('M3');
    expect(p.description).toBe('Apple GPU');
    expect(p['shader-f16']).toBe('yes');
    expect(p['max storage']).toBe('1.0 MiB');
  });

  it('falls back to requestAdapterInfo() when adapter.info is absent', async () => {
    const adapter = {
      requestAdapterInfo: vi.fn().mockResolvedValue({ vendor: 'nv', architecture: 'ada' }),
      features: new Set(),
    };
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: vi.fn().mockResolvedValue(adapter) },
    });
    await S.fillBrowserCard();
    const p = dlPairs('browser-info');
    expect(adapter.requestAdapterInfo).toHaveBeenCalled();
    expect(p.vendor).toBe('nv');
    expect(p.architecture).toBe('ada');
    expect(p['shader-f16']).toBe('no');
    // no limits -> no max storage row
    expect(p['max storage']).toBeUndefined();
  });

  it('renders an error row when requestAdapter throws', async () => {
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: vi.fn().mockRejectedValue(new Error('gpu boom')) },
    });
    await S.fillBrowserCard();
    expect(dlPairs('browser-info').error).toBe('gpu boom');
  });
});

describe('renderDoctorView', () => {
  it('drives both cards', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ adapter_name: 'A', backend: 'Vulkan' }),
    }));
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: vi.fn().mockResolvedValue(null) },
    });
    await S.renderDoctorView();
    await flushMicrotasks();
    expect(dlPairs('native-info').adapter).toBe('A');
    expect(dlPairs('browser-info').adapter)
      .toBe('navigator.gpu present but no adapter granted');
  });
});
