// Global jsdom polyfills for the studio test suite.
//
// jsdom does not implement matchMedia (the theme controller subscribes to it)
// and stubs Worker only partially. These shims make import-time studio.js code
// (boot(), makeWorkerRPC) run without throwing. Per-test behavior overrides go
// in individual test files via vi.spyOn / vi.stubGlobal.
import { vi } from 'vitest';

if (!window.matchMedia) {
  window.matchMedia = (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {}, // deprecated, some code paths still call it
    removeListener: () => {},
    dispatchEvent: () => false,
  });
}

// requestAnimationFrame is used by the ignite animation and a few renderers.
if (!window.requestAnimationFrame) {
  window.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
  window.cancelAnimationFrame = (id) => clearTimeout(id);
}

// jsdom does not give canvas a 2d context by default; drawMiniGraph touches it.
if (!HTMLCanvasElement.prototype.getContext) {
  HTMLCanvasElement.prototype.getContext = () => null;
}

// Keep console.error from failing tests that intentionally exercise error paths
// while still surfacing unexpected noise during local runs.
void vi;
