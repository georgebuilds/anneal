// Harness smoke test: proves studio.js imports cleanly against the studio.html
// DOM fixture, boot() runs without throwing, and the __studio test seam is
// reachable. Section-specific behavior lives in the other *.test.js files.
import { describe, it, expect, beforeEach } from 'vitest';
import { loadStudio } from './harness.js';

describe('studio harness', () => {
  let studio;
  beforeEach(async () => {
    studio = await loadStudio();
  });

  it('exposes the __studio test seam', () => {
    expect(studio.__studio).toBeTypeOf('object');
    expect(studio.__studio.navigate).toBeTypeOf('function');
  });

  it('exports the wasm RPC client that rejects without a worker meta tag', async () => {
    expect(studio.wasm).toBeTypeOf('object');
    expect(studio.wasm.ready).toBe(false);
    await expect(studio.wasm.call('annealGetGraph')).rejects.toThrow(/worker not loaded/);
  });

  it('viewIdForPath maps known routes', () => {
    expect(studio.__studio.viewIdForPath('/')).toBe('studio');
  });
});
