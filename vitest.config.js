import { defineConfig } from 'vitest/config';

// Frontend test config for the anneal studio (web/studio.js, web/worker.js).
// Tests live in web/test/*.test.js and run under jsdom. Coverage is restricted
// to the two hand-written browser sources; wasm_exec.js is generated Go runtime
// glue and is excluded (mirrors the codecov ignore for generated files).
export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['web/test/**/*.test.js'],
    setupFiles: ['web/test/setup.js'],
    coverage: {
      provider: 'v8',
      include: ['web/studio.js', 'web/worker.js'],
      reporter: ['text', 'lcov'],
      reportsDirectory: 'coverage-js',
    },
  },
});
