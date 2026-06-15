// Tests for the explain view (W3) in web/studio.js.
//
// Surface under test (via the __studio seam):
//   renderExplainView, selectOp, opFromExplainPath, drawMiniGraph
//
// renderExplainView() reads /x/<op>, draws the op list, then loadExplain()
// calls `await wasm.call('annealExplainOp', opName)` and JSON.parses the
// result string into the detail panel (name/desc/rules/gradient/mini-graph).
//
// wasm is an exported object; studio.js reads `wasm.call` by property at call
// time, so we overwrite studio.wasm.call AFTER loadStudio (per harness note).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStudio, flushMicrotasks } from './harness.js';

let studio;

// annealExplainOp returns a JSON *string*; drawExplainDetail consumes:
//   { op, description, symbolic_rules:[{pattern,rewrite,notes,source,name}],
//     gradient_rule:{pattern,source},
//     mini_graph:{ before:[{id,op,label}], after:[...], edges:[{from,to}] } }
function explainPayload(op) {
  return {
    op,
    description: 'the ' + op + ' op, explained',
    symbolic_rules: [
      {
        name: op + '-identity',
        pattern: op + '(x, 0)',
        rewrite: 'x',
        notes: 'additive identity',
        source: 'rewrite/symbolic.go:101',
      },
    ],
    gradient_rule: {
      pattern: 'd/dx ' + op,
      source: 'tensor/gradient.go:55',
    },
    mini_graph: {
      // before: two leaves feed one ALU op (in-degree drives layout).
      before: [
        { id: 'a', op: 'Const', label: '2' },
        { id: 'b', op: 'DefineVar', label: 'x' },
        { id: 'c', op: op, label: op },
      ],
      after: [{ id: 'c', op: op, label: op }],
      edges: [
        { from: 'a', to: 'c' },
        { from: 'b', to: 'c' },
      ],
    },
  };
}

describe('opFromExplainPath', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/x/Add' });
  });

  it('parses the op name from /x/<op>', () => {
    expect(studio.__studio.opFromExplainPath('/x/Mul')).toBe('Mul');
  });

  it('decodes URI-encoded op names', () => {
    expect(studio.__studio.opFromExplainPath('/x/Cmp%4ct')).toBe('CmpLt');
  });

  it('ignores query + hash', () => {
    expect(studio.__studio.opFromExplainPath('/x/Where?k=1#z')).toBe('Where');
  });

  it('defaults to Add when the path does not match', () => {
    expect(studio.__studio.opFromExplainPath('/v/mlp')).toBe('Add');
  });
});

describe('renderExplainView — success path', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/x/Mul' });
    studio.wasm.call = vi
      .fn()
      .mockImplementation((fn, op) => Promise.resolve(JSON.stringify(explainPayload(op))));
  });

  it('renders the op list and selects the URL op', async () => {
    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    // Op list is populated from the FALLBACK_OPS master list.
    const items = document.querySelectorAll('#op-list-items .op-list-item');
    expect(items.length).toBeGreaterThan(0);

    // The URL op is selected (aria-selected="true").
    const selected = document.querySelector('#op-list-items [aria-selected="true"]');
    expect(selected).not.toBeNull();
    expect(selected.textContent).toBe('Mul');
    expect(document.getElementById('op-list-items').getAttribute('aria-activedescendant'))
      .toBe('op-opt-Mul');
  });

  it('renders detail: name, description, rules, gradient, mini-graph', async () => {
    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    expect(studio.wasm.call).toHaveBeenCalledWith('annealExplainOp', 'Mul');
    expect(document.getElementById('exp-op-name').textContent).toBe('Mul');
    expect(document.getElementById('exp-desc').textContent).toBe('the Mul op, explained');

    const rules = document.querySelectorAll('#exp-rules .rules-list-item');
    expect(rules.length).toBe(1);
    expect(rules[0].querySelector('.rule-pattern').textContent).toContain('Mul(x, 0)');
    expect(rules[0].querySelector('.rule-pattern').textContent).toContain('x');
    expect(rules[0].querySelector('.rule-note').textContent).toBe('additive identity');
    expect(rules[0].querySelector('.rule-source').textContent).toBe('rewrite/symbolic.go:101');

    expect(document.getElementById('exp-grad').textContent).toContain('d/dx Mul');
    expect(document.getElementById('exp-grad').textContent).toContain('tensor/gradient.go:55');

    // Mini-graph SVG was populated: 3 nodes (g.mini-node) + 2 edges (line).
    const svg = document.querySelector('#exp-mini svg');
    expect(svg.querySelectorAll('g.mini-node').length).toBe(3);
    expect(svg.querySelectorAll('line.mini-edge').length).toBe(2);
    // The ALU op node (in-degree > 0) gets the alu shape class.
    expect(svg.querySelector('.mini-node-alu')).not.toBeNull();
    // Const leaf gets the const class; DefineVar gets the leaf class.
    expect(svg.querySelector('.mini-node-const')).not.toBeNull();
    expect(svg.querySelector('.mini-node-leaf')).not.toBeNull();
  });

  it('renders the empty-rules placeholder when no symbolic rules exist', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(
      JSON.stringify({ op: 'Buffer', description: 'a buffer', symbolic_rules: [] })
    );
    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    const empty = document.querySelector('#exp-rules .rules-list-empty');
    expect(empty).not.toBeNull();
    expect(empty.textContent).toContain('no symbolic rewrite rules');
    // No gradient_rule → the non-differentiable caption.
    expect(document.getElementById('exp-grad').textContent).toContain('no gradient registered');
  });
});

describe('selectOp — switches the explained op', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/x/Add' });
    studio.wasm.call = vi
      .fn()
      .mockImplementation((fn, op) => Promise.resolve(JSON.stringify(explainPayload(op))));
  });

  it('updates selection, pushes the deep link, and reloads detail', async () => {
    await studio.__studio.renderExplainView();
    await flushMicrotasks();
    expect(document.getElementById('exp-op-name').textContent).toBe('Add');

    studio.__studio.selectOp('Exp2');
    await flushMicrotasks();

    // Deep link pushed.
    expect(window.location.pathname).toBe('/x/Exp2');
    // Detail reloaded for the new op.
    expect(studio.wasm.call).toHaveBeenLastCalledWith('annealExplainOp', 'Exp2');
    expect(document.getElementById('exp-op-name').textContent).toBe('Exp2');
    // List reflects the new selection.
    const selected = document.querySelector('#op-list-items [aria-selected="true"]');
    expect(selected.textContent).toBe('Exp2');
  });

  it('clicking an op-list item selects it', async () => {
    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    const target = document.getElementById('op-opt-Neg');
    expect(target).not.toBeNull();
    target.click();
    await flushMicrotasks();

    expect(window.location.pathname).toBe('/x/Neg');
    expect(document.getElementById('exp-op-name').textContent).toBe('Neg');
  });
});

describe('drawMiniGraph — direct invocation', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/x/Add' });
  });

  it('renders before-side nodes + edges into the SVG', () => {
    const mg = explainPayload('Add').mini_graph;
    studio.__studio.drawMiniGraph(mg, 'before');

    const svg = document.querySelector('#exp-mini svg');
    expect(svg.querySelectorAll('g.mini-node').length).toBe(3);
    expect(svg.querySelectorAll('line.mini-edge').length).toBe(2);
    // <title> for the side state.
    expect(svg.querySelector('title').textContent).toBe('rewrite before state');
  });

  it('renders the after side (single node, no edges)', () => {
    const mg = explainPayload('Add').mini_graph;
    studio.__studio.drawMiniGraph(mg, 'after');

    const svg = document.querySelector('#exp-mini svg');
    expect(svg.querySelectorAll('g.mini-node').length).toBe(1);
    expect(svg.querySelectorAll('line.mini-edge').length).toBe(0);
    expect(svg.querySelector('title').textContent).toBe('rewrite after state');
  });

  it('clears the SVG and tolerates a null mini-graph', () => {
    const mg = explainPayload('Add').mini_graph;
    studio.__studio.drawMiniGraph(mg, 'before');
    expect(document.querySelector('#exp-mini svg').querySelectorAll('g.mini-node').length).toBe(3);

    // Passing null clears + early-returns without throwing.
    expect(() => studio.__studio.drawMiniGraph(null, 'before')).not.toThrow();
    expect(document.querySelector('#exp-mini svg').querySelectorAll('g.mini-node').length).toBe(0);
  });
});

describe('renderExplainView — error branch', () => {
  beforeEach(async () => {
    studio = await loadStudio({ path: '/x/Add' });
  });

  it('renders the wasm-not-loaded empty state when wasm.call rejects', async () => {
    studio.wasm.call = vi.fn().mockRejectedValue(new Error('wasm worker not loaded'));

    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    // The op list still renders (browsable without the compiler).
    expect(document.querySelectorAll('#op-list-items .op-list-item').length).toBeGreaterThan(0);
    // Detail falls back to the wasm-not-loaded caption.
    expect(document.getElementById('exp-op-name').textContent).toBe('Add');
    expect(document.getElementById('exp-desc').textContent).toContain('wasm not loaded');
  });

  it('renders the error state when annealExplainOp returns invalid JSON', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue('{nope');

    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    expect(document.getElementById('exp-desc').textContent).toContain('wasm not loaded');
  });

  it('renders the error state when the payload carries an error field', async () => {
    studio.wasm.call = vi.fn().mockResolvedValue(JSON.stringify({ error: 'unknown op Zzz' }));

    await studio.__studio.renderExplainView();
    await flushMicrotasks();

    // error state routes through drawExplainDetail's error branch.
    expect(document.getElementById('exp-desc').textContent).toContain('wasm not loaded');
  });
});
