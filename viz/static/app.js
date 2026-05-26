// anneal viz — static UOp graph renderer
// Loads the real graph from WASM (anneal.wasm) or the native REST API,
// lays it out with a layered DAG algorithm, and renders to SVG.

'use strict';

// ── Theme management ──────────────────────────────────────────────────────

const THEMES = ['system', 'dark', 'light'];
let themeIdx = 0;

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const btn = document.getElementById('themeToggle');
  if (btn) btn.title = 'theme: ' + theme;
}

function cycleTheme() {
  themeIdx = (themeIdx + 1) % THEMES.length;
  const t = THEMES[themeIdx];
  applyTheme(t);
  try { localStorage.setItem('anneal-theme', t); } catch (_) {}
}

(function initTheme() {
  try {
    const saved = localStorage.getItem('anneal-theme');
    if (saved && THEMES.includes(saved)) {
      themeIdx = THEMES.indexOf(saved);
      applyTheme(saved);
      return;
    }
  } catch (_) {}
  applyTheme('system');
})();

// ── Status helpers ────────────────────────────────────────────────────────

function setStatus(msg, isError) {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = msg;
  el.className = 'status' + (isError ? ' error' : '');
}

function setStats(data) {
  const el = document.getElementById('stats');
  if (!el) return;
  const s = data.stats;
  el.textContent =
    `${s.allNodes} nodes  ·  ${s.fwdNodes} forward  ·  ${s.bwdNodes} backward  ·  ${s.kernels} kernels`;
}

// ── Data loading ──────────────────────────────────────────────────────────

// Try WASM first; fall back to REST /api/graph?name=...
// Returns a Promise<GraphData>.
async function loadGraph(name) {
  if (window._wasmReady && typeof window.annealGetGraph === 'function') {
    const json = window.annealGetGraph(name);
    const data = JSON.parse(json);
    if (data.error) throw new Error('WASM: ' + data.error);
    return data;
  }
  // REST fallback (native server)
  const resp = await fetch('/api/graph?name=' + encodeURIComponent(name));
  if (!resp.ok) throw new Error('API ' + resp.status);
  const data = await resp.json();
  if (data.error) throw new Error('API: ' + data.error);
  return data;
}

// ── WASM initialisation ───────────────────────────────────────────────────

async function initWASM() {
  if (window._wasmExecMissing) {
    // wasm_exec.js not found — will use REST fallback
    return false;
  }
  try {
    const go = new Go(); // eslint-disable-line no-undef
    const result = await WebAssembly.instantiateStreaming(
      fetch('anneal.wasm'),
      go.importObject
    );
    go.run(result.instance);
    window._wasmReady = true;
    return true;
  } catch (e) {
    console.warn('WASM not available, using REST API:', e.message);
    return false;
  }
}

// ── Layout algorithm ──────────────────────────────────────────────────────
// Topological level assignment (longest path from sources) + barycenter
// X positioning with 4 alternating sweeps to reduce edge crossings.

const NODE_DX = 96;   // horizontal spacing between nodes in a level
const NODE_DY = 88;   // vertical spacing between levels
const MARGIN  = 48;   // SVG border margin

function computeLayout(nodes, edges) {
  // Build adjacency maps.
  const parentOf = new Map(nodes.map(n => [n.id, []]));
  const childOf  = new Map(nodes.map(n => [n.id, []]));
  edges.forEach(e => {
    if (parentOf.has(e.to))   parentOf.get(e.to).push(e.from);
    if (childOf.has(e.from))  childOf.get(e.from).push(e.to);
  });

  // Assign levels via longest-path from leaf sources.
  // nodes are in topological order (sources first) from Go's post-order DFS.
  const level = new Map();
  for (const n of nodes) {
    const ps = parentOf.get(n.id) || [];
    const maxPL = ps.reduce((m, p) => Math.max(m, level.get(p) ?? 0), -1);
    level.set(n.id, maxPL + 1);
  }

  // Group nodes by level.
  const byLevel = new Map();
  nodes.forEach(n => {
    const l = level.get(n.id);
    if (!byLevel.has(l)) byLevel.set(l, []);
    byLevel.get(l).push(n.id);
  });

  // Initial X order: sequential within level.
  const xOrder = new Map();
  byLevel.forEach(ids => ids.forEach((id, i) => xOrder.set(id, i)));

  // Barycenter sweeps: alternate top→bottom and bottom→top passes.
  const maxLevel = Math.max(...level.values());
  for (let pass = 0; pass < 4; pass++) {
    const lvls = Array.from({length: maxLevel + 1}, (_, i) => i);
    if (pass % 2 === 1) lvls.reverse();

    for (const l of lvls) {
      const ids = byLevel.get(l);
      if (!ids || ids.length < 2) continue;

      const bary = new Map();
      for (const id of ids) {
        const neighbors = pass % 2 === 0
          ? (parentOf.get(id) || []).filter(p => level.get(p) < l)
          : (childOf.get(id)  || []).filter(c => level.get(c) > l);
        bary.set(id,
          neighbors.length
            ? neighbors.reduce((s, nb) => s + (xOrder.get(nb) ?? 0), 0) / neighbors.length
            : (xOrder.get(id) ?? 0)
        );
      }
      // Re-rank within level by barycenter value, preserving tie order.
      const sorted = [...ids].sort((a, b) => bary.get(a) - bary.get(b));
      sorted.forEach((id, i) => xOrder.set(id, i));
    }
  }

  // Compute final pixel positions.
  const pos = new Map();
  nodes.forEach(n => {
    const l  = level.get(n.id) ?? 0;
    const x  = xOrder.get(n.id) ?? 0;
    const lw = byLevel.get(l).length;
    pos.set(n.id, {
      x: MARGIN + x * NODE_DX,
      y: MARGIN + l * NODE_DY,
      levelW: lw
    });
  });

  return { pos, byLevel, maxLevel };
}

// ── SVG helpers ───────────────────────────────────────────────────────────

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// ── Graph rendering ───────────────────────────────────────────────────────

function renderGraph(data, svgEl_) {
  const { nodes, edges } = data;
  if (!nodes || nodes.length === 0) {
    setStatus('graph is empty', true);
    return;
  }

  const { pos, byLevel, maxLevel } = computeLayout(nodes, edges);

  // Compute SVG dimensions.
  let maxX = 0, maxY = 0;
  pos.forEach(p => { maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); });
  const W = maxX + MARGIN + NODE_DX;
  const H = maxY + MARGIN + NODE_DY;

  svgEl_.setAttribute('width',   W);
  svgEl_.setAttribute('height',  H);
  svgEl_.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgEl_.innerHTML = '';

  // Build lookup maps.
  const nodeClass = new Map(nodes.map(n => [n.id, n.class]));
  const nodeKind  = new Map(nodes.map(n => [n.id, n.kind]));

  // ── 1. Draw edges (below nodes) ──────────────────────────────────────
  const edgeG = svgEl('g', {'class': 'edges'});

  for (const e of edges) {
    const from = pos.get(e.from);
    const to   = pos.get(e.to);
    if (!from || !to) continue;

    const srcClass = nodeClass.get(e.from) || 'forward';
    const dstKind  = nodeKind.get(e.to)   || 'default';

    let stroke    = 'var(--edge-fwd)';
    let dasharray = '';
    if (srcClass === 'backward') {
      stroke    = 'var(--edge-bwd)';
      dasharray = '5,3';
    }
    if (dstKind === 'reduce') stroke = 'var(--edge-red)';

    // Smooth cubic bezier: vertical control arms at source and destination.
    const midY = (from.y + to.y) / 2;
    const attrs = {
      d:              `M ${from.x},${from.y} C ${from.x},${midY} ${to.x},${midY} ${to.x},${to.y}`,
      stroke,
      'stroke-width': '1.5',
      fill:           'none',
    };
    if (dasharray) attrs['stroke-dasharray'] = dasharray;
    edgeG.appendChild(svgEl('path', attrs));
  }
  svgEl_.appendChild(edgeG);

  // ── 2. Draw nodes ─────────────────────────────────────────────────────
  const nodeG = svgEl('g', {'class': 'nodes'});

  for (const n of nodes) {
    const p = pos.get(n.id);
    if (!p) continue;

    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', `node ${n.class} ${n.kind}`);
    g.setAttribute('transform', `translate(${p.x},${p.y})`);
    // Accessibility: each node is an img with a descriptive label.
    g.setAttribute('role', 'img');
    g.setAttribute('aria-label', `${n.class} ${n.op} node: ${n.label}`);

    // ── Determine fill / stroke from CSS variables ────────────────────
    // This preserves the DD1 semantic across light and dark themes.
    let fillVar, strokeVar, textVar;
    let strokeW  = '1.5';
    let dasharray = '';

    if (n.kind === 'sink') {
      // Neutral aggregation point — surface color, not forward/backward.
      fillVar   = 'var(--surface)';
      strokeVar = 'var(--muted)';
      textVar   = 'var(--muted)';
    } else if (n.kind === 'reduce') {
      // Gold: kernel boundary / ReduceAxis — always gold regardless of provenance.
      // (v1 honest: every reduce is a hard boundary, removeBufferize is a no-op.)
      fillVar   = 'var(--red-fill)';
      strokeVar = 'var(--red-stroke)';
      textVar   = 'var(--red-text)';
      strokeW   = '2.5';
    } else if (n.class === 'backward') {
      // Ember: backward pass. Dashed border = second encoding channel (§9).
      fillVar   = 'var(--bwd-fill)';
      strokeVar = 'var(--bwd-stroke)';
      textVar   = 'var(--bwd-text)';
      dasharray = '5,3';
    } else if (n.kind === 'leaf') {
      // Teal leaf: parameter / input buffer. Rounded rect shape = second channel.
      fillVar   = 'var(--leaf-fill)';
      strokeVar = 'var(--leaf-stroke)';
      textVar   = 'var(--fwd-text)';
    } else {
      // Teal: forward pass. Solid circle = second encoding channel (§9).
      fillVar   = 'var(--fwd-fill)';
      strokeVar = 'var(--fwd-stroke)';
      textVar   = 'var(--fwd-text)';
    }

    // ── Shape by kind (second encoding channel, per §9) ───────────────
    let shape;
    switch (n.kind) {
      case 'reduce': {
        // Diamond: visually distinct from circles; signals "this is a boundary".
        shape = svgEl('polygon', { points: '0,-22 22,0 0,22 -22,0' });
        break;
      }
      case 'leaf': {
        // Rounded rect: parameter/input buffer — wider to fit the shape label.
        shape = svgEl('rect', { x: '-36', y: '-16', width: '72', height: '32', rx: '8', ry: '8' });
        break;
      }
      case 'sink': {
        // Hexagon: aggregation point.
        const r = 20;
        const pts = Array.from({length: 6}, (_, i) => {
          const a = (i * 60 - 30) * Math.PI / 180;
          return `${(r * Math.cos(a)).toFixed(1)},${(r * Math.sin(a)).toFixed(1)}`;
        }).join(' ');
        shape = svgEl('polygon', { points: pts });
        break;
      }
      default: {
        // Circle: standard operation node.
        shape = svgEl('circle', { r: '18' });
      }
    }
    shape.setAttribute('fill',         fillVar);
    shape.setAttribute('stroke',       strokeVar);
    shape.setAttribute('stroke-width', strokeW);
    if (dasharray) shape.setAttribute('stroke-dasharray', dasharray);
    g.appendChild(shape);

    // ── Op mnemonic inside shape ──────────────────────────────────────
    const opShort = n.op.length > 9 ? n.op.slice(0, 8) + '…' : n.op;
    const opTxt = svgEl('text', {
      y: '4', 'text-anchor': 'middle',
      'font-size': '9', 'font-family': 'monospace', 'font-weight': 'bold',
      fill: textVar, 'pointer-events': 'none',
    });
    opTxt.textContent = opShort;
    g.appendChild(opTxt);

    // ── Human label below the node ────────────────────────────────────
    const labelShort = n.label.length > 16 ? n.label.slice(0, 15) + '…' : n.label;
    const lbl = svgEl('text', {
      y: '34', 'text-anchor': 'middle',
      'font-size': '9', 'font-family': 'monospace',
      fill: 'var(--muted)', 'pointer-events': 'none',
    });
    lbl.textContent = labelShort;
    g.appendChild(lbl);

    // ── Tooltip via <title> ───────────────────────────────────────────
    const title = document.createElementNS(NS, 'title');
    title.textContent = [
      n.op,
      n.dtype !== 'void' ? n.dtype : '',
      n.shape && n.shape.length ? 'shape=' + JSON.stringify(n.shape) : '',
      n.arg ? 'arg=' + n.arg : '',
      n.class + ' / ' + n.kind,
      'id=' + n.id,
    ].filter(Boolean).join('  ');
    g.insertBefore(title, g.firstChild);

    nodeG.appendChild(g);
  }
  svgEl_.appendChild(nodeG);
}

// ── Main entry point ──────────────────────────────────────────────────────

let currentName = 'mlp';

async function loadAndRender(name) {
  currentName = name;
  setStatus('loading graph…');
  const svg = document.getElementById('graph-svg');

  try {
    const data = await loadGraph(name);
    renderGraph(data, svg);
    setStats(data);
    const src = window._wasmReady ? 'WASM' : 'REST API';
    setStatus(`${name} — rendered via ${src} (real compiler output)`);
  } catch (e) {
    setStatus('error: ' + e.message, true);
    console.error(e);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  // Wire theme toggle.
  document.getElementById('themeToggle')?.addEventListener('click', cycleTheme);

  // Wire model selector.
  document.querySelectorAll('input[name="model"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (radio.checked) loadAndRender(radio.value);
    });
  });

  // Initialise WASM (best-effort; falls back to REST).
  setStatus('initialising compiler…');
  await initWASM();

  // Render initial graph.
  await loadAndRender(currentName);
});
