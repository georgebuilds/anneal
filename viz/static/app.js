// anneal viz - scrub-timeline UOp graph renderer.
//
// Loads the real timeline from WASM (anneal.wasm) or the native REST API, lays
// out the union of all stages once with a layered DAG algorithm, then scrubs
// through compiler-pipeline stages by toggling node visibility / class without
// touching positions. Position stability is the whole point: the eye tracks
// what *changes* between stages, not a reshuffle (DESIGN.md §7).
//
// Brand: forward = teal, backward = ember, kernel boundary = gold (DD1).
// Chrome recedes; the graph is the artifact.

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
  // URL ?theme=… wins (for screenshot harnesses), then localStorage, then system.
  try {
    const u = new URL(window.location.href);
    const t = u.searchParams.get('theme');
    if (t && THEMES.includes(t)) {
      themeIdx = THEMES.indexOf(t);
      applyTheme(t);
      return;
    }
  } catch (_) {}
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

// Stats formatter: stage-aware, suppresses zero counters so the line stays
// honest at the forward stage (kernels = 0 until scheduling runs).
function setStats(stageStats, source) {
  const el = document.getElementById('stats');
  if (!el) return;
  const parts = [];
  parts.push(`${stageStats.fwdNodes || 0} forward`);
  if (stageStats.bwdNodes) parts.push(`${stageStats.bwdNodes} backward`);
  if (stageStats.kernels)  parts.push(`${stageStats.kernels} kernels`);
  if (stageStats.fused)    parts.push(`${stageStats.fused} fused`);
  if (source) parts.push(`(real compiler via ${source})`);
  el.textContent = parts.join('  ·  ');
}

// ── Data loading ──────────────────────────────────────────────────────────

// Try WASM first; fall back to REST /api/timeline?name=...
async function loadTimeline(name) {
  if (window._wasmReady && typeof window.annealGetTimeline === 'function') {
    const json = window.annealGetTimeline(name);
    const data = JSON.parse(json);
    if (data.error) throw new Error('WASM: ' + data.error);
    return data;
  }
  const resp = await fetch('/api/timeline?name=' + encodeURIComponent(name));
  if (!resp.ok) throw new Error('API ' + resp.status);
  const data = await resp.json();
  if (data.error) throw new Error('API: ' + data.error);
  return data;
}

// ── WASM initialisation ───────────────────────────────────────────────────

async function initWASM() {
  if (window._wasmExecMissing) return false;
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
// Layered DAG layout on the *union* graph (every node from every stage). The
// layout is computed once per (model) load; stages share these coordinates.

const NODE_DX = 96;
const NODE_DY = 88;
const MARGIN  = 48;

function computeLayout(nodes, edges) {
  const parentOf = new Map(nodes.map(n => [n.id, []]));
  const childOf  = new Map(nodes.map(n => [n.id, []]));
  edges.forEach(e => {
    if (parentOf.has(e.to))   parentOf.get(e.to).push(e.from);
    if (childOf.has(e.from))  childOf.get(e.from).push(e.to);
  });

  const level = new Map();
  for (const n of nodes) {
    const ps = parentOf.get(n.id) || [];
    const maxPL = ps.reduce((m, p) => Math.max(m, level.get(p) ?? 0), -1);
    level.set(n.id, maxPL + 1);
  }

  const byLevel = new Map();
  nodes.forEach(n => {
    const l = level.get(n.id);
    if (!byLevel.has(l)) byLevel.set(l, []);
    byLevel.get(l).push(n.id);
  });

  const xOrder = new Map();
  byLevel.forEach(ids => ids.forEach((id, i) => xOrder.set(id, i)));

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
      const sorted = [...ids].sort((a, b) => bary.get(a) - bary.get(b));
      sorted.forEach((id, i) => xOrder.set(id, i));
    }
  }

  const pos = new Map();
  nodes.forEach(n => {
    const l  = level.get(n.id) ?? 0;
    const x  = xOrder.get(n.id) ?? 0;
    pos.set(n.id, {
      x: MARGIN + x * NODE_DX,
      y: MARGIN + l * NODE_DY,
    });
  });

  return { pos, maxLevel };
}

// ── SVG helpers ───────────────────────────────────────────────────────────

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// ── Per-stage rendering helpers ───────────────────────────────────────────

function classFillStroke(cls, kind) {
  // Mirrors the original BuildGraph rendering rules. Sink is neutral surface;
  // reduce (kernel boundary) is always gold; backward is ember; leaf/default
  // keep their teal forward styling.
  if (kind === 'sink') {
    return { fill: 'var(--surface)', stroke: 'var(--muted)', text: 'var(--muted)', sw: '1.5', dash: '' };
  }
  if (kind === 'reduce') {
    return { fill: 'var(--red-fill)', stroke: 'var(--red-stroke)', text: 'var(--red-text)', sw: '2.5', dash: '' };
  }
  if (cls === 'backward') {
    return { fill: 'var(--bwd-fill)', stroke: 'var(--bwd-stroke)', text: 'var(--bwd-text)', sw: '1.5', dash: '5,3' };
  }
  if (kind === 'leaf') {
    return { fill: 'var(--leaf-fill)', stroke: 'var(--leaf-stroke)', text: 'var(--fwd-text)', sw: '1.5', dash: '' };
  }
  return { fill: 'var(--fwd-fill)', stroke: 'var(--fwd-stroke)', text: 'var(--fwd-text)', sw: '1.5', dash: '' };
}

function makeShape(kind) {
  switch (kind) {
    case 'reduce':
      return svgEl('polygon', { points: '0,-22 22,0 0,22 -22,0' });
    case 'leaf':
      return svgEl('rect', { x: '-36', y: '-16', width: '72', height: '32', rx: '8', ry: '8' });
    case 'sink': {
      const r = 20;
      const pts = Array.from({length: 6}, (_, i) => {
        const a = (i * 60 - 30) * Math.PI / 180;
        return `${(r * Math.cos(a)).toFixed(1)},${(r * Math.sin(a)).toFixed(1)}`;
      }).join(' ');
      return svgEl('polygon', { points: pts });
    }
    default:
      return svgEl('circle', { r: '18' });
  }
}

// ── Graph rendering ───────────────────────────────────────────────────────
//
// Render the SVG skeleton once per loaded timeline (union nodes + edges in
// fixed positions). applyStage() then mutates per-node attributes to reflect
// the current stage's overrides - cheap and avoids re-laying out.

let timelineState = null; // { data, pos, nodeEls, edgeEls, currentStage }

function renderTimelineSkeleton(data, svgRoot) {
  const { nodes, edges } = data;
  if (!nodes || nodes.length === 0) {
    setStatus('graph is empty', true);
    return null;
  }

  const { pos } = computeLayout(nodes, edges);

  let maxX = 0, maxY = 0;
  pos.forEach(p => { maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); });
  const W = maxX + MARGIN + NODE_DX;
  const H = maxY + MARGIN + NODE_DY;

  svgRoot.setAttribute('width',   W);
  svgRoot.setAttribute('height',  H);
  svgRoot.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgRoot.innerHTML = '';

  // Edges first, below nodes.
  const edgeG = svgEl('g', {'class': 'edges'});
  const edgeEls = new Map(); // "from→to" -> {path}
  for (const e of edges) {
    const from = pos.get(e.from);
    const to   = pos.get(e.to);
    if (!from || !to) continue;
    const midY = (from.y + to.y) / 2;
    const path = svgEl('path', {
      class:           'edge',
      d:               `M ${from.x},${from.y} C ${from.x},${midY} ${to.x},${midY} ${to.x},${to.y}`,
      stroke:          'var(--edge-fwd)',
      'stroke-width':  '1.5',
      fill:            'none',
    });
    edgeG.appendChild(path);
    edgeEls.set(`${e.from}→${e.to}`, path);
  }
  svgRoot.appendChild(edgeG);

  // Nodes.
  const nodeG = svgEl('g', {'class': 'nodes'});
  const nodeEls = new Map();
  for (const n of nodes) {
    const p = pos.get(n.id);
    if (!p) continue;

    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', `node ${n.class} ${n.kind}`);
    g.setAttribute('transform', `translate(${p.x},${p.y})`);
    g.setAttribute('role', 'img');
    g.setAttribute('aria-label', `${n.class} ${n.op} node: ${n.label}`);
    // data-node-id is the arena index serialised as the viz-side string
    // identifier. The standalone `anneal viz` page does not read this; the
    // studio's W4 visualize embed (web/visualize_embed.html) uses it to
    // post {type:"nodeClick", nodeId} to the parent on click.
    g.setAttribute('data-node-id', `n${n.id}`);

    // Shape: container we'll mutate on stage change. We render the union's
    // shape here; applyStage swaps if a stage overrides kind (e.g. reduce
    // demoted to default at forward stage).
    const shape = makeShape(n.kind);
    const fs = classFillStroke(n.class, n.kind);
    shape.setAttribute('class', 'node-shape');
    shape.setAttribute('fill', fs.fill);
    shape.setAttribute('stroke', fs.stroke);
    shape.setAttribute('stroke-width', fs.sw);
    if (fs.dash) shape.setAttribute('stroke-dasharray', fs.dash);
    g.appendChild(shape);

    const opShort = n.op.length > 9 ? n.op.slice(0, 8) + '…' : n.op;
    const opTxt = svgEl('text', {
      class: 'node-op',
      y: '4', 'text-anchor': 'middle',
      'font-size': '9', 'font-family': 'monospace', 'font-weight': 'bold',
      fill: fs.text, 'pointer-events': 'none',
    });
    opTxt.textContent = opShort;
    g.appendChild(opTxt);

    const labelShort = n.label.length > 16 ? n.label.slice(0, 15) + '…' : n.label;
    const lbl = svgEl('text', {
      class: 'node-label',
      y: '34', 'text-anchor': 'middle',
      'font-size': '9', 'font-family': 'monospace',
      fill: 'var(--muted)', 'pointer-events': 'none',
    });
    lbl.textContent = labelShort;
    g.appendChild(lbl);

    const ruleLine = n.gradRule
      ? 'rule=' + n.gradRule + '·backward @ seq=' + (n.gradFiredSeq || 0)
      : '';
    const title = document.createElementNS(NS, 'title');
    title.textContent = [
      n.op,
      n.dtype !== 'void' ? n.dtype : '',
      n.shape && n.shape.length ? 'shape=' + JSON.stringify(n.shape) : '',
      n.arg ? 'arg=' + n.arg : '',
      n.class + ' / ' + n.kind,
      ruleLine,
      'id=' + n.id,
    ].filter(Boolean).join('  ');
    g.insertBefore(title, g.firstChild);

    nodeG.appendChild(g);
    nodeEls.set(n.id, {
      g, shape, opTxt,
      baseKind: n.kind, baseClass: n.class,
      gradRule: n.gradRule || '',
      gradFiredSeq: n.gradRule ? (n.gradFiredSeq || 0) : -1,
    });
  }
  svgRoot.appendChild(nodeG);

  return { pos, nodeEls, edgeEls };
}

// computeRulesIndex scans the union nodes and returns the max gradFiredSeq
// across all backward nodes (those with a non-empty gradRule), plus a
// seq -> ordered list of {id, rule} groups for label display. -1 means no
// rules fired (e.g. forward-only graph), in which case the rules sub-
// timeline stays hidden.
function computeRulesIndex(nodes) {
  let maxSeq = -1;
  const bySeq = new Map();
  for (const n of nodes) {
    if (!n.gradRule) continue;
    const seq = n.gradFiredSeq || 0;
    if (seq > maxSeq) maxSeq = seq;
    if (!bySeq.has(seq)) bySeq.set(seq, []);
    bySeq.get(seq).push({ id: n.id, rule: n.gradRule });
  }
  return { maxSeq, bySeq };
}

// applyRulesSeq sets the current rules cursor and re-runs the active
// stage so the seq filter composes with stage Overrides (node and edge
// visibility, restyling). Only meaningful on stages that include backward
// nodes; other stages ignore the cursor.
function applyRulesSeq(seq) {
  if (!timelineState || !timelineState.rules) return;
  timelineState.rules.cursor = seq;
  const slider = document.getElementById('rulesSlider');
  if (slider) slider.value = String(seq);
  const cur = document.getElementById('rulesCurrent');
  const cnt = document.getElementById('rulesCounter');
  const group = timelineState.rules.bySeq.get(seq) || [];
  const labels = group.map(g => '∂(' + g.rule + ')').join(', ');
  if (cur) cur.textContent = labels || '-';
  if (cnt) cnt.textContent = `seq ${seq + 1} / ${timelineState.rules.maxSeq + 1}`;
  applyStage(timelineState.currentStage);
}

// applyStage mutates the rendered SVG to reflect stage.Overrides without
// touching positions. Nodes absent from overrides become hidden; nodes
// present get reclassified per the stage's hints.
function applyStage(stageIdx) {
  if (!timelineState) return;
  const { data, nodeEls, edgeEls } = timelineState;
  const stage = data.stages[stageIdx];
  if (!stage) return;
  timelineState.currentStage = stageIdx;

  const overrides = stage.overrides || {};
  // Track which nodes are visible so edges can hide both endpoints are absent.
  const visible = new Set();

  // Rules-cursor: on stages that include backward nodes, hide any backward
  // node whose gradFiredSeq is past the current cursor. cursor == maxSeq
  // (the default) reveals every rule, restoring the original stage view.
  // Stages that do not include backward nodes (forward stage) skip this
  // filter implicitly because backward nodes are not in overrides.
  const rules = timelineState.rules;
  const stageHasBackward = stage.id === 'gradient' || stage.id === 'scheduled';
  const cursor = rules && stageHasBackward ? rules.cursor : Infinity;

  // First pass: visibility + per-node restyling.
  for (const [id, refs] of nodeEls.entries()) {
    const ov = overrides[id];
    if (!ov) {
      refs.g.classList.add('hidden');
      refs.shape.classList.remove('rule-just-fired');
      continue;
    }
    // Rules-cursor filter: backward nodes past the cursor stay hidden.
    if (refs.gradFiredSeq >= 0 && refs.gradFiredSeq > cursor) {
      refs.g.classList.add('hidden');
      refs.shape.classList.remove('rule-just-fired');
      continue;
    }
    refs.g.classList.remove('hidden');
    visible.add(id);
    // Highlight: nodes fired exactly at the cursor pulse briefly.
    if (refs.gradFiredSeq === cursor && stageHasBackward) {
      refs.shape.classList.add('rule-just-fired');
    } else {
      refs.shape.classList.remove('rule-just-fired');
    }

    // Determine effective kind / class for this stage.
    const cls  = ov.class || refs.baseClass;
    const kind = ov.kind  || refs.baseKind;

    // If kind changed, replace the shape element (preserves DOM order so the
    // op text overlay stays on top).
    if (kind !== refs.shape.dataset.kind) {
      const newShape = makeShape(kind);
      newShape.setAttribute('class', 'node-shape');
      newShape.dataset.kind = kind;
      refs.g.replaceChild(newShape, refs.shape);
      refs.shape = newShape;
    } else if (!refs.shape.dataset.kind) {
      refs.shape.dataset.kind = kind;
    }

    const fs = classFillStroke(cls, kind);
    refs.shape.setAttribute('fill',   fs.fill);
    refs.shape.setAttribute('stroke', fs.stroke);
    refs.shape.setAttribute('stroke-width', fs.sw);
    if (fs.dash) refs.shape.setAttribute('stroke-dasharray', fs.dash);
    else         refs.shape.removeAttribute('stroke-dasharray');
    refs.opTxt.setAttribute('fill', fs.text);

    // Keep group class accurate so CSS selectors line up.
    refs.g.setAttribute('class', `node ${cls} ${kind}`);
  }

  // Edge pass: hide edges whose endpoints aren't both visible; recolour by
  // the source node's effective class + destination kind.
  for (const [key, path] of edgeEls.entries()) {
    const [fromS, toS] = key.split('→');
    const from = Number(fromS), to = Number(toS);
    if (!visible.has(from) || !visible.has(to)) {
      path.classList.add('hidden');
      continue;
    }
    path.classList.remove('hidden');

    const srcOv = overrides[from] || {};
    const dstOv = overrides[to]   || {};
    const srcCls  = srcOv.class || (nodeEls.get(from) || {}).baseClass || 'forward';
    const dstKind = dstOv.kind  || (nodeEls.get(to)   || {}).baseKind  || 'default';

    let stroke = 'var(--edge-fwd)';
    let dash = '';
    if (srcCls === 'backward') {
      stroke = 'var(--edge-bwd)';
      dash = '5,3';
    }
    if (dstKind === 'reduce') stroke = 'var(--edge-red)';
    path.setAttribute('stroke', stroke);
    if (dash) path.setAttribute('stroke-dasharray', dash);
    else      path.removeAttribute('stroke-dasharray');
  }

  // Header + footer copy.
  document.getElementById('stageLabel').textContent = stage.label || stage.id;
  document.getElementById('stageDesc').textContent  = stage.description || '';

  const source = window._wasmReady ? 'WASM' : 'REST API';
  setStats(stage.stats || {}, source);

  // Sync the tick row.
  document.querySelectorAll('.timeline-tick').forEach((t, i) => {
    t.classList.toggle('active',  i === stageIdx);
    t.classList.toggle('visited', i <  stageIdx);
  });
  const slider = document.getElementById('timelineSlider');
  if (slider) slider.value = String(stageIdx);

  // Show the rules sub-timeline when the active stage includes backward
  // nodes and there's at least one fired rule to scrub through. Otherwise
  // hide the section so the layout collapses back to the original.
  const rulesSection = document.getElementById('rulesTimeline');
  if (rulesSection) {
    const rules = timelineState.rules;
    const visible = stageHasBackward && rules && rules.maxSeq >= 0;
    rulesSection.hidden = !visible;
    rulesSection.setAttribute('aria-hidden', visible ? 'false' : 'true');
  }
}

// ── Timeline UI wiring ────────────────────────────────────────────────────

function buildTickRow(stages) {
  const track = document.getElementById('timelineTrack');
  if (!track) return;
  track.innerHTML = '';
  stages.forEach((s, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'timeline-tick';
    btn.setAttribute('aria-label', `stage ${i + 1} of ${stages.length}: ${s.label}`);
    const dot = document.createElement('span');
    dot.className = 'tick-dot';
    btn.appendChild(dot);
    const lbl = document.createElement('span');
    lbl.className = 'tick-text';
    lbl.textContent = s.label;
    btn.appendChild(lbl);
    btn.addEventListener('click', () => applyStage(i));
    track.appendChild(btn);
  });
  const slider = document.getElementById('timelineSlider');
  if (slider) {
    slider.max = String(stages.length - 1);
    slider.value = '0';
  }
}

function wireSliderAndKeys() {
  const slider = document.getElementById('timelineSlider');
  if (slider) {
    slider.addEventListener('input', e => applyStage(Number(e.target.value)));
  }
  const rulesSlider = document.getElementById('rulesSlider');
  if (rulesSlider) {
    rulesSlider.addEventListener('input', e => applyRulesSeq(Number(e.target.value)));
  }
  // Global keyboard scrub: arrows step through stages, home/end jump to ends.
  document.addEventListener('keydown', e => {
    if (!timelineState) return;
    const n = timelineState.data.stages.length;
    let next = timelineState.currentStage;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': next = Math.min(n - 1, next + 1); break;
      case 'ArrowLeft':  case 'ArrowUp':   next = Math.max(0,     next - 1); break;
      case 'Home':       next = 0; break;
      case 'End':        next = n - 1; break;
      default: return;
    }
    e.preventDefault();
    applyStage(next);
  });
}

// ── Main entry point ──────────────────────────────────────────────────────

let currentName = 'mlp';

async function loadAndRender(name) {
  currentName = name;
  setStatus('loading timeline…');
  const svg = document.getElementById('graph-svg');

  try {
    const data = await loadTimeline(name);
    if (!data.stages || data.stages.length === 0) {
      throw new Error('timeline has no stages');
    }

    const skel = renderTimelineSkeleton(data, svg);
    if (!skel) return;
    const rulesIdx = computeRulesIndex(data.nodes);
    timelineState = {
      data, ...skel, currentStage: 0,
      rules: rulesIdx.maxSeq >= 0
        ? { maxSeq: rulesIdx.maxSeq, bySeq: rulesIdx.bySeq, cursor: rulesIdx.maxSeq }
        : null,
    };

    // Configure the rules sub-slider range and initial cursor.
    const rulesSlider = document.getElementById('rulesSlider');
    if (rulesSlider && timelineState.rules) {
      rulesSlider.max = String(timelineState.rules.maxSeq);
      rulesSlider.value = String(timelineState.rules.maxSeq);
      applyRulesSeq(timelineState.rules.maxSeq);
    }

    buildTickRow(data.stages);

    // Optional URL param ?stage=N for deep-linking and screenshot harnesses.
    let startStage = 0;
    try {
      const u = new URL(window.location.href);
      const s = parseInt(u.searchParams.get('stage') ?? '', 10);
      if (Number.isFinite(s) && s >= 0 && s < data.stages.length) startStage = s;
    } catch (_) {}
    applyStage(startStage);

    const src = window._wasmReady ? 'WASM' : 'REST API';
    setStatus(`${name} - real compiler via ${src} · ${data.stages.length} stages · arrows to scrub`);
  } catch (e) {
    setStatus('error: ' + e.message, true);
    console.error(e);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('themeToggle')?.addEventListener('click', cycleTheme);

  document.querySelectorAll('input[name="model"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (radio.checked) loadAndRender(radio.value);
    });
  });

  wireSliderAndKeys();

  setStatus('initialising compiler…');
  await initWASM();

  await loadAndRender(currentName);
});
