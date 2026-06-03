# DESIGN.md — anneal visual + interaction design

Status: canonical. Companion to `SPEC.md` (compiler architecture) and `notes/roadmap.md`
(project status). This document is what `viz/`, `tui/`, `cmd/anneal/`, and `web/` cite
when their comments say "see DESIGN.md §X". It is not a style guide in the cosmetic
sense; the invariants here are load-bearing for the project's thesis (DD3: the library
is the product; the surfaces are how the library is read).

No em-dashes anywhere in this doc, per project convention.

---

## §0. Purpose

This file canonicalises the visual and interaction layer of anneal:

- brand colour tokens (the literal hex values), their semantics, and their dark and
  light variants;
- the four design decisions (DD1 through DD4) that gate every surface change;
- layout rules for the studio's left-nav + main pattern and the existing viz/TUI;
- the motion budget and what is allowed to move;
- type, themes, error voice, routing, and the eight studio views;
- carried debt and binding exclusions.

What it does **not** cover: compiler architecture (`SPEC.md`), task status
(`notes/roadmap.md`), or the live spec for `anneal web` (`notes/anneal_web_spec.md`).

When code cites `DESIGN.md §N`, it means the corresponding numbered section here.
Subsection citations (§3.3, §3.4) point at named subsections of the parent §.

---

## §1. Brand tokens

Colour is meaning. Three roles run through every surface:

- **forward = teal `#00ADD8`** (Go-lineage nod; also the forward-pass node colour);
- **backward = ember `#FF7A45`** (gradient pass, "things flow back");
- **fused / kernel boundary = gold `#F2C57C`** (where kernels materialise).

These are not decorative palette picks. They are the only three meaningful colours in
the product, used everywhere a forward / backward / fused distinction is being made:
node fill+stroke, kernel firing pulses, log-stream tags, TUI ANSI, history pills.

### §1.1 Dark theme (canonical)

```
--bg          #14110F   page background
--surface     #1F1A17   panel / card
--surface-2   #2A2320   hovered panel
--text        #E8E2DA   body
--muted       #8A817A   secondary text
--faint       #5C544D   tertiary, dividers
--hair        #2D2723   1px borders

--fwd-stroke  #00ADD8   teal (forward)
--fwd-fill    #003847
--bwd-stroke  #FF7A45   ember (backward)
--bwd-fill    #3D1200
--red-stroke  #F2C57C   gold (kernel boundary / fused)
--red-fill    #3D2A00
--leaf-fill   #005566   buffer / parameter
--leaf-stroke #00ADD8
```

Dark is canonical because terminals are dark and the audience is CLI users. Every
mock and screenshot in the project is rendered against `#14110F`.

### §1.2 Light theme

```
--bg          #FBF8F3   warm paper, not pure white
--surface     #EDE9E3
--surface-2   #E3DED6
--text        #14110F
--muted       #5C544D
--faint       #8A817A
--hair        #D9D2C7

--fwd-stroke  #006F9E   teal, contrast-darkened for WCAG AA on #FBF8F3
--fwd-fill    #D5EFF6
--bwd-stroke  #B84A16   ember, contrast-darkened
--bwd-fill    #FFE5D9
--red-stroke  #7A5800   gold, contrast-darkened
--red-fill    #FFF0CE
--leaf-fill   #B3DFF0
--leaf-stroke #006F9E
```

Light theme hex values are computed so each role still meets WCAG AA against the new
background. The semantic (teal=forward, ember=backward, gold=fused) is preserved; the
luminance is shifted.

### §1.3 Where the tokens live in code

- `viz/static/style.css` — the visualiser stylesheet, cites this section as
  `DESIGN.md §3.3` (historical; this file's §1 is the canonical source).
- `tui/theme.go` — the TUI's `lipgloss` colour constants. The test
  `tui/dashboard_test.go::TestColorTokenValues` pins exact hexes against this §.
- `web/studio.css` — the studio stylesheet (this dispatch). Copies the same hexes.

The hexes are duplicated, not shared from a single Go constant, because CSS and Go
have different lifecycles; the test in §1.3 enforces they stay in sync.

---

## §2. The four design decisions (DD1 through DD4)

These are binding. Every surface change runs through this gate.

### §2.1 DD1 — colour is never alone

Teal, ember, and gold each carry a forward / backward / fused meaning, and the
meaning must not be lost when colour is unavailable. Every coloured element pairs
its colour with a second channel:

- **shape**: forward nodes are circles, backward nodes are dashed circles, kernel
  boundaries are diamonds, leaves are rounded rectangles. The viz legend names all
  four shapes alongside their colours.
- **line style**: backward edges are dashed; forward edges are solid; fused edges
  use the gold stroke weight.
- **glyph or tag**: log lines tag forward as `fwd`, backward as `bwd`, optimizer
  as `opt`. TUI substitutes ASCII tags when ANSI is unavailable.

Honoured in: `viz/static/index.html` (legend), `viz/static/style.css` (node
classes), `tui/dashboard_test.go::TestNoColorDegradation`, this DESIGN file.

### §2.2 DD2 — real compiler only

Every UOp graph, every kernel, every log line rendered in a real anneal surface is
produced by the real compiler. No mocks; no faked numbers. The exceptions are
explicit and labelled in-page:

- `notes/anneal_web_demo.html` — the static mockup; labelled "mockup" in its top bar.
- The fineprint banner in the demo file repeats "this is a static mockup, not the
  running compiler" in the status bar.

Anywhere else a measured number appears, it came from a run. The studio enforces
this with the "no measured numbers in WASM" rule (anneal\_web\_spec §1.2).

### §2.3 DD3 — the library is the product

`viz`, `tui`, `cmd/anneal`, and `web` are **surfaces** over a library. They read
the engine; they do not change how you write models. A user authoring a model in
Go must never need a browser. Conversely, a feature that only makes sense as a
browser product (a notebook, a model hub, a hosted IDE) does not get built.

Honoured by: every surface is read-only with respect to model definitions; the
public Go API is unchanged across all browser work; CLI commands stay primary.

### §2.4 DD4 — restraint

No celebratory motion. No telemetry. No phone-home. No notifications. No analytics.
No "share to social". No badges or streaks. Brand chrome recedes; the artifact
(the graph, the kernel, the loss curve) is what the eye lands on.

The motion budget (§3.4) is the operational form of DD4.

---

## §3. Layout

### §3.1 The mark

Two arcs converge into a gold node: a teal solid arc (forward), an ember dashed
arc (backward), and a `#F2C57C` gold dot where they meet. The mark is inline SVG;
no icon font, no raster. Used at 22px in the studio brand cell and at 96×72px in
the studio hero. The mark **ignites once** on first load per browser session
(localStorage flag); reload does not re-ignite.

### §3.2 Studio layout

The studio uses a fixed three-zone grid:

```
[ brand   ][ topbar              ]   44px
[ nav     ][ main                ]   1fr
[ nav     ][ status              ]   28px
```

- **brand cell** — wordmark + mockup tag (mockup tag absent in production).
- **topbar** — breadcrumbs (left), device pill + theme toggle (right).
- **nav rail** — fixed 220px left rail; sections (surfaces / persistence);
  per-item keyboard shortcut shown as a `g X` chip.
- **main** — scrollable content area, one `.view` active at a time.
- **status bar** — build, git SHA, run cache count, connection dot.

Only one optional element ever lands inside main: an inspector drawer (the
visualize view's node inspector). The drawer is dismissible; it never becomes a
permanent sidebar (`anneal_web_spec §5.2`).

### §3.3 The eight studio views (cross-ref §8)

The nav rail holds eight entries. See §8 for the per-view charter. Tokens for the
left-rail active-state highlight: gold border-left and a faint gold gradient
overlay.

### §3.4 Motion budget

Only four things may move; nothing else does. `prefers-reduced-motion: reduce`
disables all four.

1. **Ignite-once wordmark** on first page load (per browser session, persisted
   via localStorage). Two arcs draw in, the gold node fades in. Total ≈600ms.
2. **Gold pulse on dispatch** when a fused kernel fires (train view, viz). Box
   shadow expands to 3px gold at 28% opacity over 1.8s, ease-in-out, infinite
   while the kernel is the currently-active dispatch.
3. **Rule-fired pulse** in the explain view and on backward nodes in viz when
   the rules sub-timeline advances. Stroke width briefly grows from 1.6 to 3.4
   and back over 700ms; runs once per advance.
4. **Cross-stage node fades** in viz when the pipeline scrubber advances. Nodes
   joining the current stage fade in; nodes leaving fade out. 250ms ease.

Anything else is a bug. No spinners (the WASM bridge is fast or it surfaces an
error). No bouncing buttons. No confetti. No toast notifications.

### §3.5 The TUI dashboard

The TUI mirrors the studio's information hierarchy with the same colour
semantics and the same restraint. The chrome is ASCII box-drawing; the colour
tokens come from `tui/theme.go`. `tui/dashboard.go` cites this section for the
viz prototype gateway (see `tui/dashboard.go:262`).

---

## §4. Type

- **Primary: JetBrains Mono** (also accept `Commit Mono`, `IBM Plex Mono`,
  `ui-monospace`). 13px body, 11px chrome, 10px hint chips. `font-feature-settings:
  "tnum"` on any cell that shows a number (loss, step, time) so columns align.
- **Display: Inter** for view `<h1>` and the hero quote, 18px and 22px respectively,
  letter-spacing -0.01em.
- **No icon font.** Icons are inline SVG. The wordmark mark and the legend glyphs
  are the only "icons" the product ships.
- **No emoji in production UI.** The legend uses `●`, `⊙`, `◆`, `▭` (geometric
  shapes, not emoji codepoints) and the brand mark.

`web/studio.html` preconnects to `fonts.googleapis.com` for both families.

Error voice (cross-ref §6) is sentence case and lowercase; usage text in
`cmd/anneal/main.go::usageText` follows the same convention (the cli test
`TestHelpSentenceCase` pins this).

---

## §5. Themes

Tri-state: `system`, `dark`, `light`. `system` is the default. The value is
persisted in `localStorage['anneal-theme']`. A `?theme=` URL query parameter
wins over localStorage for screenshot harnesses.

Cycle order is `system → dark → light → system`. The theme toggle button is
the same control on viz and on the studio.

When `system` is selected the page honours `prefers-color-scheme` via a CSS
`@media (prefers-color-scheme: light) :root[data-theme="system"]` block, **and**
the studio attaches a `matchMedia('(prefers-color-scheme: dark)')` change
listener so a live OS theme change re-themes the page without a reload. This is
the spec §10 fix called out for the studio specifically.

---

## §6. Errors

Blameless and actionable. Two surfaces:

- **Native errors** (no GPU, SSE drop, importer failure) name the offending op
  or path and carry remediation. `cmd/anneal/output.go::noAdapterError()` is the
  template; it cites this section as `DESIGN.md §4` historically (the file's §6
  here is the canonical source; treat §4 citations as §6 going forward).
- **WASM errors** (import failed, unsupported op) name the offending node id and
  op kind in-view, never as a stack trace. The studio's ONNX dropzone surfaces an
  "unsupported ops" panel (`anneal_web_spec §8`) rather than a silent failure.

Error voice is sentence case, lowercase headings, no exclamation marks, no
"sorry". The TUI's no-color degradation is tested at `tui/dashboard_test.go`
under "DESIGN.md §6 degradation hard requirement".

---

## §7. Routing and URLs

History API only. No hash routing. `viz` may still use hash for legacy
compatibility; the studio does not.

Deep links per view (the studio's URL contract):

```
/                       home (studio)
/v/<model>?stage=&node= visualize, optional stage and node id
/k/<model>?kernel=      kernels, optional kernel id
/x/<op>                 explain, scoped to an op
/run/<bundleId>         a saved run
/t/<model>              train, model preselected
/g/<model>              generate, model preselected
/h                      history
/d                      doctor
```

`pushState` advances the URL; `popstate` re-renders the matching view. Every
view is responsible for serializing its own state back into the URL (e.g.
visualize writes `?stage=3&node=42` when the scrubber moves).

The viz prototype's static deep-link semantics are mirrored in viz comments
under `DESIGN.md §7` ("the visual transitions are the teaching"). The studio's
node inspector cites the same property.

---

## §8. The studio's eight views

One paragraph per view; the binding spec lives in `notes/anneal_web_spec.md §5`.

1. **studio (home).** Device card, model cards, recent runs, ONNX and tensor
   inspect dropzones, restrained thesis statement. WASM. Cross-ref spec §5.1.
2. **visualize.** Embed the existing viz artifact verbatim plus a node inspector
   drawer. WASM. Spec §5.2.
3. **kernels.** Per-kernel WGSL with fusion boundaries; tuned vs default toggle.
   WASM (untuned) + native-assisted (tuned). Spec §5.3.
4. **explain.** Op list and rule firings for the selected op with a mini-graph
   animation. WASM. Spec §5.4.
5. **train.** Live training dashboard over SSE. Native. Spec §5.5.
6. **generate.** Inference playground over SSE, with click-through from token
   to producing kernel. Native. Spec §5.6.
7. **history.** Sortable table over `~/.cache/anneal/runs/`; resurrect any run,
   compare any two. Native (disk) + WASM (re-render). Spec §5.7.
8. **doctor.** Native device card + in-page `navigator.gpu` probe, side by side.
   Native + browser probe. Spec §5.8.

Older citations (e.g. `tui/dashboard.go:262` "DESIGN.md §8") referenced this
section before the studio existed; the gist holds (the eighth surface is the
prototype gateway), and the studio's view-8 (doctor) takes the same slot.

---

## §9. Encoding rules (colour, shape, weight)

Two channels minimum on every meaningful element. The viz legend in
`viz/static/index.html` is the canonical example: each entry is colour + shape +
label. The TUI dashboard restates the same encoding with ANSI + glyph + label.
Tests at `tui/dashboard_test.go` ("DESIGN.md §9, §3.3") pin the encoding.

For tables and chips:
- pill borders use `--hair` plus a role-tinted dim variant (`--gold-dim`,
  `--teal-dim`, `--ember-dim`) so the pill carries colour + outline.
- numeric cells use `font-feature-settings: "tnum"` so a column of numbers
  reads as a column even without a tabular font.
- a status pill ("done", "cancelled") is text + colour + position.

When a surface degrades to no-colour (`NO_COLOR=1`, monochrome printer, light
mode at low contrast), the shape and label still carry the meaning.

---

## §10. Exclusions (binding)

Reaffirmed here so anyone landing in DESIGN.md first knows the shape of the
product. These match `notes/anneal_web_spec.md §5.9`:

- **No notebook or code-execution surface.** Writing models is done in Go, in
  an editor. DD3.
- **No model hub, registry, sharing, or leaderboard.** Runs are disk artifacts
  the user owns; they are not uploaded.
- **No telemetry. No phone-home. No analytics.** Not in WASM, not in native, not
  conditional, not opt-in-with-defaults.
- **No hosted or multi-user mode. No auth.** Local single-user only.
- **No marketing motion.** No "celebrate" interactions. No streaks. No badges.
- **No new third-party JS dependencies.** Pure browser APIs. No npm, no
  bundler.

---

## §11. Carried debt

Tracked so the next pass catches it.

- **AGENTS.md is referenced but not present.** `viz/graph.go:82` cites it as
  "SPEC §1.3 / AGENTS.md"; `notes/roadmap.md` flags it as carried debt; the
  roadmap entry has been open since the file was first proposed. Either author
  AGENTS.md or update the citing comment. This file does not invent AGENTS.md's
  contents.
- **Historical section-number drift.** Some in-code citations reference older
  section numbers (e.g. `output.go` "§4" for what is now §6; `tui/dashboard.go`
  "§8" for what is now §8 but in a slightly different layout). The citations
  still resolve to a real section here; a sweep to normalise the numbers is
  worth doing but is not gating.
- **Spec §11.1.** This file is the closing of that item. The studio spec
  (`notes/anneal_web_spec.md`) names DESIGN.md as the canonical home for visual
  invariants; this file is now that.

---

## Appendix A. Cross-references

- `SPEC.md` — compiler architecture (UOps, rangeify, scheduler, codegen).
- `notes/roadmap.md` — phase status and what's next.
- `notes/anneal_web_spec.md` — the binding architecture spec for the studio.
- `notes/anneal_web_demo.html` — static mockup of the studio; DD2's only
  exception, labelled in-page.
- `CONTRIBUTING.md §"keep the docs honest"` — the rule that change to a
  surface lands with a DESIGN.md update in the same PR.
