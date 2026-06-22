# anneal studio - foundation (W0) developer notes

This directory is the static surface for `anneal web`. Files are embedded into
the `anneal` binary via `//go:embed` from `cmd/anneal/cmd_web.go`; the studio is
the same single binary, no bundler, no npm.

See `DESIGN.md` for the visual + interaction invariants this implements, and
`notes/anneal_web_spec.md` for the full architecture spec.

---

## File layout

```
web/
  studio.html      shell: brand, topbar, nav rail, mount points for every view
  studio.css       brand tokens (DESIGN.md §1) + layout + motion budget
  studio.js        ES module: routing, theme, keyboard, ignite, worker RPC
  worker.js        Web Worker scaffold: boots WASM, forwards RPC calls
  wasm_exec.js     copied from $(go env GOROOT)/lib/wasm/wasm_exec.js
  anneal.wasm      [not in W0] the WASM build (`GOOS=js GOARCH=wasm`)
  README-DEV.md    this file
```

Each hand-authored file is a few hundred lines and individually greppable. If
a file grows past that, extract a helper rather than letting it sprawl.

---

## Copy step (wasm_exec.js)

The Go runtime ships `wasm_exec.js` per Go version. We copy it in tree to keep
the studio self-contained. To refresh after a Go upgrade:

```bash
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" web/wasm_exec.js
```

On older Go versions the file lived at `$(go env GOROOT)/misc/wasm/wasm_exec.js`
(this is what `cmd_viz.go` and `notes/anneal_web_spec.md` reference). Either
path works; the file content is identical for a given Go version.

---

## Theme tokens

The dark and light hex values in `studio.css` are the canonical brand colours.
`DESIGN.md §1.1` (dark) and `§1.2` (light) are the source of truth. The cross-
surface pin lives in `tui/dashboard_test.go::TestColorTokenValues`; if you
change a hex here, change it there too.

Theme cycle order: `system → dark → light → system`. Default: `system`.
Persistence: `localStorage['anneal-theme']`. URL override: `?theme=dark|light|system`.

The `system` state honours `prefers-color-scheme` via CSS **and** a JS
`matchMedia` listener so a live OS theme change re-renders without a page
reload (DESIGN.md §5; spec §10).

---

## Routing table

History API only. No hash routing.

| Path                     | View      | Notes                                    |
|--------------------------|-----------|------------------------------------------|
| `/`                      | studio    | home; device card + model cards + runs   |
| `/v/<model>?stage=&node=`| visualize | viz embedded with node inspector drawer  |
| `/k/<model>?kernel=`     | kernels   | WGSL with fusion + tuned/default toggle  |
| `/x/<op>`                | explain   | rule + gradient rule list for one op     |
| `/t/<model>`             | train     | live SSE dashboard                       |
| `/g/<model>`             | generate  | inference playground                     |
| `/run/<id>`              | history   | one saved run                            |
| `/h`                     | history   | sortable table over `~/.cache/anneal/runs/` |
| `/d`                     | doctor    | native device card + browser GPU probe   |

Keyboard chord destinations: `g d` studio, `g v` visualize, `g k` kernels,
`g x` explain, `g t` train, `g g` generate, `g h` history, `g r` doctor.
`/` focuses the search input.

---

## Worker RPC protocol

The Worker is gated behind `<meta name="anneal-worker" content="/static/worker.js">`
in `studio.html`. In W0 the tag is commented out so no Worker is constructed
and no 404 for `anneal.wasm` appears in the console. When the WASM build lands,
uncomment the meta and the RPC lights up.

Wire protocol:

```js
// main → worker
{ id: <int>, fn: 'annealFoo', args: [<args>] }

// worker → main
{ id: <int>, ok: true,  result: <value> }
{ id: <int>, ok: false, error: <string> }
```

`id: 0` is reserved for unsolicited worker → main events (e.g. `{event: 'ready'}`).

From application code:

```js
import { wasm } from '/static/studio.js';
const json = await wasm.call('annealGetGraph', 'mlp', 'scheduled');
```

---

## Adding a new RPC function

1. **Go side** (inside `viz/wasm/main.go` or the studio's WASM entry):

   ```go
   js.Global().Set("annealFoo", js.FuncOf(func(this js.Value, args []js.Value) any {
       name := args[0].String()
       result, err := doFoo(name)
       if err != nil {
           return map[string]any{"error": err.Error()}
       }
       return result // string or map[string]any
   }))
   ```

2. **JS side** (anywhere in the studio):

   ```js
   import { wasm } from '/static/studio.js';
   const res = await wasm.call('annealFoo', 'mlp');
   ```

3. **Document it** in `notes/anneal_web_spec.md §4` (the WASM bridge table).

---

## Build commands

Foundation (W0):

```bash
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go test ./cmd/anneal/
CGO_ENABLED=0 go build -o /tmp/anneal ./cmd/anneal && /tmp/anneal web :0
```

WASM (when the W0+ build lands):

```bash
GOOS=js GOARCH=wasm go build -o web/anneal.wasm ./web/wasm/
```

Then uncomment the `<meta name="anneal-worker">` tag in `studio.html` and
re-embed.
