//go:build !js

// Package web embeds the anneal studio's static assets (studio.html,
// studio.css, studio.js, worker.js, wasm_exec.js) so cmd/anneal can serve
// them as a single binary.
//
// The WASM artifact (anneal.wasm) is deliberately NOT embedded yet; it lands
// when the WASM build step lands. TODO(W-wasm): add `anneal.wasm` to the
// embed line and uncomment the <meta name="anneal-worker"> tag in
// studio.html.
//
// See cmd/anneal/cmd_web.go for the HTTP mux, DESIGN.md for visual
// invariants, and notes/anneal_web_spec.md for the architecture spec.
package web

import (
	"embed"
	"io/fs"
)

//go:embed studio.html studio.css studio.js worker.js wasm_exec.js visualize_embed.html
var files embed.FS

// FS returns the embedded studio assets rooted at this package (so a caller
// reads "studio.html", not "web/studio.html").
func FS() fs.FS { return files }
