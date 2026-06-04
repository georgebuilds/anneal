// captrace is a one-shot trace serializer. It runs the real anneal compiler
// over the named example via viz.BuildTimeline and writes the JSON payload to
// stdout. Used to freeze a real compilation for the static Pages demo.
package main

import (
	"fmt"
	"io"
	"os"

	_ "github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/viz"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	name := "mlp"
	if len(args) > 0 {
		name = args[0]
	}
	t, err := viz.BuildTimeline(name)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "captrace:", err)
		return 1
	}
	b, err := t.ToJSON()
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "captrace: json:", err)
		return 1
	}
	_, _ = stdout.Write(b)
	return 0
}
