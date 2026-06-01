// captrace is a one-shot trace serializer. It runs the real anneal compiler
// over the named example via viz.BuildTimeline and writes the JSON payload to
// stdout. Used to freeze a real compilation for the static Pages demo.
package main

import (
	"fmt"
	"os"

	_ "github.com/georgebuilds/anneal/examples"
	"github.com/georgebuilds/anneal/viz"
)

func main() {
	name := "mlp"
	if len(os.Args) > 1 {
		name = os.Args[1]
	}
	t, err := viz.BuildTimeline(name)
	if err != nil {
		fmt.Fprintln(os.Stderr, "captrace:", err)
		os.Exit(1)
	}
	b, err := t.ToJSON()
	if err != nil {
		fmt.Fprintln(os.Stderr, "captrace: json:", err)
		os.Exit(1)
	}
	_, _ = os.Stdout.Write(b)
}
