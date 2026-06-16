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

// timelineJSON is the seam captrace uses to build and serialize a timeline.
// It is a variable so tests can force the serialization error path without a
// GPU or a real compilation.
var timelineJSON = func(name string) ([]byte, error) {
	t, err := viz.BuildTimeline(name)
	if err != nil {
		return nil, err
	}
	return t.ToJSON()
}

// osExit is a seam so tests can observe main()'s exit code without terminating
// the test process.
var osExit = os.Exit

func main() {
	osExit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	name := "mlp"
	if len(args) > 0 {
		name = args[0]
	}
	b, err := timelineJSON(name)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "captrace:", err)
		return 1
	}
	_, _ = stdout.Write(b)
	return 0
}
