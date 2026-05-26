package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
)

// parsedFlags holds the values of the global flags.
type parsedFlags struct {
	device string
	debug  int
	viz    bool
}

// parseFlags adds --device, --debug, --viz to a FlagSet named `name`, parses
// args, then applies VIZ=1 and DEBUG=N env aliases for any flag not explicitly
// set via the command line. Explicit flags always win over env aliases.
//
// Returns the parsed flags, the remaining positional args, and any error.
func parseFlags(name string, args []string) (*parsedFlags, []string, error) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)

	device := fs.String("device", "webgpu", "target device")
	debug := fs.Int("debug", 0, "debug verbosity level (0–3)")
	viz := fs.Bool("viz", false, "enable graph visualization")

	if err := fs.Parse(args); err != nil {
		return nil, nil, fmt.Errorf("flag parse: %w", err)
	}

	// Detect which flags were explicitly set on the command line.
	explicitlySet := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		explicitlySet[f.Name] = true
	})

	// Apply env aliases only when the corresponding flag was not explicitly set.
	if !explicitlySet["viz"] {
		if v := os.Getenv("VIZ"); v == "1" {
			*viz = true
		}
	}
	if !explicitlySet["debug"] {
		if v := os.Getenv("DEBUG"); v != "" {
			n, err := strconv.Atoi(v)
			if err == nil {
				*debug = n
			}
		}
	}

	return &parsedFlags{
		device: *device,
		debug:  *debug,
		viz:    *viz,
	}, fs.Args(), nil
}
