//go:build darwin || linux

package assets

import (
	"fmt"
	"syscall"
)

// checkDiskSpace returns an error when the filesystem holding dir has
// fewer than need bytes free. Best-effort: a Statfs failure is silently
// ignored so we never block a download on a transient stat issue.
func checkDiskSpace(dir string, need int64) error {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return nil
	}
	// Bsize is int32 on darwin and int64 on linux; cast both operands to int64
	// for cross-platform arithmetic. The redundant cast on linux is intentional.
	avail := int64(st.Bavail) * int64(st.Bsize) //nolint:unconvert
	if avail < need {
		return fmt.Errorf("assets: insufficient disk space at %s: need %d bytes, have %d", dir, need, avail)
	}
	return nil
}
