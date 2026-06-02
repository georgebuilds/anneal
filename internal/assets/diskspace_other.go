//go:build !darwin && !linux

package assets

// checkDiskSpace is a no-op on platforms where we have not wired up a
// Statfs equivalent. The project targets macOS and Linux; Windows builds
// just skip the precheck and rely on the download itself to fail with
// ENOSPC if the disk fills.
func checkDiskSpace(dir string, need int64) error {
	return nil
}
