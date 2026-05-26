package uop

// KernelInfo is the arg payload for a kernel-level SINK node.
// NumParams is the total number of PARAM nodes in the kernel:
// PARAM(arg=0) is always the output buffer; PARAM(arg=1..NumParams-1) are inputs.
type KernelInfo struct {
	NumParams int
}
