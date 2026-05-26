package shape

import "fmt"

// View is the per-op stride/offset/mask representation of a tensor's index mapping.
//
// All fields are int64 in v1; see sint.go for the symbolic-seam migration path.
// Contiguous is precomputed at construction and never recomputed lazily.
type View struct {
	Shape      []int64
	Strides    []int64
	Offset     int64
	Mask       [][2]int64 // nil = no mask (all valid)
	Contiguous bool
}

// NewView constructs a View, normalises a trivial full-range mask to nil, and
// precomputes Contiguous.  Strides are taken as-is; use stridesForShape to
// produce C-contiguous strides for a fresh tensor.
func NewView(shape, strides []int64, offset int64, mask [][2]int64) View {
	mask = normalizeMask(mask, shape)
	contig := offset == 0 && mask == nil && stridesEqual(strides, stridesForShape(shape))
	return View{
		Shape:      cloneI64(shape),
		Strides:    cloneI64(strides),
		Offset:     offset,
		Mask:       mask,
		Contiguous: contig,
	}
}

// NewContiguousView returns a fresh row-major View for shape (offset 0, no mask).
func NewContiguousView(shape []int64) View {
	return NewView(shape, stridesForShape(shape), 0, nil)
}

// ── stride helpers ────────────────────────────────────────────────────────────

// stridesForShape returns C-contiguous (row-major) strides for shape.
// Dimensions of size 1 get stride 0 (canonicalized).
func stridesForShape(shape []int64) []int64 {
	n := len(shape)
	if n == 0 {
		return []int64{}
	}
	st := make([]int64, n)
	acc := int64(1)
	for i := n - 1; i >= 0; i-- {
		if shape[i] != 1 {
			st[i] = acc
		}
		acc *= shape[i]
	}
	return st
}

// ── mask helpers ──────────────────────────────────────────────────────────────

// normalizeMask returns nil if mask is nil or every dim covers its full range.
func normalizeMask(mask [][2]int64, shape []int64) [][2]int64 {
	if mask == nil {
		return nil
	}
	for i, m := range mask {
		if m[0] != 0 || m[1] != shape[i] {
			return mask
		}
	}
	return nil
}

// ── movement ops ──────────────────────────────────────────────────────────────

// Expand broadcasts dimensions.  Caller must ensure new_shape[i] == shape[i]
// for all dims where shape[i] != 1.  Expanded dims keep stride 0.
func (v View) Expand(newShape []int64) View {
	if len(newShape) != len(v.Shape) {
		panic(fmt.Sprintf("shape: expand: rank mismatch %d vs %d", len(v.Shape), len(newShape)))
	}
	for i, s := range v.Shape {
		if s != newShape[i] && s != 1 {
			panic(fmt.Sprintf("shape: expand: cannot expand dim %d size %d to %d", i, s, newShape[i]))
		}
	}

	// zero-size input → fresh contiguous view
	for _, s := range v.Shape {
		if s == 0 {
			return NewContiguousView(newShape)
		}
	}

	strides := cloneI64(v.Strides)
	var mask [][2]int64
	if v.Mask != nil {
		mask = cloneMask(v.Mask)
	}

	for i, ns := range newShape {
		if v.Shape[i] == ns {
			continue
		}
		// Expanding size-1 dim; its stride is already 0 from canonicalization.
		if mask != nil {
			if v.Mask[i] == ([2]int64{0, 1}) {
				mask[i] = [2]int64{0, ns}
			} else {
				mask[i] = [2]int64{0, 0} // was already masked out → stays out
			}
		}
	}

	ns := cloneI64(newShape)
	return NewView(ns, strides, v.Offset, mask)
}

// Permute reorders dimensions.  order is a permutation of [0, n).
func (v View) Permute(order []int) View {
	n := len(v.Shape)
	if len(order) != n {
		panic("shape: permute: order length mismatch")
	}
	shape := make([]int64, n)
	strides := make([]int64, n)
	var mask [][2]int64
	if v.Mask != nil {
		mask = make([][2]int64, n)
	}
	for i, a := range order {
		shape[i] = v.Shape[a]
		strides[i] = v.Strides[a]
		if mask != nil {
			mask[i] = v.Mask[a]
		}
	}
	return NewView(shape, strides, v.Offset, mask)
}

// Pad adds zero padding.  arg[i] = {lo, hi} adds lo elements before and hi after dim i.
func (v View) Pad(arg [][2]int64) View {
	if len(arg) != len(v.Shape) {
		panic("shape: pad: arg length mismatch")
	}
	anyNonzero := false
	for _, ab := range arg {
		if ab[0] != 0 || ab[1] != 0 {
			anyNonzero = true
			break
		}
	}
	if !anyNonzero {
		return v
	}

	// zvarg[i] = {-lo, shape[i]+hi}  — the resize bounds in current coordinates
	// newMask[i] = {lo, shape[i]+lo} — the valid region after padding
	zvarg := make([][2]int64, len(arg))
	newMask := make([][2]int64, len(arg))
	for i, ab := range arg {
		lo, hi := ab[0], ab[1]
		zvarg[i] = [2]int64{-lo, v.Shape[i] + hi}
		newMask[i] = [2]int64{lo, v.Shape[i] + lo}
	}
	return v.unsafeResize(zvarg, newMask)
}

// Shrink reduces dimensions.  arg[i] = {lo, hi} selects the half-open range [lo, hi) of dim i.
func (v View) Shrink(arg [][2]int64) View {
	if len(arg) != len(v.Shape) {
		panic("shape: shrink: arg length mismatch")
	}
	return v.unsafeResize(arg, nil)
}

// Flip reverses elements along dimensions where axes[i] is true.
func (v View) Flip(axes []bool) View {
	if len(axes) != len(v.Shape) {
		panic("shape: flip: axes length mismatch")
	}
	offset := v.Offset
	strides := cloneI64(v.Strides)
	var mask [][2]int64
	if v.Mask != nil {
		mask = cloneMask(v.Mask)
	}
	for i, flip := range axes {
		if !flip {
			continue
		}
		offset += (v.Shape[i] - 1) * v.Strides[i]
		strides[i] = -v.Strides[i]
		if mask != nil {
			s := v.Shape[i]
			mask[i] = [2]int64{s - v.Mask[i][1], s - v.Mask[i][0]}
		}
	}
	return NewView(cloneI64(v.Shape), strides, offset, mask)
}

// Reshape attempts to produce a View with newShape over the same data.
// Returns (newView, true) on success.
// Returns (View{}, false) if strides or mask cannot be expressed in newShape;
// callers (ShapeTracker) must then push a fresh contiguous view.
func (v View) Reshape(newShape []int64) (View, bool) {
	if !sizeMatch(v.Shape, newShape) {
		panic(fmt.Sprintf("shape: reshape: size mismatch %v -> %v", v.Shape, newShape))
	}
	if shapesEqual(v.Shape, newShape) {
		return v, true
	}

	// Zero-size source → any new shape is a fresh contiguous view.
	for _, s := range v.Shape {
		if s == 0 {
			return NewContiguousView(newShape), true
		}
	}

	// Reshaping to scalar with a fully-masked-out dimension.
	if len(newShape) == 0 && v.Mask != nil {
		for _, m := range v.Mask {
			if m[0] == m[1] {
				return View{}, false
			}
		}
	}

	if v.Contiguous {
		return NewContiguousView(newShape), true
	}

	// Non-contiguous: try to re-express strides via merged dimension groups.
	newStrides, ok := reshapeStrides(v.Shape, v.Strides, v.Mask, newShape)
	if !ok {
		return View{}, false
	}

	// Translate the mask into the new coordinate system.
	newMask, ok := reshapeMask(v.Mask, v.Shape, newShape)
	if !ok {
		return View{}, false
	}

	// extra_offset adjusts for any per-dim mask lo shift between old and new coordinates.
	extraOffset := int64(0)
	if v.Mask != nil {
		for i, m := range v.Mask {
			extraOffset += m[0] * v.Strides[i]
		}
	}
	if newMask != nil {
		for i, m := range newMask {
			extraOffset -= m[0] * newStrides[i]
		}
	}

	return NewView(newShape, newStrides, v.Offset+extraOffset, newMask), true
}

// ── internal helpers ──────────────────────────────────────────────────────────

// unsafeResize is the shared core of Pad and Shrink.
// arg[i] = {lo, hi} sets the new slice bounds in the CURRENT coordinate system.
// newMask, if non-nil, is intersected with (the transformed) existing mask.
func (v View) unsafeResize(arg [][2]int64, newMask [][2]int64) View {
	n := len(v.Shape)
	offset := v.Offset
	for i := 0; i < n; i++ {
		offset += v.Strides[i] * arg[i][0]
	}

	shape := make([]int64, n)
	for i, ab := range arg {
		shape[i] = ab[1] - ab[0]
	}

	var mask [][2]int64
	if v.Mask != nil {
		// Shift the existing mask into the new coordinate system, then clamp to [0, new_size).
		mask = make([][2]int64, n)
		for i, m := range v.Mask {
			ax, ay := arg[i][0], arg[i][1]
			lo := imax(0, imin(m[0]-ax, ay-ax))
			hi := imax(0, imin(m[1]-ax, ay-ax))
			mask[i] = [2]int64{lo, hi}
		}
		if newMask != nil {
			for i, nm := range newMask {
				mask[i] = [2]int64{imax(mask[i][0], nm[0]), imin(mask[i][1], nm[1])}
			}
		}
	} else if newMask != nil {
		mask = cloneMask(newMask)
	}

	return NewView(shape, cloneI64(v.Strides), offset, mask)
}

// ── mergeDim and reshapeStrides ───────────────────────────────────────────────

// mergeDim is one contiguous group produced by collectMergeDims.
type mergeDim struct {
	Size     int64 // product of dim sizes in group
	Stride   int64 // stride of the rightmost (innermost) dim in the group
	RealSize int64 // effective size excluding broadcast (stride-0) dimensions
}

// collectMergeDims groups consecutive dimensions that are contiguous in memory
// or broadcast (stride-0), returning the minimum number of groups needed to
// describe the strides.
func collectMergeDims(shape, strides []int64, mask [][2]int64) []mergeDim {
	n := len(shape)
	if n == 0 {
		return nil
	}

	merging := maskRangeOne(mask, 0, shape[0])
	realSize := shape[0]
	if strides[0] == 0 {
		realSize = 0
	}
	ret := []mergeDim{{shape[0], strides[0], realSize}}

	for i := 1; i < n; i++ {
		s := shape[i]
		st := strides[i]
		if s == 1 {
			merging = maskRangeOne(mask, i, s)
			continue
		}
		last := &ret[len(ret)-1]
		if merging || last.Stride == s*st {
			var newReal int64
			if merging {
				newReal = s
			} else {
				newReal = last.RealSize * s
			}
			last.Size *= s
			last.Stride = st
			last.RealSize = newReal
		} else {
			ret = append(ret, mergeDim{s, st, s})
		}
		merging = maskRangeOne(mask, i, s)
	}
	return ret
}

// maskRangeOne reports whether dim i has a mask range of exactly 1 element
// (used to detect the "merging" flag for broadcast groups in collectMergeDims).
func maskRangeOne(mask [][2]int64, i int, dimSize int64) bool {
	if mask != nil {
		return mask[i][1]-mask[i][0] == 1
	}
	return dimSize == 1
}

// reshapeStrides computes the new stride slice for Reshape via collectMergeDims.
// Returns (strides, true) or (nil, false) if the strides cannot be re-expressed.
func reshapeStrides(shape, strides []int64, mask [][2]int64, newShape []int64) ([]int64, bool) {
	dims := collectMergeDims(shape, strides, mask)

	// newShapeRev: new dims in right-to-left order (innermost first).
	newShapeRev := make([]int64, len(newShape))
	for i, s := range newShape {
		newShapeRev[len(newShape)-1-i] = s
	}

	var rStrides []int64 // built right-to-left
	ni := 0

	for di := len(dims) - 1; di >= 0; di-- {
		d := dims[di]
		acc := int64(1)
		newSt := d.Stride

		for acc <= d.Size && acc != d.Size {
			if ni >= len(newShapeRev) {
				// More old product than new dims → mismatch.
				return nil, false
			}
			nd := newShapeRev[ni]
			ni++
			if nd == 0 {
				break
			}
			rStrides = append(rStrides, newSt*acc)
			acc *= nd
			if acc >= d.RealSize {
				newSt = 0
			}
		}
		if acc != d.Size {
			return nil, false
		}
	}

	out := make([]int64, len(newShape))
	// Leftmost new dims that weren't consumed get stride 0.
	for i, st := range rStrides {
		out[len(newShape)-1-i] = st
	}
	return out, true
}

// ── reshapeMask ───────────────────────────────────────────────────────────────

// reshapeMask translates mask from shape into newShape coordinates.
// Returns (newMask, true) on success; newMask == nil means no mask needed.
// Returns (nil, false) when the masked valid region is not rectangular in newShape.
//
// Algorithm: process new dims right-to-left (innermost first), maintaining a
// flat valid range [lo, hi) within an accumulated old-dim group of size sz.
// At each step the rightmost new dim nd is "extracted" from the flat range via
// one of two rectangle-preserving splits:
//
//	A. single outer row  (lo/nd == (hi-1)/nd)         → dim range [lo%nd, …)
//	B. all full inner rows (lo%nd==0 && hi%nd==0)      → dim range [0, nd) (full)
//
// Any other configuration is non-rectangular → return false.
func reshapeMask(mask [][2]int64, shape, newShape []int64) ([][2]int64, bool) {
	if mask == nil {
		return nil, true
	}

	// Trivially-full mask → no new mask needed.
	allFull := true
	for i, m := range mask {
		if m[0] != 0 || m[1] != shape[i] {
			allFull = false
			break
		}
	}
	if allFull {
		return nil, true
	}

	newMask := make([][2]int64, len(newShape))

	sz := shape[len(shape)-1]
	lo := mask[len(mask)-1][0]
	hi := mask[len(mask)-1][1]
	oi := len(shape) - 2

	for ni := len(newShape) - 1; ni >= 0; ni-- {
		nd := newShape[ni]

		// Grow the old group until sz >= nd.
		for sz < nd {
			if oi < 0 {
				return nil, false
			}
			prevLo := mask[oi][0]
			prevHi := mask[oi][1]
			prevSz := shape[oi]
			oi--

			if lo == 0 && hi == sz {
				// Inner group is full: scale up the flat range.
				lo = prevLo * sz
				hi = prevHi * sz
			} else if prevHi-prevLo == 1 {
				// Single outer row: offset the inner range.
				lo = prevLo*sz + lo
				hi = prevLo*sz + hi
			} else {
				return nil, false
			}
			sz *= prevSz
		}

		if sz%nd != 0 {
			return nil, false
		}

		loOuter := lo / nd
		hiOuterIncl := (hi - 1) / nd
		loInner := lo % nd
		hiInner := (hi-1)%nd + 1

		switch {
		case loOuter == hiOuterIncl:
			// Case A: single outer row.
			newMask[ni] = [2]int64{loInner, hiInner}
			lo = loOuter
			hi = loOuter + 1
		case loInner == 0 && hiInner == nd:
			// Case B: every row has full inner range.
			newMask[ni] = [2]int64{0, nd}
			lo = loOuter
			hi = hiOuterIncl + 1
		default:
			return nil, false
		}
		sz /= nd
	}

	// Any remaining old dims must have full mask.
	for ; oi >= 0; oi-- {
		if mask[oi][0] != 0 || mask[oi][1] != shape[oi] {
			return nil, false
		}
	}
	if lo != 0 || hi != sz {
		return nil, false
	}

	return normalizeMask(newMask, newShape), true
}

// ── small utilities ───────────────────────────────────────────────────────────

func imax(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
func imin(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func cloneI64(s []int64) []int64 {
	if s == nil {
		return nil
	}
	c := make([]int64, len(s))
	copy(c, s)
	return c
}

func cloneMask(m [][2]int64) [][2]int64 {
	if m == nil {
		return nil
	}
	c := make([][2]int64, len(m))
	copy(c, m)
	return c
}

func shapesEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func stridesEqual(a, b []int64) bool { return shapesEqual(a, b) }

func sizeMatch(a, b []int64) bool {
	pa, pb := int64(1), int64(1)
	for _, s := range a {
		pa *= s
	}
	for _, s := range b {
		pb *= s
	}
	return pa == pb
}
