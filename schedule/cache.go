package schedule

import (
	"sync/atomic"

	"github.com/georgebuilds/anneal/uop"
)

// arenaLocalKey is the lookup key within a single arena's Ext cache.
// The arena identity is implicit — entries live in that arena's own map
// and are GC'd with the arena, preventing cross-step retention in training loops.
//
// sinkIdx — the pre-expansion SINK node index.  Within one arena, OpSink is
//   interned, so the same logical graph always produces the same index.
//
// device  — different backends schedule differently; the same graph keyed to
//   "cpu" must not be returned for "webgpu".
type arenaLocalKey struct {
	sinkIdx uint32
	device  string
}

// arenaSchedCache is the concrete type stored in Arena.Ext by this package.
// Arena is single-threaded (not safe for concurrent mutation), so no mutex.
type arenaSchedCache map[arenaLocalKey][]ExecItem

// WGSLRenderFunc, when non-nil, is called in cacheStore to pre-render each
// kernel's WGSL source before storing.  Pre-rendering lets cacheStore zero the
// Ast field, severing the arena reference that would otherwise prevent the
// arena from being garbage-collected between training steps.
//
// Set this from the codegen package's init() so that GPU binaries automatically
// enable arena-free caching without requiring call-site changes.
var WGSLRenderFunc func(ExecItem) string

var (
	schedCacheHits   atomic.Int64
	schedCacheMisses atomic.Int64
)

func extCache(a *uop.Arena) arenaSchedCache {
	if a.Ext != nil {
		return a.Ext.(arenaSchedCache)
	}
	c := make(arenaSchedCache)
	a.Ext = c
	return c
}

// cacheLookup checks the arena-local cache. Returns (items, true) on hit.
func cacheLookup(sink uop.UOp, device string) ([]ExecItem, bool) {
	if sink.Arena().Ext == nil {
		schedCacheMisses.Add(1)
		return nil, false
	}
	key := arenaLocalKey{sinkIdx: sink.Index(), device: device}
	if items, ok := sink.Arena().Ext.(arenaSchedCache)[key]; ok {
		schedCacheHits.Add(1)
		return items, true
	}
	schedCacheMisses.Add(1)
	return nil, false
}

// cacheStore inserts items into the arena-local cache.
// When WGSLRenderFunc is set, WGSL is pre-rendered once and written into items
// in-place (so the caller's first Run call uses it too), then a stripped copy
// with Ast zeroed is stored in the cache to release the arena reference.
// When WGSLRenderFunc is nil (CPU-only paths), items are stored as-is.
func cacheStore(sink uop.UOp, device string, items []ExecItem) {
	var toStore []ExecItem
	if fn := WGSLRenderFunc; fn != nil {
		toStore = make([]ExecItem, len(items))
		for i := range items {
			wgsl := fn(items[i])
			items[i].WGSL = wgsl // populate caller's slice so executor uses cached WGSL
			toStore[i] = ExecItem{
				Bufs:    items[i].Bufs,
				SymVars: items[i].SymVars,
				WGSL:    wgsl,
				// Ast intentionally zeroed: arena reference released.
			}
		}
	} else {
		toStore = items
	}
	key := arenaLocalKey{sinkIdx: sink.Index(), device: device}
	c := extCache(sink.Arena())
	if _, exists := c[key]; !exists {
		c[key] = toStore
	}
}

// ScheduleCacheStats returns the cumulative (hits, misses) counts since the last
// ResetScheduleCache call.  Safe to call concurrently.
func ScheduleCacheStats() (hits, misses int64) {
	return schedCacheHits.Load(), schedCacheMisses.Load()
}

// ResetScheduleCache zeroes the hit/miss counters.
// Per-arena cache entries are scoped to their arena's lifetime and need no
// explicit reset; tests that check counts simply create a fresh arena.
// Intended for tests that assert specific hit/miss counts; not needed in production.
func ResetScheduleCache() {
	schedCacheHits.Store(0)
	schedCacheMisses.Store(0)
}
