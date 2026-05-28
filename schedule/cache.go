package schedule

import (
	"sync/atomic"

	"github.com/georgebuilds/anneal/uop"
)

// arenaLocalKey is the lookup key within a single arena's Ext cache.
type arenaLocalKey struct {
	sinkIdx uint32
	device  string
}

// kernelKey is a secondary key for a single kernel's structural identity,
// including dispatch metadata that affects codegen (like local size).
type kernelKey struct {
	astIdx    uint32
	localSize [3]int
}

// arenaSchedCache is the concrete type stored in Arena.Ext by this package.
type arenaSchedCache map[arenaLocalKey][]ExecItem

// RenderResult carries the outputs of a shader-rendering call.
type RenderResult struct {
	WGSL           string
	LocalSize      [3]int
	WorkgroupCount [3]int
}

// WGSLRenderFunc, when non-nil, is called in cacheStore to pre-render each
// kernel's WGSL source before storing.
var WGSLRenderFunc func(ExecItem) RenderResult

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
		// Return a copy to prevent mutation of the cached items.
		cp := make([]ExecItem, len(items))
		copy(cp, items)
		return cp, true
	}
	schedCacheMisses.Add(1)
	return nil, false
}

// cacheStore inserts items into the arena-local cache.
func cacheStore(sink uop.UOp, device string, items []ExecItem) {
	var toStore []ExecItem
	if fn := WGSLRenderFunc; fn != nil {
		toStore = make([]ExecItem, len(items))
		for i := range items {
			// Populate WGSL/WS/WC in the caller's slice.
			res := fn(items[i])
			items[i].WGSL = res.WGSL
			items[i].LocalSize = res.LocalSize
			items[i].WorkgroupCount = res.WorkgroupCount

			// Store a stripped copy in the cache.
			toStore[i] = ExecItem{
				Bufs:           items[i].Bufs,
				SymVars:        items[i].SymVars,
				WGSL:           res.WGSL,
				LocalSize:      res.LocalSize,
				WorkgroupCount: res.WorkgroupCount,
				// Ast intentionally zeroed: arena reference released.
			}
		}
	} else {
		// Non-WGSL path (e.g. CPU): store as-is.
		toStore = make([]ExecItem, len(items))
		copy(toStore, items)
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
