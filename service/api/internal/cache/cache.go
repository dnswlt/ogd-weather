package cache

import (
	"container/heap"
	"container/list"
	"fmt"
	"sync"
	"time"
)

type Cache interface {
	Len() int
	Capacity() int
	Usage() int
	Get(key string) (any, bool)
	Put(key string, item any, size int, ttl time.Duration)
	Purge()
}

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (c realClock) Now() time.Time {
	return time.Now()
}

// entry represents a cached item. It includes pointers for the LRU list
// and an index for the TTL min-heap.
type entry struct {
	key     string
	item    any
	size    int
	expires time.Time

	// element is the pointer to the item in the LRU list.
	element *list.Element
	// index is the item's index in the TTL heap.
	// It's -1 if the item has no TTL and is not in the heap.
	index int
}

// lruCache implements a high-performance LRU cache with TTL.
// It uses a hash map, a doubly-linked list for LRU, and a min-heap for TTL.
type lruCache struct {
	mu       sync.Mutex
	capacity int
	usage    int

	// data provides O(1) lookup for entries.
	data map[string]*entry
	// lru tracks the least recently used entries. Front is newest.
	lru *list.List
	// ttlHeap is a min-heap ordered by expiration time, for O(log n) expiration checks.
	ttlHeap priorityQueue
	// A RealClock in normal use cases,
	clock clock
}

// disabledCache is an implementation of Cache that does not cache anything.
type disabledCache struct{}

func (c disabledCache) Len() int                                              { return 0 }
func (c disabledCache) Capacity() int                                         { return 0 }
func (c disabledCache) Usage() int                                            { return 0 }
func (c disabledCache) Get(key string) (any, bool)                            { return nil, false }
func (c disabledCache) Put(key string, item any, size int, ttl time.Duration) {}
func (c disabledCache) Purge()                                                {}

func newWithClock(clock clock, capacity int) *lruCache {
	return &lruCache{
		capacity: capacity,
		data:     make(map[string]*entry),
		lru:      list.New(),
		ttlHeap:  make(priorityQueue, 0),
		clock:    clock,
	}
}

// New creates a capacity-limited, high-performance LRU cache.
// capacity < 0 means the cache is unbounded.
// capacity == 0 means the cache is disabled and does not cache anything.
func New(capacity int) Cache {
	if capacity == 0 {
		return disabledCache{}
	}
	return newWithClock(realClock{}, capacity)
}

// Len returns number of items currently in cache.
func (c *lruCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.data)
}

func (c *lruCache) Usage() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

func (c *lruCache) Capacity() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.capacity
}

// Purge resets the cache to an empty state.
func (c *lruCache) Purge() {
	c.mu.Lock()
	defer c.mu.Unlock()

	clear(c.data)
	c.lru.Init()
	// Best practice: nil out pointers in the slice to help the GC
	clear(c.ttlHeap)
	c.ttlHeap = c.ttlHeap[:0]
	c.usage = 0
}

// Put adds or updates an item, its size, and its TTL.
// A ttl of 0 or less means the item never expires.
// The value of `size` MUST be greater than zero.
func (c *lruCache) Put(key string, item any, size int, ttl time.Duration) {
	if size <= 0 {
		panic(fmt.Sprintf("cache: Put called with invalid size %d for key %q", size, key))
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	var expires time.Time
	if ttl > 0 {
		expires = c.clock.Now().Add(ttl)
	}

	// Update existing entry.
	if e, ok := c.data[key]; ok {
		// Update usage and properties.
		c.usage -= e.size
		c.usage += size
		e.item = item
		e.size = size

		// Move to front of LRU list.
		c.lru.MoveToFront(e.element)

		// Update TTL and adjust its position in the heap.
		hadTTL := !e.expires.IsZero()
		e.expires = expires
		hasTTL := !e.expires.IsZero()

		if hadTTL && hasTTL {
			heap.Fix(&c.ttlHeap, e.index)
		} else if hadTTL && !hasTTL {
			heap.Remove(&c.ttlHeap, e.index)
		} else if !hadTTL && hasTTL {
			heap.Push(&c.ttlHeap, e)
		}

		c.evictIfNeeded()
		return
	}

	// Add new entry.
	e := &entry{
		key:     key,
		item:    item,
		size:    size,
		expires: expires,
		index:   -1, // Not in the heap until we push it.
	}
	e.element = c.lru.PushFront(e)
	c.data[key] = e
	c.usage += size

	if !e.expires.IsZero() {
		heap.Push(&c.ttlHeap, e)
	}

	c.evictIfNeeded()
}

// Get retrieves an item. It returns nil, false if the item is not found or is expired.
func (c *lruCache) Get(key string) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	e, ok := c.data[key]
	if !ok {
		return nil, false
	}

	// Check for expiration. If expired, remove it.
	if !e.expires.IsZero() && c.clock.Now().After(e.expires) {
		c.removeEntry(e)
		return nil, false
	}

	// Mark as recently used.
	c.lru.MoveToFront(e.element)
	return e.item, true
}

// evictIfNeeded evicts items until the cache is within capacity.
// It first purges expired items, then evicts items by LRU policy.
func (c *lruCache) evictIfNeeded() {
	// Pass 1: Purge all expired items. This is fast thanks to the heap.
	now := c.clock.Now()
	for c.ttlHeap.Len() > 0 {
		e := c.ttlHeap[0]
		// Since the heap is ordered, we can stop when we find a non-expired item.
		if e.expires.After(now) {
			break
		}
		// This item is expired, remove it.
		c.removeEntry(e)
	}

	// Pass 2: If still over capacity, evict by LRU.
	for c.capacity > 0 && c.usage > c.capacity {
		if elem := c.lru.Back(); elem != nil {
			c.removeEntry(elem.Value.(*entry))
		}
	}
}

// removeEntry is a helper to remove an entry from all internal data structures.
func (c *lruCache) removeEntry(e *entry) {
	// Remove from the main map.
	delete(c.data, e.key)

	// Remove from the LRU list.
	c.lru.Remove(e.element)

	// Remove from the TTL heap if it's in there.
	if e.index != -1 {
		heap.Remove(&c.ttlHeap, e.index)
	}

	// Adjust usage.
	c.usage -= e.size
}

// A priorityQueue implements heap.Interface and holds entries.
type priorityQueue []*entry

func (pq priorityQueue) Len() int { return len(pq) }

// Less orders by expiration time. For items with the same time, order is not guaranteed.
func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].expires.Before(pq[j].expires)
}

// Swap swaps elements and updates their heap index.
func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

// Push adds an entry to the heap.
func (pq *priorityQueue) Push(x any) {
	n := len(*pq)
	e := x.(*entry)
	e.index = n
	*pq = append(*pq, e)
}

// Pop removes an entry from the heap.
func (pq *priorityQueue) Pop() any {
	old := *pq
	n := len(old)
	e := old[n-1]
	old[n-1] = nil // avoid memory leak
	e.index = -1   // for safety
	*pq = old[0 : n-1]
	return e
}
