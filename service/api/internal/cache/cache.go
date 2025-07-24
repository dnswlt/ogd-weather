package cache

import (
	"container/list"
	"sync"
)

// entry represents a cached item
type entry struct {
	key     string
	item    any
	size    int           // "weight" of this entry
	element *list.Element // pointer to its LRU position
}

// Cache implements an LRU cache with total capacity limit (in arbitrary units, e.g. bytes)
type Cache struct {
	mu       sync.Mutex
	capacity int // total allowed capacity
	usage    int // current sum of all entry sizes

	data map[string]*entry
	lru  *list.List // most recently used at front
}

// New creates a capacity-limited LRU cache.
// capacity <= 0 means unbounded.
func New(capacity int) *Cache {
	return &Cache{
		capacity: capacity,
		data:     make(map[string]*entry),
		lru:      list.New(),
	}
}

// Put inserts or updates an item with a given size.
// Size should be the "weight" (e.g. response Content-Length).
func (c *Cache) Put(key string, item any, size int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// If already exists, update size & move to front
	if e, ok := c.data[key]; ok {
		// adjust current usage (remove old size, add new)
		c.usage -= e.size
		e.item = item
		e.size = size
		c.usage += size
		c.lru.MoveToFront(e.element)

		// evict if needed
		c.evictIfNeeded()
		return
	}

	// Create new entry
	e := &entry{key: key, item: item, size: size}
	elem := c.lru.PushFront(key) // list holds only the key
	e.element = elem
	c.data[key] = e
	c.usage += size

	// Evict until within capacity
	c.evictIfNeeded()
}

// Get retrieves an item and marks it as recently used
func (c *Cache) Get(key string) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	e, ok := c.data[key]
	if !ok {
		return nil, false
	}
	// Move to front (most recently used)
	c.lru.MoveToFront(e.element)
	return e.item, true
}

// evictIfNeeded evicts from the back until within maxCapacity
func (c *Cache) evictIfNeeded() {
	if c.capacity <= 0 {
		return // unbounded
	}
	for c.usage > c.capacity {
		c.evictOldest()
	}
}

// evictOldest removes the least recently used entry
func (c *Cache) evictOldest() {
	oldestElem := c.lru.Back()
	if oldestElem == nil {
		return
	}
	oldestKey := oldestElem.Value.(string)

	// Remove from map & list
	if e, ok := c.data[oldestKey]; ok {
		c.usage -= e.size
		delete(c.data, oldestKey)
	}
	c.lru.Remove(oldestElem)
}

func (c *Cache) Capacity() int {
	return c.capacity
}

// Usage returns the current total size usage
func (c *Cache) Usage() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

// Len returns number of items currently in cache
func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.data)
}
