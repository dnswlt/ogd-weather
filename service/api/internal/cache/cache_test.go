package cache

import (
	"testing"
	"time"
)

type FakeClock struct {
	now time.Time
}

func (f *FakeClock) Now() time.Time {
	return f.now
}

func (f *FakeClock) Advance(d time.Duration) {
	f.now = f.now.Add(d)
}

func TestCacheEvictionForCapacity(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}

	keys := []string{"A", "B", "C"}
	c := NewWithClock(clock, len(keys))

	for _, k := range keys {
		c.Put(k, k, 1, 1*time.Minute)
	}

	for _, k := range keys {
		_, ok := c.Get(k)
		if !ok {
			t.Errorf("%s not in cache", k)
		}
	}

	// This should lead to eviction of A, the one least recently used.
	c.Put("D", "D", 1, 1*time.Minute)

	if _, ok := c.Get("A"); ok {
		t.Error("Expected A to be eviced from the cache")
	}

	if _, ok := c.Get("D"); !ok {
		t.Error("Expected D to be in the cache")
	}
}

func TestCacheEvictionForTTL(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10) // plenty of capacity

	// Put an item with a TTL of 1 minute
	c.Put("X", "valueX", 1, 1*time.Minute)

	// Should still be available before TTL
	if v, ok := c.Get("X"); !ok || v != "valueX" {
		t.Errorf("Expected X to be in cache before TTL expiry, got %v", v)
	}

	// Advance just before TTL expiry
	clock.Advance(59 * time.Second)
	if v, ok := c.Get("X"); !ok || v != "valueX" {
		t.Errorf("Expected X to still be in cache at 59s, got %v", v)
	}

	// Advance past TTL expiry
	clock.Advance(2 * time.Second) // now total 61s > 1 min

	if _, ok := c.Get("X"); ok {
		t.Error("Expected X to be evicted due to TTL expiry")
	}

	// Also test that after eviction, a fresh insert works
	c.Put("Y", "valueY", 1, 1*time.Minute)
	if v, ok := c.Get("Y"); !ok || v != "valueY" {
		t.Errorf("Expected Y to be inserted after X expired, got %v", v)
	}
}

func TestCacheHeapExpiryPurgeOnPut(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10) // plenty of capacity

	// Put two items: A expires sooner, B later
	c.Put("A", "valueA", 1, 1*time.Minute)
	c.Put("B", "valueB", 1, 10*time.Minute)

	// Access A so it becomes MRU -> B becomes the true LRU tail
	if _, ok := c.Get("A"); !ok {
		t.Fatal("Expected A to be in cache initially")
	}

	// Now LRU order: head=A, tail=B

	// Advance clock so that A is expired but B is still valid
	clock.Advance(2 * time.Minute)

	// Put C → triggers evictIfNeeded() → TTL heap should purge A
	c.Put("C", "valueC", 1, 5*time.Minute)

	// A should be removed due to TTL (even though it was MRU!)
	if _, ok := c.Get("A"); ok {
		t.Error("Expected A to be purged by TTL heap during Put, but it's still present")
	}

	// B should still remain because it has a long TTL
	if v, ok := c.Get("B"); !ok || v != "valueB" {
		t.Errorf("Expected B to remain in cache, got %v", v)
	}

	// C should now be present
	if v, ok := c.Get("C"); !ok || v != "valueC" {
		t.Errorf("Expected C to be in cache, got %v", v)
	}
}

func TestCacheTTLThenCapacityEviction(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10) // total capacity 10

	// --- Insert 3 small soon-to-expire items ---
	c.Put("E1", "exp1", 1, 1*time.Minute)
	c.Put("E2", "exp2", 1, 1*time.Minute)
	c.Put("E3", "exp3", 1, 1*time.Minute)

	// --- Insert 2 larger valid items ---
	c.Put("V1", "valid1", 3, 10*time.Minute)
	c.Put("V2", "valid2", 3, 10*time.Minute)

	// Access V1 so it becomes MRU, leaving V2 as true LRU tail
	if _, ok := c.Get("V1"); !ok {
		t.Fatal("Expected V1 in cache")
	}

	// Usage now: 1+1+1 + 3+3 = 9
	// Expiry soonest = E1
	// LRU tail = V2

	// Advance clock beyond E1-E3 TTL, but not enough to expire V1/V2
	clock.Advance(2 * time.Minute)

	// Now insert a large item (size 5), which will exceed capacity by 4
	c.Put("NewBig", "big!", 5, 10*time.Minute)

	// --- Verify expected outcome ---

	// All expired E* should be gone
	for _, k := range []string{"E1", "E2", "E3"} {
		if _, ok := c.Get(k); ok {
			t.Errorf("Expected %s to be purged by TTL before capacity eviction", k)
		}
	}

	// Only V2 should be evicted by LRU
	if _, ok := c.Get("V2"); ok {
		t.Error("Expected V2 to be evicted as the least recently used valid item")
	}

	// V1 should remain
	if v, ok := c.Get("V1"); !ok || v != "valid1" {
		t.Errorf("Expected V1 to remain, got %v", v)
	}

	// NewBig should be present
	if v, ok := c.Get("NewBig"); !ok || v != "big!" {
		t.Errorf("Expected NewBig to be inserted, got %v", v)
	}

	// Final usage check: should now be 3 (V1) + 5 (NewBig) = 8
	if u := c.Usage(); u != 8 {
		t.Errorf("Expected usage=8 after eviction, got %d", u)
	}
}

func TestCacheUpdateNoTTLToWithTTL(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10)

	// Insert item with no TTL (never expires)
	c.Put("X", "initial", 1, 0)

	// Should be present, no expiry
	if v, ok := c.Get("X"); !ok || v != "initial" {
		t.Fatalf("Expected X with no TTL, got %v", v)
	}

	// Now update the same item with a TTL of 1 minute
	c.Put("X", "withTTL", 1, 1*time.Minute)

	// Immediately after update, should reflect new value and still present
	if v, ok := c.Get("X"); !ok || v != "withTTL" {
		t.Fatalf("Expected X updated with TTL, got %v", v)
	}

	// Advance clock past TTL → should now expire
	clock.Advance(2 * time.Minute)

	// Trigger expiry cleanup (lazy via Get)
	if _, ok := c.Get("X"); ok {
		t.Error("Expected X to expire after adding TTL")
	}
}

func TestCacheUpdateTTLRemoved(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10)

	// Insert item with TTL of 1 minute
	c.Put("Y", "willExpire", 1, 1*time.Minute)

	// Should be present initially
	if v, ok := c.Get("Y"); !ok || v != "willExpire" {
		t.Fatalf("Expected Y to be initially present, got %v", v)
	}

	// Now update the same item with NO TTL → should never expire
	c.Put("Y", "noTTLanymore", 1, 0)

	// Advance time past original TTL
	clock.Advance(2 * time.Minute)

	// It should still remain, because TTL was removed
	if v, ok := c.Get("Y"); !ok || v != "noTTLanymore" {
		t.Errorf("Expected Y to remain after TTL removal, got %v", v)
	}

	// Advance even further to be sure
	clock.Advance(10 * time.Minute)
	if v, ok := c.Get("Y"); !ok || v != "noTTLanymore" {
		t.Errorf("Expected Y to still remain after long time, got %v", v)
	}
}

func TestCacheUpdateTTLReschedule(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10)

	// Insert three items with different TTLs
	c.Put("A", "valueA", 1, 2*time.Minute)  // expires soonest
	c.Put("B", "valueB", 1, 5*time.Minute)  // will be rescheduled
	c.Put("C", "valueC", 1, 10*time.Minute) // expires latest

	// Update B to have a much longer TTL (15 minutes)
	c.Put("B", "valueBnewTTL", 1, 15*time.Minute)

	// --- Advance clock beyond A's TTL ---
	clock.Advance(3 * time.Minute)

	// Trigger TTL-based purge by inserting a dummy item
	c.Put("dummy", "dummy", 1, 1*time.Minute)

	// A should be expired
	if _, ok := c.Get("A"); ok {
		t.Error("Expected A to be expired after 3 minutes")
	}

	// B should still be present because its TTL was extended
	if v, ok := c.Get("B"); !ok || v != "valueBnewTTL" {
		t.Errorf("Expected B to still be valid after TTL extension, got %v", v)
	}

	// C should still be valid
	if v, ok := c.Get("C"); !ok || v != "valueC" {
		t.Errorf("Expected C to remain valid, got %v", v)
	}

	// --- Advance clock beyond B's *new* TTL (15 minutes total) ---
	clock.Advance(13 * time.Minute) // total now 16m

	// Trigger TTL-based purge again
	c.Put("dummy2", "dummy2", 1, 1*time.Minute)

	// Now B should expire at new TTL
	if _, ok := c.Get("B"); ok {
		t.Error("Expected B to expire after its updated TTL (15 minutes)")
	}

	// C should still remain (10 + 16m total > 10min → should now be expired!)
	if _, ok := c.Get("C"); ok {
		t.Error("Expected C to be expired after 16 minutes")
	}
}

func TestCacheUpdateTTLNoOp(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 10)

	// Insert two items with different TTLs
	c.Put("M", "valueM", 1, 5*time.Minute)  // will be updated with same TTL
	c.Put("N", "valueN", 1, 10*time.Minute) // stays untouched

	// Update M with the SAME TTL (no-op on expiry time)
	c.Put("M", "valueMupdated", 1, 5*time.Minute)

	// Advance clock just before original TTL expires
	clock.Advance(4 * time.Minute)

	// Trigger TTL-based purge
	c.Put("dummy", "dummy", 1, 1*time.Minute)

	// Both M and N should still be present
	if v, ok := c.Get("M"); !ok || v != "valueMupdated" {
		t.Errorf("Expected M still valid before TTL, got %v", v)
	}
	if v, ok := c.Get("N"); !ok || v != "valueN" {
		t.Errorf("Expected N still valid, got %v", v)
	}

	// Advance clock beyond M's TTL
	clock.Advance(2 * time.Minute) // total 6m > 5m TTL

	// Trigger TTL purge again
	c.Put("dummy2", "dummy2", 1, 1*time.Minute)

	// M should now expire at original TTL
	if _, ok := c.Get("M"); ok {
		t.Error("Expected M to expire at original TTL after no-op update")
	}

	// N should still remain since it had longer TTL
	if v, ok := c.Get("N"); !ok || v != "valueN" {
		t.Errorf("Expected N still valid, got %v", v)
	}
}

func TestCacheBasicAPI(t *testing.T) {
	clock := &FakeClock{now: time.Date(2025, 6, 3, 12, 0, 0, 0, time.UTC)}
	c := NewWithClock(clock, 100) // arbitrary large capacity

	capacityVal := c.Capacity()
	if capacityVal != 100 {
		t.Errorf("Expected Capacity=100, got %d", capacityVal)
	}

	// Insert 3 items
	c.Put("A", "valueA", 5, 10*time.Minute) // size 5
	c.Put("B", "valueB", 3, 10*time.Minute) // size 3
	c.Put("C", "valueC", 2, 1*time.Minute)  // size 2

	// Should have 3 items
	if ln := c.Len(); ln != 3 {
		t.Errorf("Expected Len=3, got %d", ln)
	}

	// Usage should be 5+3+2 = 10
	if usage := c.Usage(); usage != 10 {
		t.Errorf("Expected Usage=10, got %d", usage)
	}

	// Advance clock to expire C only
	clock.Advance(2 * time.Minute)
	c.Put("triggerPurge", "dummy", 1, 1*time.Minute) // triggers TTL purge

	// Now only A & B + triggerPurge remain
	if ln := c.Len(); ln != 3 { // still 3 because purge removed 1 and we added 1
		t.Errorf("Expected Len=3 after purge+add, got %d", ln)
	}

	// Expected Usage: A(5)+B(3)+triggerPurge(1) = 9
	if usage := c.Usage(); usage != 9 {
		t.Errorf("Expected Usage=9 after purge+add, got %d", usage)
	}

	// Expire everything
	clock.Advance(20 * time.Minute)
	c.Put("finalPurge", "dummy", 1, 1*time.Minute)

	// Now only finalPurge should remain
	if ln := c.Len(); ln != 1 {
		t.Errorf("Expected Len=1 after final purge, got %d", ln)
	}
	if usage := c.Usage(); usage != 1 {
		t.Errorf("Expected Usage=1 after final purge, got %d", usage)
	}
}

func TestPutPanicsOnZeroOrNegativeSize(t *testing.T) {
	clock := &FakeClock{now: time.Now()}
	c := NewWithClock(clock, 10)

	cases := []struct {
		name string
		size int
	}{
		{"zero size", 0},
		{"negative size", -5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Expected panic for size=%d but got none", tc.size)
				}
			}()
			c.Put("bad", "value", tc.size, time.Minute)
		})
	}
}
