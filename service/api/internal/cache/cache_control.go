package cache

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

type CachePolicy int

const (
	CachePolicyNone    CachePolicy = iota // No info, up to caller
	CachePolicyNoCache                    // Explicitly uncacheable
	CachePolicyTTL                        // Cacheable with TTL
)

func (p CachePolicy) String() string {
	switch p {
	case CachePolicyNoCache:
		return "NoCache"
	case CachePolicyTTL:
		return "TTL"
	default:
		return "None"
	}
}

// ComputeCachePolicy inspects HTTP caching headers and determines how a response should be cached.
//
// It evaluates the headers in this order:
//
//  1. If Cache-Control contains "no-cache", "no-store", or "must-revalidate",
//     it returns CachePolicyNoCache (explicitly uncacheable).
//
//  2. If Cache-Control contains "max-age=<seconds>", it returns CachePolicyTTL
//     with the corresponding duration.
//
//  3. If no max-age is present, but an Expires header exists, it returns:
//     - CachePolicyTTL if the Expires date is in the future
//     - CachePolicyNoCache if the Expires date is in the past
//
//  4. If none of the above headers are present, it returns CachePolicyNone,
//     meaning no explicit caching information was provided.
//
// The function always prefers Cache-Control over Expires, per RFC 7234.
func ComputeCachePolicy(h http.Header, now time.Time) (CachePolicy, time.Duration) {
	cc := strings.ToLower(h.Get("Cache-Control"))

	// 1. Explicit NO CACHE
	if strings.Contains(cc, "no-cache") || strings.Contains(cc, "no-store") {
		return CachePolicyNoCache, 0
	}

	// 2. Cache-Control max-age
	if i := strings.Index(cc, "max-age="); i != -1 {
		rest := cc[i+len("max-age="):]
		if j := strings.IndexAny(rest, ",; "); j != -1 {
			rest = rest[:j]
		}
		if v, err := strconv.Atoi(rest); err == nil {
			return CachePolicyTTL, time.Duration(v) * time.Second
		}
	}

	// 3. Expires header fallback
	if exp := h.Get("Expires"); exp != "" {
		if t, err := http.ParseTime(exp); err == nil {
			ttl := t.Sub(now)
			if ttl > 0 {
				return CachePolicyTTL, ttl
			}
			return CachePolicyNoCache, 0 // already expired → explicitly uncacheable
		}
	}

	// 4. No info → caller decides
	return CachePolicyNone, 0
}
