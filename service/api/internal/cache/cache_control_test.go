package cache

import (
	"net/http"
	"testing"
	"time"
)

func TestComputeCachePolicy(t *testing.T) {
	now := time.Date(2025, 7, 24, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name       string
		header     http.Header
		wantPolicy CachePolicy
		wantTTL    time.Duration
	}{
		{
			name: "Cache-Control max-age only",
			header: http.Header{
				"Cache-Control": []string{"public, max-age=3600, immutable"},
			},
			wantPolicy: CachePolicyTTL,
			wantTTL:    3600 * time.Second,
		},
		{
			name: "Cache-Control max-age with must-revalidate (still cacheable)",
			header: http.Header{
				"Cache-Control": []string{"public, max-age=120, must-revalidate"},
			},
			wantPolicy: CachePolicyTTL,
			wantTTL:    120 * time.Second,
		},
		{
			name: "Cache-Control explicit no-cache",
			header: http.Header{
				"Cache-Control": []string{"no-cache"},
			},
			wantPolicy: CachePolicyNoCache,
			wantTTL:    0,
		},
		{
			name: "Cache-Control no-store wins over max-age",
			header: http.Header{
				"Cache-Control": []string{"max-age=600, no-store"},
			},
			wantPolicy: CachePolicyNoCache, // no-store overrides max-age
			wantTTL:    0,
		},
		{
			name: "Expires only",
			header: http.Header{
				"Expires": []string{"Thu, 24 Jul 2025 13:00:00 GMT"}, // 1h later
			},
			wantPolicy: CachePolicyTTL,
			wantTTL:    1 * time.Hour,
		},
		{
			name: "Both Cache-Control and Expires (max-age wins)",
			header: http.Header{
				"Cache-Control": []string{"max-age=120"},
				"Expires":       []string{"Thu, 24 Jul 2025 18:00:00 GMT"}, // ignored
			},
			wantPolicy: CachePolicyTTL,
			wantTTL:    120 * time.Second,
		},
		{
			name: "Expired Expires header",
			header: http.Header{
				"Expires": []string{"Wed, 23 Jul 2025 12:00:00 GMT"},
			},
			wantPolicy: CachePolicyNoCache,
			wantTTL:    0,
		},
		{
			name: "No caching headers",
			header: http.Header{
				"Content-Type": []string{"text/plain"},
			},
			wantPolicy: CachePolicyNone,
			wantTTL:    0,
		},
		{
			name: "Malformed Expires",
			header: http.Header{
				"Expires": []string{"Not a date"},
			},
			wantPolicy: CachePolicyNone,
			wantTTL:    0,
		},
		{
			name: "Malformed Cache-Control max-age",
			header: http.Header{
				"Cache-Control": []string{"max-age=abc"},
			},
			wantPolicy: CachePolicyNone,
			wantTTL:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotPolicy, gotTTL := ComputeCachePolicy(tt.header, now)
			if gotPolicy != tt.wantPolicy {
				t.Errorf("policy mismatch: got %v, want %v", gotPolicy, tt.wantPolicy)
			}
			if gotTTL != tt.wantTTL {
				t.Errorf("ttl mismatch: got %v, want %v", gotTTL, tt.wantTTL)
			}
		})
	}
}
