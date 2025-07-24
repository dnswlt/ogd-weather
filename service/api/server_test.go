package api

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestCanonicalURL(t *testing.T) {
	tests := []struct {
		raw      string
		expected string
	}{
		{
			raw:      "https://example.com/some?y=2&x=1",
			expected: "https://example.com/some?x=1&y=2",
		},
		{
			raw:      "https://example.com/some?x=1&y=2",
			expected: "https://example.com/some?x=1&y=2",
		},
		{
			raw:      "https://example.com/some?b=2&a=1&b=1",
			expected: "https://example.com/some?a=1&b=1&b=2", // repeated values sorted
		},
		{
			raw:      "https://example.com:9090/some",
			expected: "https://example.com:9090/some",
		},
		{
			raw:      "/some/path?foo=bar&exe=bat",
			expected: "/some/path?exe=bat&foo=bar",
		},
	}

	for _, tt := range tests {
		got, err := canonicalURL(tt.raw)
		if err != nil {
			t.Fatalf("CanonicalURL(%q) unexpected error: %v", tt.raw, err)
		}
		if diff := cmp.Diff(tt.expected, got); diff != "" {
			t.Errorf("CanonicalURL(%q) mismatch (-want +got):\n%s", tt.raw, diff)
		}
	}
}
