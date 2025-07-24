package ui

import (
	"strings"
	"testing"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
)

func TestDateFmt(t *testing.T) {
	// Define a sample date for testing using your actual types.Date struct
	sampleTime := time.Date(2023, time.January, 15, 0, 0, 0, 0, time.UTC)
	sampleDate := types.Date{Time: sampleTime} // Initialize with the embedded time.Time field
	sampleDatePtr := &types.Date{Time: sampleTime}

	tests := []struct {
		name        string
		format      string
		date        any
		expected    string
		expectError bool
	}{
		{
			name:        "Format Ymd with types.Date",
			format:      "Ymd",
			date:        sampleDate,
			expected:    "20230115",
			expectError: false,
		},
		{
			name:        "Format including time with time.Time in Europe/Zurich",
			format:      "Y-m-d H:M:S",
			date:        sampleTime.Add(18*time.Hour + 30*time.Minute + 50*time.Second),
			expected:    "2023-01-15 19:30:50",
			expectError: false,
		},
		{
			name:        "Format Ymd with types.Date",
			format:      "Y",
			date:        sampleDate,
			expected:    "2023",
			expectError: false,
		},
		{
			name:        "Format Y-m-d with types.Date",
			format:      "Y-m-d",
			date:        sampleDate,
			expected:    "2023-01-15",
			expectError: false,
		},
		{
			name:        "Format m/d/Y with types.Date",
			format:      "m/d/Y",
			date:        sampleDate,
			expected:    "01/15/2023",
			expectError: false,
		},
		{
			name:        "Format Ymd with *types.Date",
			format:      "Ymd",
			date:        sampleDatePtr,
			expected:    "20230115",
			expectError: false,
		},
		{
			name:        "Format Y-m-d with *types.Date",
			format:      "Y-m-d",
			date:        sampleDatePtr,
			expected:    "2023-01-15",
			expectError: false,
		},
		{
			name:        "Format m/d/Y with *types.Date",
			format:      "m/d/Y",
			date:        sampleDatePtr,
			expected:    "01/15/2023",
			expectError: false,
		},
		{
			name:        "Nil *types.Date",
			format:      "Ymd",
			date:        (*types.Date)(nil), // Explicitly pass a nil pointer to types.Date
			expected:    "",
			expectError: false, // Expect no error, but an empty string
		},
		{
			name:        "Unsupported type (string)",
			format:      "Ymd",
			date:        "2023-01-15", // A string, which is not supported
			expected:    "",
			expectError: true,
		},
		{
			name:        "Unsupported type (int)",
			format:      "Ymd",
			date:        12345, // An int, which is not supported
			expected:    "",
			expectError: true,
		},
		{
			name:        "Format with additional characters",
			format:      "Y-m-d XZ", // XZ should pass through unchanged
			date:        sampleDate,
			expected:    "2023-01-15 XZ",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DateFmt(tt.format, tt.date)

			if tt.expectError {
				if err == nil {
					t.Errorf("DateFmt() expected an error but got none")
				}
				// Optionally, check the error message if it's specific
				if err != nil && !strings.Contains(err.Error(), "unsupported type") {
					t.Errorf("DateFmt() expected 'unsupported type' error but got: %v", err)
				}
			} else {
				if err != nil {
					t.Errorf("DateFmt() unexpected error: %v", err)
				}
				if got != tt.expected {
					t.Errorf("DateFmt() got = %q, want %q", got, tt.expected)
				}
			}
		})
	}
}
