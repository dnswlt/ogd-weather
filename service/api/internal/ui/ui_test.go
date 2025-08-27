package ui

import (
	"html/template"
	"strings"
	"testing"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
)

func TestMinutesToHours(t *testing.T) {
	tests := []struct {
		minutes float64
		want    string
	}{
		// Whole hours
		{0, "0:00"},
		{60, "1:00"},
		{120, "2:00"},

		// Regular minutes
		{15, "0:15"},
		{59, "0:59"},
		{61, "1:01"},
		{135, "2:15"},

		// Rounded minutes
		{90.4, "1:30"},  // round down
		{90.5, "1:31"},  // round up
		{59.6, "1:00"},  // rolls to next hour
		{119.6, "2:00"}, // correctly carries over

		// Negative
		{-30, "-0:30"},
		{-59.6, "-1:00"},
		{-119.6, "-2:00"},
	}

	for _, tt := range tests {
		got := MinutesToHours(tt.minutes)
		if got != tt.want {
			t.Errorf("MinutesToHours(%v) = %q, want %q", tt.minutes, got, tt.want)
		}
	}
}

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

func TestFormatBytesIEC(t *testing.T) {
	tests := []struct {
		input    int64
		expected string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1023, "1023 B"},
		{1024, "1.00 KiB"},
		{1536, "1.50 KiB"},
		{1048576, "1.00 MiB"},
		{3145728, "3.00 MiB"},
		{1073741824, "1.00 GiB"},
		{1234567890, "1.15 GiB"},
		{1099511627776, "1.00 TiB"},
	}

	for _, tt := range tests {
		got := FormatBytesIEC(tt.input)
		if got != tt.expected {
			t.Errorf("formatBytesIEC(%d) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestStationVarsPercent(t *testing.T) {
	// Helper to create a NullFloat64 struct.
	v := func(val float64) types.NullFloat64 {
		return types.NullFloat64{HasValue: true, Value: val}
	}
	// Helper for an empty/non-existent value.
	noV := types.NullFloat64{HasValue: false}

	// Define all test cases in a table.
	testCases := []struct {
		name string
		row  *types.StationComparisonRow
		want template.CSS
	}{
		{
			name: "Normal case, no bounds",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{v(10), v(55), v(100)},
			},
			// minVal becomes 10, maxVal 100. Span is 90.
			// To avoid 0% bar, original span (90) is treated as 90% of new span.
			// New span = 90 / 0.9 = 100.
			// New minVal = 100 (maxVal) - 100 (newSpan) = 0.
			// Pcts: (10-0)/100*100=10%, (55-0)/100*100=55%, (100-0)/100*100=100%
			want: "--station0: 10.0%; --station1: 55.0%; --station2: 100.0%;",
		},
		{
			name: "Empty values slice",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{},
			},
			want: "",
		},
		{
			name: "All values are identical",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{v(50), v(50), v(50)},
			},
			// span is 0, so all should be 100%
			want: "--station0: 100.0%; --station1: 100.0%; --station2: 100.0%;",
		},
		{
			name: "With fixed lower bound",
			row: &types.StationComparisonRow{
				Values:     []types.NullFloat64{v(10), v(55), v(100)},
				LowerBound: v(0),
			},
			// minVal=0, maxVal=100, span=100. No 0% adjustment needed.
			// Pcts: (10-0)/100*100=10%, (55-0)/100*100=55%, (100-0)/100*100=100%
			want: "--station0: 10.0%; --station1: 55.0%; --station2: 100.0%;",
		},
		{
			name: "With fixed upper bound",
			row: &types.StationComparisonRow{
				Values:     []types.NullFloat64{v(10), v(50)},
				UpperBound: v(100),
			},
			// minVal=10, maxVal=100, span=90. 0% adjustment is triggered.
			// New span = 90 / 0.9 = 100.
			// New minVal = 100 - 100 = 0.
			// Pcts: (10-0)/100*100=10%, (50-0)/100*100=50%
			want: "--station0: 10.0%; --station1: 50.0%;",
		},
		{
			name: "With both bounds",
			row: &types.StationComparisonRow{
				Values:     []types.NullFloat64{v(20), v(80)},
				LowerBound: v(0),
				UpperBound: v(100),
			},
			// minVal=0, maxVal=100, span=100.
			// Pcts: (20-0)/100*100=20%, (80-0)/100*100=80%
			want: "--station0: 20.0%; --station1: 80.0%;",
		},
		{
			name: "Values outside fixed bounds",
			row: &types.StationComparisonRow{
				Values:     []types.NullFloat64{v(-10), v(110)},
				LowerBound: v(0),
				UpperBound: v(100),
			},
			// Actual value should push minVal down so its value is at
			// 10%.
			want: "--station0: 10.0%; --station1: 100.0%;",
		},
		{
			name: "With missing/nil values",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{v(10), noV, v(100)},
			},
			// Same as normal case, but middle value is 0%.
			want: "--station0: 10.0%; --station1: 0.0%; --station2: 100.0%;",
		},
		{
			name: "All missing values",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{noV, noV, noV},
			},
			// No min/max found, span is 0, all pcts are 0.
			want: "--station0: 0.0%; --station1: 0.0%; --station2: 0.0%;",
		},
		{
			name: "Single value, no bounds",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{v(42)},
			},
			// minVal==maxVal, span is 0, so pct is 100%.
			want: "--station0: 100.0%;",
		},
		{
			name: "Single value with bounds",
			row: &types.StationComparisonRow{
				Values:     []types.NullFloat64{v(50)},
				LowerBound: v(0),
				UpperBound: v(100),
			},
			// minVal=0, maxVal=100, span=100.
			// Pct: (50-0)/100*100=50%
			want: "--station0: 50.0%;",
		},
		{
			name: "Negative values",
			row: &types.StationComparisonRow{
				Values: []types.NullFloat64{v(-50), v(-10), v(0)},
			},
			// minVal=-50, maxVal=0, span=50. 0% adjustment triggered.
			// New span = 50 / 0.9 = 55.55...
			// New minVal = 0 - 55.55... = -55.55...
			// Pcts: (-50 - -55.55)/55.55*100=10%, (-10 - -55.55)/55.55*100=82%, (0 - -55.55)/55.55*100=100%
			want: "--station0: 10.0%; --station1: 82.0%; --station2: 100.0%;",
		},
	}

	// Run the test cases.
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := StationVarsPercent(tc.row)
			if got != tc.want {
				t.Errorf("StationVarsPercent()\n got: %q\nwant: %q", got, tc.want)
			}
		})
	}
}
