package types

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestNullFloat64_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		input       string
		wantValue   float64
		wantHas     bool
		expectError bool
	}{
		{`null`, 0, false, false},
		{``, 0, false, false},
		{`1.23`, 1.23, true, false},
		{`0`, 0, true, false},
		{`-5.5`, -5.5, true, false},
		{`"bad"`, 0, false, true},
	}

	for _, tt := range tests {
		var f NullFloat64
		err := f.UnmarshalJSON([]byte(tt.input))
		if (err != nil) != tt.expectError {
			t.Errorf("UnmarshalJSON(%q) error = %v, want error? %v", tt.input, err, tt.expectError)
			continue
		}
		if f.Value != tt.wantValue || f.HasValue != tt.wantHas {
			t.Errorf("UnmarshalJSON(%q) = %+v, want value=%v has=%v", tt.input, f, tt.wantValue, tt.wantHas)
		}
	}
}

func TestNullFloat64_SliceUnmarshalJSON(t *testing.T) {
	tests := []struct {
		input     string
		wantValue []NullFloat64
	}{
		{
			input: `[null, 1.3]`,
			wantValue: []NullFloat64{
				{0, false},
				{1.3, true},
			},
		},
	}

	for _, tt := range tests {
		var fs []NullFloat64
		err := json.Unmarshal([]byte(tt.input), &fs)
		if err != nil {
			t.Errorf("UnmarshalJSON(%q) error = %v", tt.input, err)
			continue
		}
		if diff := cmp.Diff(tt.wantValue, fs); diff != "" {
			t.Errorf("Unmarshal diff (-want, +got): %v", diff)
		}
	}
}

func TestNullFloat64_MarshalJSON(t *testing.T) {
	tests := []struct {
		f    NullFloat64
		want string
	}{
		{NullFloat64{Value: 1.23, HasValue: true}, `1.23`},
		{NullFloat64{Value: 0, HasValue: true}, `0`},
		{NullFloat64{HasValue: false}, `null`},
		{NullFloat64{Value: math.NaN(), HasValue: true}, `null`},
		{NullFloat64{Value: math.Inf(1), HasValue: true}, `null`},
		{NullFloat64{Value: math.Inf(-1), HasValue: true}, `null`},
	}

	for _, tt := range tests {
		data, err := tt.f.MarshalJSON()
		if err != nil {
			t.Errorf("MarshalJSON(%+v) error: %v", tt.f, err)
			continue
		}
		if string(data) != tt.want {
			t.Errorf("MarshalJSON(%+v) = %s, want %s", tt.f, data, tt.want)
		}
	}
}

func TestNullFloat64_String(t *testing.T) {
	tests := []struct {
		f    NullFloat64
		want string
	}{
		{NullFloat64{Value: 1.23, HasValue: true}, "1.23"},
		{NullFloat64{Value: 0, HasValue: true}, "0"},
		{NullFloat64{HasValue: false}, "null"},
	}

	for _, tt := range tests {
		got := tt.f.String()
		if got != tt.want {
			t.Errorf("String(%+v) = %q, want %q", tt.f, got, tt.want)
		}
	}
}

func TestNullDate_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		input       string
		wantValue   Date
		wantHas     bool
		expectError bool
	}{
		{`null`, Date{}, false, false},
		{`""`, Date{}, false, true}, // invalid empty string
		{
			`"2024-12-25"`,
			Date{time.Date(2024, 12, 25, 0, 0, 0, 0, time.UTC)},
			true, false,
		},
		{`"not-a-date"`, Date{}, false, true},
	}

	for _, tt := range tests {
		var d NullDate
		err := d.UnmarshalJSON([]byte(tt.input))
		if (err != nil) != tt.expectError {
			t.Errorf("UnmarshalJSON(%q) error = %v, want error? %v", tt.input, err, tt.expectError)
			continue
		}
		if d.HasValue != tt.wantHas || (d.HasValue && !d.Value.Equal(tt.wantValue.Time)) {
			t.Errorf("UnmarshalJSON(%q) = %+v, want value=%v has=%v", tt.input, d, tt.wantValue, tt.wantHas)
		}
	}
}

func TestNullDate_MarshalJSON(t *testing.T) {
	tests := []struct {
		d    NullDate
		want string
	}{
		{
			NullDate{HasValue: false},
			`null`,
		},
		{
			NullDate{
				HasValue: true,
				Value:    Date{time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)},
			},
			`"2025-01-01"`,
		},
	}

	for _, tt := range tests {
		data, err := json.Marshal(tt.d)
		if err != nil {
			t.Errorf("MarshalJSON(%+v) error: %v", tt.d, err)
			continue
		}
		if string(data) != tt.want {
			t.Errorf("MarshalJSON(%+v) = %q, want %q", tt.d, string(data), tt.want)
		}
	}
}
