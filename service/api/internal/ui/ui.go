package ui

import (
	"fmt"
	"strings"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
)

var (
	zurichLoc = mustLoadLocation("Europe/Zurich")
)

type Option struct {
	Value       string
	DisplayName string
	IsSelected  bool
}

func (o *Option) Selected() string {
	if o.IsSelected {
		return "selected"
	}
	return ""
}

func Periods(selected string) []Option {
	opts := []Option{
		{"1", "January", false},
		{"2", "February", false},
		{"3", "March", false},
		{"4", "April", false},
		{"5", "May", false},
		{"6", "June", false},
		{"7", "July", false},
		{"8", "August", false},
		{"9", "September", false},
		{"10", "October", false},
		{"11", "November", false},
		{"12", "December", false},
		{"spring", "Spring (Mar-May)", false},
		{"summer", "Summer (Jun-Aug)", false},
		{"autumn", "Autumn (Sep-Nov)", false},
		{"winter", "Winter (Dec-Feb)", false},
		{"all", "Whole Year", false},
	}
	for i := range opts {
		if selected == opts[i].Value {
			opts[i].IsSelected = true
			break
		}
	}
	return opts
}

func mustLoadLocation(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	if err != nil {
		panic(err) // or handle gracefully
	}
	return loc
}

func DateFmt(format string, date any) (string, error) {
	// Translate format string using Ymd to Go format string
	var sb strings.Builder
	for _, r := range format {
		switch r {
		case 'Y':
			sb.WriteString("2006")
		case 'm':
			sb.WriteString("01")
		case 'd':
			sb.WriteString("02")
		case 'H':
			sb.WriteString("15")
		case 'M':
			sb.WriteString("04")
		case 'S':
			sb.WriteString("05")
		case 'J':
			sb.WriteString("Jan")
		default:
			// Append all other characters unchanged.
			sb.WriteRune(r)
		}
	}
	var t time.Time
	switch d := date.(type) {
	case *types.Date:
		if d == nil {
			return "", nil
		}
		t = d.Time
	case types.Date:
		t = d.Time
	case time.Time:
		t = d
	default:
		return "", fmt.Errorf("unsupported type for DateFmt: %T", date)
	}
	return t.In(zurichLoc).Format(sb.String()), nil
}
