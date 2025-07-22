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
