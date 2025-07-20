package ui

import (
	"fmt"
	"strings"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
)

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
		default:
			// Append all other characters unchanged.
			sb.WriteRune(r)
		}
	}
	goFormat := sb.String()
	switch d := date.(type) {
	case *types.Date:
		if d == nil {
			return "", nil
		}
		return d.Format(goFormat), nil
	case types.Date:
		return d.Format(goFormat), nil
	}
	return "", fmt.Errorf("unsupported type for DateFmt: %T", date)
}
