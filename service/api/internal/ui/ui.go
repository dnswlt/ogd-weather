package ui

import (
	"fmt"
	"html/template"
	"math"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
)

var (
	zurichLoc = mustLoadLocation("Europe/Zurich")

	AllFuncs = map[string]any{
		"datefmt":            DateFmt,
		"wgs84tosvg":         WGS84ToSVG,
		"min2hours":          MinutesToHours,
		"ms2kmh":             MetersPerSecondToKilometersPerHour,
		"float":              PrintfFloat64,
		"floatOr":            PrintfFloat64OrDefault,
		"floatFormat":        FloatFormat,
		"vegaSpecHTMLID":     VegaSpecHTMLID,
		"first":              PrefixSlice,
		"stationVarsPercent": StationVarsPercent,
	}
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
		{"all", "Whole Year", false},
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
		{"winter", "Winter (Jan-Feb,Dec)", false},
	}
	if selected == "" {
		selected = "all" // By default, select "Whole Year"
	}
	for i := range opts {
		if selected == opts[i].Value {
			opts[i].IsSelected = true
			break
		}
	}
	return opts
}

func FloatFormat(f float64) string {
	if math.Abs(f) > 100 {
		return "%.0f"
	}
	return "%.1f"
}

func PrintfFloat64(format string, f types.NullFloat64) string {
	if !f.HasValue {
		return ""
	}
	return fmt.Sprintf(format, f.Value)
}

func PrintfFloat64OrDefault(format string, f types.NullFloat64, defaultVal string) string {
	if !f.HasValue {
		return defaultVal
	}
	return fmt.Sprintf(format, f.Value)
}

func mustLoadLocation(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	if err != nil {
		panic(err) // or handle gracefully
	}
	return loc
}

func MinutesToHours(minutes float64) string {
	sign := ""
	if minutes < 0 {
		sign = "-"
		minutes = -minutes
	}

	totalMinutes := math.Round(minutes)
	h := int(totalMinutes) / 60
	m := int(totalMinutes) % 60

	return fmt.Sprintf("%s%d:%02d", sign, h, m)
}

func MetersPerSecondToKilometersPerHour(mps float64) float64 {
	return 3.6 * mps
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
		case 'a':
			sb.WriteString("2")
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
	case types.NullDate:
		if !d.HasValue {
			return "", nil
		}
		t = d.Value.Time
	case time.Time:
		t = d
	default:
		return "", fmt.Errorf("unsupported type for DateFmt: %T", date)
	}
	return t.In(zurichLoc).Format(sb.String()), nil
}

type SVGPoint struct {
	X float64
	Y float64
}

type affineTransform struct {
	a float64
	b float64
	c float64
	d float64
	e float64
	f float64
}

func (at affineTransform) Apply(p SVGPoint) SVGPoint {
	x := at.a*p.X + at.b*p.Y + at.c
	y := at.d*p.X + at.e*p.Y + at.f
	return SVGPoint{x, y}
}

var (
	affineTransforms = map[string]affineTransform{
		// See tools/svgmap/README.md
		"suisse": {
			a: 222.7456099377238,
			b: -2.6828866160026887,
			c: -1167.4821813828262,
			d: -0.9799860475869895,
			e: -325.44941520771715,
			f: 15580.77103793717,
		},
	}
)

func WGS84ToSVG(transform string, lon, lat float64) (SVGPoint, error) {
	at, ok := affineTransforms[transform]
	if !ok {
		return SVGPoint{}, fmt.Errorf("invalid affine transform %q", transform)
	}
	return at.Apply(SVGPoint{lon, lat}), nil
}

type NavBar []*NavBarItem

type NavBarItem struct {
	path        string
	queryParams map[string]string
	params      []string
	Title       string
	Active      bool
}

func (n *NavBarItem) URI() string {
	var u url.URL
	u.Path = n.path
	q := make(url.Values)
	for k, v := range n.queryParams {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func (n *NavBarItem) Params(params ...string) *NavBarItem {
	n.params = params
	return n
}

func (n *NavBarItem) ParamsList() string {
	return strings.Join(n.params, ",")
}

type NavQueryParam struct {
	Key   string
	Value string
}

func NavItem(path, title string) *NavBarItem {
	return &NavBarItem{
		path:        path,
		Title:       title,
		queryParams: make(map[string]string),
	}
}

func NewNavBar(items ...*NavBarItem) NavBar {
	return items
}

func (ns NavBar) SetActive(activePath string) NavBar {
	activePath = strings.TrimSuffix(activePath, "/")
	for _, n := range ns {
		if activePath == strings.TrimSuffix(n.path, "/") {
			n.Active = true
			break
		}
	}
	return ns
}

func (ns NavBar) SetParam(key, value string) NavBar {
	for _, n := range ns {
		if slices.Contains(n.params, key) {
			n.queryParams[key] = value
		}
	}
	return ns
}

func (ns NavBar) SetParams(q url.Values) NavBar {
	for k := range q {
		if v := q.Get(k); v != "" {
			ns = ns.SetParam(k, q.Get(k))
		}
	}
	return ns
}

func FormatBytesIEC(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n >= unit*div && exp < 6 {
		div *= unit
		exp++
	}
	value := float64(n) / float64(div)
	return fmt.Sprintf("%.2f %ciB", value, "KMGTPE"[exp])
}

// VegaSpecHTMLID translates a Vega spec name into its corresponding
// HTML element ID.
// For now, the only character allowed in chart types that is not
// convenient to use in element IDs is the ":" character, so
// replace it by "-".
func VegaSpecHTMLID(specName string) string {
	return strings.ReplaceAll(specName, ":", "-")
}

func PrefixSlice(k int, items any) (any, error) {
	if items == nil {
		return nil, nil
	}
	v := reflect.ValueOf(items)
	if v.Kind() != reflect.Slice {
		return nil, fmt.Errorf("PrefixSlice: items is not a slice, but %T", items)
	}
	n := v.Len()
	if k < 0 {
		return nil, fmt.Errorf("PrefixSlice: k < 0")
	}
	if k > n {
		k = n
	}
	return v.Slice(0, k).Interface(), nil
}

// MapColors returns a map from
func MapColors() map[string]string {
	return map[string]string{
		// Light gray palette
		// "Background1": "#F5F5F5",
		// "Background2": "#E0E0E0",
		// "Background3": "#CCCCCC",
		// "Background4": "#BDBDBD",
		// "Background5": "#A9A9A9",

		// Alternative palette: cool blue
		"Background1": "#F0F4F8",
		"Background2": "#D9E2EC",
		"Background3": "#BCCCDC",
		"Background4": "#9FB3C8",
		"Background5": "#829AB1",
	}
}

// StationVarsPercent creates the style= attribute string that defines
// --station{i} variables for the widths of the station comparison table
// bars. Example:
// "--station1: 49.1%; --station2: 52.3%; --station3: 74.3%; --station4: 74.3%;"
func StationVarsPercent(row *types.StationComparisonRow) template.CSS {
	if len(row.Values) == 0 {
		return ""
	}
	minVal := math.Inf(1)
	if row.LowerBound.HasValue {
		minVal = row.LowerBound.Value
	}
	maxVal := math.Inf(-1)
	if row.UpperBound.HasValue {
		maxVal = row.UpperBound.Value
	}
	for _, v := range row.Values {
		if !v.HasValue {
			continue
		}
		vv := v.Value
		if vv > maxVal {
			maxVal = vv
		}
		if vv < minVal {
			minVal = vv
		}
	}
	var span float64
	if math.IsInf(minVal, 1) || minVal == maxVal {
		span = 0
	} else if !row.LowerBound.HasValue || minVal < row.LowerBound.Value {
		// minVal was pushed down by an actual value.
		// To avoid having a 0% bar width for that value,
		// Push minVal down further so that the old minVal is at 10%.
		minVal = maxVal - (maxVal-minVal)/0.9
		span = maxVal - minVal
	} else {
		span = maxVal - minVal
	}

	vars := make([]string, len(row.Values))
	for i, v := range row.Values {
		var pct float64
		if !v.HasValue || v.Value == 0 && span == 0 {
			// No value or all values are zero.
			pct = 0
		} else if span == 0 {
			// All values are identical => set to 100%.
			pct = 100
		} else {
			pct = (v.Value - minVal) / span * 100
		}
		vars[i] = fmt.Sprintf("--station%d: %.1f%%;", i, pct)
	}
	return template.CSS(strings.Join(vars, " "))
}
