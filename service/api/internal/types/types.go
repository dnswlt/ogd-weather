package types

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

type localizedFields struct {
	DE string `json:"de"`
	FR string `json:"fr"`
	IT string `json:"it"`
	EN string `json:"en"`
}

// LocalizedString is an immutable, localized string for DE/FR/IT/EN text.
type LocalizedString struct {
	f localizedFields
}

func NewLocalizedString(de, fr, it, en string) LocalizedString {
	return LocalizedString{
		f: localizedFields{DE: de, FR: fr, IT: it, EN: en},
	}
}

func (l LocalizedString) DE() string { return l.f.DE }
func (l LocalizedString) FR() string { return l.f.FR }
func (l LocalizedString) IT() string { return l.f.IT }
func (l LocalizedString) EN() string { return l.f.EN }

type NullFloat64 struct {
	Value    float64
	HasValue bool
}

type NullDate struct {
	Value    Date
	HasValue bool
}

type Station struct {
	Abbr                 string          `json:"abbr"`
	Name                 string          `json:"name"`
	Canton               string          `json:"canton"`
	Typ                  string          `json:"typ"`
	Exposition           LocalizedString `json:"exposition"`
	URL                  LocalizedString `json:"url"`
	HeightMASL           NullFloat64     `json:"height_masl"`
	CoordinatesWGS84Lat  NullFloat64     `json:"coordinates_wgs84_lat"`
	CoordinatesWGS84Lon  NullFloat64     `json:"coordinates_wgs84_lon"`
	TemperatureMinDate   NullDate        `json:"temperature_min_date"`
	TemperatureMaxDate   NullDate        `json:"temperature_max_date"`
	PrecipitationMinDate NullDate        `json:"precipitation_min_date"`
	PrecipitationMaxDate NullDate        `json:"precipitation_max_date"`
}

type StationsResponse struct {
	Stations []*Station `json:"stations"`
}

type StationStats struct {
	FirstDate          Date        `json:"first_date"`
	LastDate           Date        `json:"last_date"`
	Period             string      `json:"period"`
	AnnualTempIncrease NullFloat64 `json:"annual_temp_increase"`
	ColdestYear        int         `json:"coldest_year,omitempty"`
	ColdestYearTemp    NullFloat64 `json:"coldest_year_temp,omitempty"`
	WarmestYear        int         `json:"warmest_year,omitempty"`
	WarmestYearTemp    NullFloat64 `json:"warmest_year_temp,omitempty"`
	DriestYear         int         `json:"driest_year,omitempty"`
	WettestYear        int         `json:"wettest_year,omitempty"`
}

type StationSummary struct {
	Station *Station      `json:"station"`
	Stats   *StationStats `json:"stats"`
}

type StationSummaryResponse struct {
	Summary *StationSummary `json:"summary,omitempty"`
}

type VariableStats struct {
	MinValue          float64 `json:"min_value"`
	MinValueDate      Date    `json:"min_value_date"`
	MeanValue         float64 `json:"mean_value"`
	MaxValue          float64 `json:"max_value"`
	MaxValueDate      Date    `json:"max_value_date"`
	SourceGranularity string  `json:"source_granularity"`
	ValueCount        int     `json:"value_count"`
}

type StationPeriodStats struct {
	StartDate     Date                      `json:"start_date"`
	EndDate       Date                      `json:"end_date"`
	VariableStats map[string]*VariableStats `json:"variable_stats"`
	templateVars  map[string]*VariableStats
}

// A helper for HTML templates: Access variables in CamelCase.
func (s *StationPeriodStats) Vars() map[string]*VariableStats {
	if s.templateVars == nil {
		s.templateVars = make(map[string]*VariableStats)
		for k, v := range s.VariableStats {
			s.templateVars[snakeToCamelCase(k)] = v
		}
	}
	return s.templateVars
}

type StationInfo struct {
	Station            *Station            `json:"station"`
	Ref1991To2020Stats *StationPeriodStats `json:"ref_1991_2020_stats"`
}

type StationInfoResponse struct {
	Info *StationInfo `json:"info"`
}

type VegaSpecResponse struct {
	VegaSpec map[string]any `json:"vega_spec"`
}

type MeasurementsRow struct {
	ReferenceTimestamp time.Time `json:"reference_timestamp"`
	Measurements       []float64 `json:"measurements"`
}

type ColumnInfo struct {
	Name        string `json:"name"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
	Dtype       string `json:"dtype"`
}

type StationMeasurementsData struct {
	StationAbbr string             `json:"station_abbr"`
	Rows        []*MeasurementsRow `json:"rows"`
	Columns     []*ColumnInfo      `json:"columns"`
}

type StationMeasurementsResponse struct {
	Data *StationMeasurementsData `json:"data"`
}

// MinDate returns the smaller of TemperatureMinDate and PrecipitationMinDate.
// It is used in JavaScript to get the lowest possible year to select.
func (s *Station) MinDate() NullDate {
	var m NullDate
	if s.TemperatureMinDate.HasValue {
		m = s.TemperatureMinDate
	}
	if s.PrecipitationMinDate.HasValue {
		if !m.HasValue || m.Value.After(s.PrecipitationMinDate.Value.Time) {
			m = s.PrecipitationMinDate
		}
	}
	return m
}

// MaxDate returns the larger of TemperatureMinDate and PrecipitationMinDate.
func (s *Station) MaxDate() NullDate {
	var m NullDate
	if s.TemperatureMinDate.HasValue {
		m = s.TemperatureMinDate
	}
	if s.PrecipitationMinDate.HasValue {
		if !m.HasValue || m.Value.Before(s.PrecipitationMinDate.Value.Time) {
			m = s.PrecipitationMinDate
		}
	}
	return m
}

// Date wraps time.Time but marshals/unmarshals as YYYY-MM-DD.
type Date struct {
	time.Time
}

func (d Date) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

func (d *Date) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return err
	}
	d.Time = t
	return nil
}

func (d Date) String() string {
	return d.Format("2006-01-02")
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (d *NullDate) UnmarshalJSON(data []byte) error {
	// If the value is the JSON literal 'null' or empty, it's not set.
	if bytes.Equal(data, []byte("null")) || len(data) == 0 {
		d.HasValue = false
		d.Value = Date{}
		return nil
	}

	// Try to unmarshal into the float64 value.
	var v Date
	if err := json.Unmarshal(data, &v); err != nil {
		return fmt.Errorf("error unmarshaling Date: %w", err)
	}

	d.Value = v
	d.HasValue = true
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (d NullDate) MarshalJSON() ([]byte, error) {
	// If the value was not set, marshal as JSON null.
	if !d.HasValue {
		return []byte("null"), nil
	}
	// Marshal the underlying value.
	return json.Marshal(d.Value)
}

// String implements the fmt.Stringer interface and makes Float64
// more convenient to use in templates.
func (d NullDate) String() string {
	if !d.HasValue {
		return "null"
	}
	return d.Value.String()
}

func (l LocalizedString) MarshalJSON() ([]byte, error)  { return json.Marshal(l.f) }
func (l *LocalizedString) UnmarshalJSON(b []byte) error { return json.Unmarshal(b, &l.f) }

func snakeToCamelCase(s string) string {
	parts := strings.Split(strings.ToLower(s), "_")
	for i, p := range parts {
		if len(p) > 0 {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, "")
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (f *NullFloat64) UnmarshalJSON(data []byte) error {
	// If the value is the JSON literal 'null' or empty, it's not set.
	if bytes.Equal(data, []byte("null")) || len(data) == 0 {
		f.HasValue = false
		f.Value = 0
		return nil
	}

	// Try to unmarshal into the float64 value.
	var v float64
	if err := json.Unmarshal(data, &v); err != nil {
		return fmt.Errorf("error unmarshaling float: %w", err)
	}

	f.Value = v
	f.HasValue = true
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (f NullFloat64) MarshalJSON() ([]byte, error) {
	// If the value was not set or is NaN or Inf, marshal as JSON null.
	if !f.HasValue || math.IsNaN(f.Value) || math.IsInf(f.Value, 0) {
		return []byte("null"), nil
	}

	// Otherwise, marshal the underlying float64 value.
	return json.Marshal(f.Value)
}

// String implements the fmt.Stringer interface and makes Float64
// more convenient to use in templates.
func (f NullFloat64) String() string {
	if !f.HasValue {
		return "null"
	}
	return strconv.FormatFloat(f.Value, 'f', -1, 64)
}
