package types

import (
	"encoding/json"
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

type Station struct {
	Abbr                string          `json:"abbr"`
	Name                string          `json:"name"`
	Canton              string          `json:"canton"`
	Typ                 string          `json:"typ"`
	Exposition          LocalizedString `json:"exposition"`
	URL                 LocalizedString `json:"url"`
	HeightMASL          float64         `json:"height_masl"`
	CoordinatesWGS84Lat float64         `json:"coordinates_wgs84_lat"`
	CoordinatesWGS84Lon float64         `json:"coordinates_wgs84_lon"`
	// These dates must be optional so they don't get serialized as 0000-00-00 in JSON.
	TemperatureMinDate   *Date `json:"temperature_min_date,omitempty"`
	TemperatureMaxDate   *Date `json:"temperature_max_date,omitempty"`
	PrecipitationMinDate *Date `json:"precipitation_min_date,omitempty"`
	PrecipitationMaxDate *Date `json:"precipitation_max_date,omitempty"`
}

func (s *Station) MinDate() Date {
	var m Date
	if s.TemperatureMinDate != nil {
		m = *s.TemperatureMinDate
	}
	if s.PrecipitationMinDate != nil {
		if m.IsZero() || m.After(s.PrecipitationMinDate.Time) {
			m = *s.PrecipitationMinDate
		}
	}
	return m
}

func (s *Station) MaxDate() Date {
	var m Date
	if s.TemperatureMaxDate != nil {
		m = *s.TemperatureMaxDate
	}
	if s.PrecipitationMaxDate != nil {
		if m.IsZero() || m.Before(s.PrecipitationMaxDate.Time) {
			m = *s.PrecipitationMaxDate
		}
	}
	return m
}

type StationsResponse struct {
	Stations []*Station `json:"stations"`
}

type StationStats struct {
	FirstDate            Date    `json:"first_date"`
	LastDate             Date    `json:"last_date"`
	Period               string  `json:"period"`
	AnnualTempIncrease   float64 `json:"annual_temp_increase"`
	AnnualPrecipIncrease float64 `json:"annual_precip_increase"`
	ColdestYear          int     `json:"coldest_year,omitempty"`
	ColdestYearTemp      float64 `json:"coldest_year_temp,omitempty"`
	WarmestYear          int     `json:"warmest_year,omitempty"`
	WarmestYearTemp      float64 `json:"warmest_year_temp,omitempty"`
	DriestYear           int     `json:"driest_year,omitempty"`
	WettestYear          int     `json:"wettest_year,omitempty"`
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
