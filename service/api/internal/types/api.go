package types

import (
	"encoding/json"
	"time"
)

type Station struct {
	Abbr                string  `json:"abbr"`
	Name                string  `json:"name"`
	Canton              string  `json:"canton"`
	Typ                 string  `json:"typ"`
	Exposition          string  `json:"exposition"`
	HeightMASL          float64 `json:"height_masl"`
	CoordinatesWGS84Lat float64 `json:"coordinates_wgs84_lat"`
	CoordinatesWGS84Lon float64 `json:"coordinates_wgs84_lon"`
	FirstAvailableDate  *Date   `json:"first_available_date,omitempty"`
	LastAvailableDate   *Date   `json:"last_available_date,omitempty"`
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
	Station *Station
	Stats   *StationStats
}

type StationSummaryResponse struct {
	Summary *StationSummary `json:"summary,omitempty"`
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
