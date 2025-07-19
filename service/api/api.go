package api

import (
	"encoding/json"
	"time"
)

type Station struct {
	Abbr   string `json:"abbr"`
	Name   string `json:"name"`
	Canton string `json:"canton"`
}

type StationsResponse struct {
	Stations []*Station `json:"stations"`
}

type StationSummary struct {
	StationAbbr          string  `json:"station_abbr"`
	Month                int     `json:"month"`
	FirstDate            Date    `json:"first_date"`
	LastDate             Date    `json:"last_date"`
	AnnualTempIncrease   float64 `json:"annual_temp_increase"`
	AnnualPrecipIncrease float64 `json:"annual_precip_increase"`
	ColdestYear          int     `json:"coldest_year,omitempty"`
	ColdestYearTemp      float64 `json:"coldest_year_temp,omitempty"`
	WarmestYear          int     `json:"warmest_year,omitempty"`
	WarmestYearTemp      float64 `json:"warmest_year_temp,omitempty"`
	DriestYear           int     `json:"driest_year,omitempty"`
	WettestYear          int     `json:"wettest_year,omitempty"`
}

type StationSummaryResponse struct {
	Summary *StationSummary `json:"summary,omitempty"`
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
