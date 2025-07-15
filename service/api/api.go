package api

type Station struct {
	Abbr   string `json:"abbr"`
	Name   string `json:"name"`
	Canton string `json:"canton"`
}

type StationsResponse struct {
	Stations []*Station `json:"stations"`
}
