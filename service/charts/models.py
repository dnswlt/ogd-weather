from pydantic import BaseModel
from datetime import date


class Station(BaseModel):
    abbr: str
    name: str
    canton: str


class StationSummary(BaseModel):
    station_abbr: str
    period: str
    first_date: date
    last_date: date
    annual_temp_increase: float
    annual_precip_increase: float
    coldest_year: int | None = None
    coldest_year_temp: float | None = None
    warmest_year: int | None = None
    warmest_year_temp: float | None = None
    driest_year: int | None = None
    wettest_year: int | None = None
