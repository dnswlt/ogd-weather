from pydantic import BaseModel
from datetime import date


class Station(BaseModel):
    abbr: str
    name: str
    canton: str
    first_available_date: date | None = None
    last_available_date: date | None = None


class StationStats(BaseModel):
    first_date: date
    last_date: date
    period: str
    annual_temp_increase: float
    annual_precip_increase: float
    coldest_year: int | None = None
    coldest_year_temp: float | None = None
    warmest_year: int | None = None
    warmest_year_temp: float | None = None
    driest_year: int | None = None
    wettest_year: int | None = None


class StationSummary(BaseModel):
    station: Station
    stats: StationStats
