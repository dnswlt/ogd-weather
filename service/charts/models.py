from pydantic import BaseModel
from datetime import date


class Station(BaseModel):
    abbr: str
    name: str
    canton: str


class WeatherStats(BaseModel):
    station_abbr: str
    month: int
    first_date: date
    last_date: date
    annual_temp_increase: float
    annual_precip_increase: float
    hottest_year: int | None = None
