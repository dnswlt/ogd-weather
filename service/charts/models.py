import datetime
from pydantic import BaseModel
from datetime import date


class Station(BaseModel):
    abbr: str
    name: str
    canton: str
    typ: str | None = None
    exposition: str | None = None
    height_masl: float | None = None
    coordinates_wgs84_lat: float | None = None
    coordinates_wgs84_lon: float | None = None
    first_available_date: date | None = None
    last_available_date: date | None = None


class StationStats(BaseModel):
    """Weather data statistics for the requested period."""

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


class MeasurementsRow(BaseModel):
    reference_timestamp: datetime.datetime
    measurements: list[float]


class ColumnInfo(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    dtype: str = ""


class StationMeasurementsData(BaseModel):
    """Generic holder of measurement data for a single station."""

    station_abbr: str
    rows: list[MeasurementsRow]
    columns: list[ColumnInfo]
