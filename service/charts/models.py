import datetime
from pydantic import BaseModel, ConfigDict
from datetime import date


class LocalizedString(BaseModel):
    de: str
    fr: str
    it: str
    en: str

    @classmethod
    def from_nullable(
        cls,
        de: str | None = None,
        fr: str | None = None,
        it: str | None = None,
        en: str | None = None,
    ) -> "LocalizedString":
        return cls(de=de or "", fr=fr or "", it=it or "", en=en or "")

    model_config = ConfigDict(frozen=True)


class Station(BaseModel):
    abbr: str
    name: str
    canton: str
    typ: str | None = None
    exposition: LocalizedString | None = None
    url: LocalizedString | None = None
    height_masl: float | None = None
    coordinates_wgs84_lat: float | None = None
    coordinates_wgs84_lon: float | None = None
    temperature_min_date: date | None = None
    temperature_max_date: date | None = None
    precipitation_min_date: date | None = None
    precipitation_max_date: date | None = None


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
