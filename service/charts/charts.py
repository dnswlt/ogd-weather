from operator import attrgetter
import altair as alt
import functools
import numpy as np
import os
import pandas as pd
from . import models

BASE_DIR = os.environ.get("OGD_BASE_DIR", ".")


class StationNotFoundError(ValueError):
    """Raised when a requested station doesn't exist."""


class NoDataError(ValueError):
    """Raised when a request is valid, but no data is available."""


def read_timeseries_csv(filename: str) -> pd.DataFrame:
    """Reads a CSV file from Swiss Meteo, parsing date columns and using the right (cp1252) encoding."""
    df = pd.read_csv(
        filename,
        sep=";",
        encoding="cp1252",
        parse_dates=["reference_timestamp"],
        date_format="%d.%m.%Y %H:%M",
    )
    return df.set_index("reference_timestamp").sort_index()


@functools.cache
def read_stations(filename: str) -> list[models.Station]:
    """Reads the metadata CSV (typically ogd-smn_meta_stations.csv) and returns the station_abbr column."""
    df = pd.read_csv(filename, sep=";", encoding="cp1252")
    stations = [
        models.Station(
            abbr=row["station_abbr"],
            name=row["station_name"],
            canton=row["station_canton"],
        )
        for _, row in df.iterrows()
    ]
    return sorted(stations, key=attrgetter("abbr"))


def read_station_data(base_dir: str, station_abbr: str = "ber") -> pd.DataFrame:
    filename = os.path.join(
        base_dir, f"ogd-smn_{station_abbr.lower()}_d_historical.csv"
    )
    if not os.path.isfile(filename):
        raise StationNotFoundError(f"CSV file {filename} does not exist")
    return read_timeseries_csv(filename)


def extract_temperature(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["tre200d0", "tre200dx", "tre200dn"]]
    column_renames = {
        "tre200d0": "temp_2m_mean",
        "tre200dx": "temp_2m_max",
        "tre200dn": "temp_2m_min",
    }
    return df.rename(columns=column_renames).dropna()


def extract_precipitation(df):
    df = df[["rka150d0"]]
    return df.rename(columns={"rka150d0": "precip_mm"}).dropna()


def annual_agg(df, func):
    """Returns a DataFrame with one row per year containing average values."""
    df_y = df.groupby(df.index.year).agg(func)
    df_y.index.name = "year"
    return df_y


def monthly_average(df, month):
    """Returns a DataFrame with one row per year containing average values for the given month (1 = January)."""
    return annual_agg(df[df.index.month == month], "mean")


def monthly_sum(df, month):
    """Returns a DataFrame with one row per year containing cumulative values for the given month (1 = January)."""
    return annual_agg(df[df.index.month == month], "sum")


def rolling_mean_long(df, window=5):
    # Compute rolling mean. Drop initial rows that don't have enough periods.
    rolling = df.rolling(window=window, min_periods=window).mean().dropna()

    # Convert to long format
    return rolling.reset_index(names="year").melt(id_vars="year")


def polyfit_columns(df, deg=1):
    """Fits a curve (using np.polyfit with degree deg) to each column of df."""
    trend = {}
    for col in df.columns:
        x = df.index.values
        y = df[col].values
        coeffs = np.polyfit(x, y, deg=deg)
        y_fit = np.polyval(coeffs, x)
        trend[col] = y_fit
    return pd.DataFrame(trend, index=df.index)


def create_chart(
    values_long, trend_long, y_label="value", title="Untitled chart"
) -> alt.LayerChart:
    highlight = alt.selection_point(fields=["variable"], bind="legend")

    # Actual data
    lines = (
        alt.Chart(values_long)
        .mark_line()
        .encode(
            x=alt.X("year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("value:Q", title=y_label),
            color="variable:N",
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.1)),
            tooltip=["year:Q", "variable:N", "value:Q"],
        )
    )

    # Trendlines
    trend = (
        alt.Chart(trend_long)
        .mark_line(strokeDash=[4, 4])
        .encode(
            x="year:Q",
            y="value:Q",
            color="variable:N",
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.1)),
        )
    )

    chart = (
        (lines + trend)
        .add_params(highlight)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=title,
        )
    )

    return chart


def temperature_chart(station_abbr: str, month: int = 6):
    data = read_station_data(BASE_DIR, station_abbr)
    temp = extract_temperature(data)
    if temp.empty:
        raise NoDataError(f"No temperature data for {station_abbr}")

    temp_m = monthly_average(temp, month)

    trend = polyfit_columns(temp_m, deg=1)
    trend_long = trend.reset_index().melt(id_vars="year")
    rolling_long = rolling_mean_long(temp_m)

    return create_chart(
        rolling_long,
        trend_long,
        title="Temperatures in given month (5y rolling avg + trendline)",
        y_label="temperature",
    ).to_dict()


def list_stations(cantons: list[str] = None):
    all_stations = read_stations(os.path.join(BASE_DIR, "ogd-smn_meta_stations.csv"))
    if not cantons:
        return all_stations
    cantons = set(c.upper() for c in cantons)
    return [s for s in all_stations if s.canton in cantons]
