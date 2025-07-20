import altair as alt
import numpy as np
import os
import pandas as pd
from . import db
from . import models

BASE_DIR = os.environ.get("OGD_BASE_DIR", ".")


class StationNotFoundError(ValueError):
    """Raised when a requested station doesn't exist."""


class NoDataError(ValueError):
    """Raised when a request is valid, but no data is available."""


MEASUREMENT_NAMES = {
    db.TEMP_DAILY_MEAN: "temp_2m_mean",
    db.TEMP_DAILY_MAX: "temp_2m_max",
    db.TEMP_DAILY_MIN: "temp_2m_min",
    db.PRECIP_DAILY_MM: "precip_mm",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

SEASON_NAMES = {
    "spring": "Spring (Mar-May)",
    "summer": "Summer (Jun-Aug)",
    "autumn": "Autumn (Sep-Nov)",
    "winter": "Winter (Dec-Feb)",
}

SEASONS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}


def period_to_title(period: str) -> str:
    if period.isdigit():
        return MONTH_NAMES[int(period)]
    elif period in SEASON_NAMES:
        return SEASON_NAMES[period]
    elif period == "all":
        return "Whole Year"
    return "Unknown Period"


def verify_period(df: pd.DataFrame, period: str):
    """Verifies that all dates in the DataFrame match the given period."""
    if df.empty:
        return

    months_in_df = df.index.month.unique()

    if period.isdigit():
        expected_month = int(period)
        if not (months_in_df.size == 1 and months_in_df[0] == expected_month):
            raise ValueError(
                f"Data contains months other than the expected month {expected_month} for period '{period}'"
            )
    elif period in SEASONS:
        expected_months = set(SEASONS[period])
        if not set(months_in_df).issubset(expected_months):
            raise ValueError(
                f"Data contains months outside the expected season {period} ({expected_months})"
            )
    elif period == "all":
        # All months are fine
        pass
    else:
        # Should not happen if validation is done before, but good to have
        raise ValueError(f"Unknown period: {period}")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames all well-known columns from their ODG SwissMetNet (smn) names to human-readable names.

    Example: tre200d0 -> temp_2m_mean. See MEASUREMENT_NAMES for the full list.
    """
    return df.rename(columns=MEASUREMENT_NAMES)


def annual_agg(df, func):
    """Returns a DataFrame with one row per year containing average values."""
    df_y = df.groupby(df.index.year).agg(func)
    df_y.index.name = "year"
    return df_y


def long_format(df):
    index_name = df.index.name
    return df.reset_index().melt(id_vars=index_name)


def rolling_mean_long(df, window=5):
    # Compute rolling mean. Drop initial rows that don't have enough periods.
    rolling = df.rolling(window=window, min_periods=window).mean()

    # Convert to long format
    return rolling.reset_index(names="year").melt(id_vars="year").dropna()


def polyfit_columns(df: pd.DataFrame, deg: int = 1) -> tuple[np.ndarray, pd.DataFrame]:
    """Fits a curve (using np.polyfit with degree deg) to each column of df."""
    trend = {}
    coeffs = {}
    for col in df.columns:
        vals = df[col].dropna()
        x = vals.index.values
        y = vals.values
        p = np.polyfit(x, y, deg=deg)
        y_fit = np.polyval(p, x)
        trend[col] = pd.Series(y_fit, index=vals.index)
        coeffs[col] = p
    return pd.DataFrame(coeffs), pd.concat(trend, axis=1)


def create_chart_trendline(
    values_long, trend_long, typ="line", y_label="value", title="Untitled chart"
) -> alt.LayerChart:
    highlight = alt.selection_point(fields=["variable"], bind="legend")

    # Actual data
    values = alt.Chart(values_long)
    if typ == "line":
        values = values.mark_line()
    elif typ == "bar":
        values = values.mark_bar()
    else:
        raise ValueError(f"Unsupported chart type {typ}")

    values = values.encode(
        x=alt.X("year:Q", axis=alt.Axis(format="d", title=None)),
        y=alt.Y("value:Q", title=y_label),
        color=alt.Color("variable:N", legend=alt.Legend(title="Variable")),
        opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.1)),
        tooltip=["year:Q", "variable:N", "value:Q"],
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
        (values + trend)
        .add_params(highlight)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=title,
        )
        .interactive()
    )

    return chart


def temperature_chart(df: pd.DataFrame, station_abbr: str, period: str = "6"):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No temperature data for {station_abbr}")
    verify_period(df, period)

    temp = rename_columns(
        df[[db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX]]
    )

    temp_m = annual_agg(temp, "mean")

    _, trend = polyfit_columns(temp_m, deg=1)
    trend_long = trend.reset_index().melt(id_vars="year").dropna()
    rolling_long = rolling_mean_long(temp_m)

    title = f"Temperatures in {period_to_title(period)} (5y rolling avg. + trendline)"
    return create_chart_trendline(
        rolling_long,
        trend_long,
        typ="line",
        title=title,
        y_label="°C",
    ).to_dict()


def precipitation_chart(df: pd.DataFrame, station_abbr: str, period: str = "6"):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No precipitation data for {station_abbr}")
    verify_period(df, period)

    precip = rename_columns(df[[db.PRECIP_DAILY_MM]])

    precip_m = annual_agg(precip, "sum")
    precip_long = long_format(precip_m).dropna()

    _, trend = polyfit_columns(precip_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Monthly precipitation in {period_to_title(period)}"
    return create_chart_trendline(
        precip_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="mm",
    ).to_dict()


def station_summary(df: pd.DataFrame, station_abbr: str, period: str = "6"):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No temperature data for {station_abbr}")
    verify_period(df, period)

    df = df[
        [db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX, db.PRECIP_DAILY_MM]
    ]
    first_date = df.index.min().to_pydatetime().date()
    last_date = df.index.max().to_pydatetime().date()
    df_m = annual_agg(df, "mean")
    coldest_year = df_m[db.TEMP_DAILY_MEAN].idxmin()
    coldest_year_temp = df_m[db.TEMP_DAILY_MEAN].min()
    warmest_year = df_m[db.TEMP_DAILY_MEAN].idxmax()
    warmest_year_temp = df_m[db.TEMP_DAILY_MEAN].max()
    driest_year = df_m[db.PRECIP_DAILY_MM].idxmin()
    wettest_year = df_m[db.PRECIP_DAILY_MM].idxmax()

    coeffs, _ = polyfit_columns(df_m, deg=1)
    return models.StationSummary(
        station_abbr=station_abbr,
        period=period,
        annual_temp_increase=coeffs[db.TEMP_DAILY_MEAN].iloc[0],
        annual_precip_increase=coeffs[db.PRECIP_DAILY_MM].iloc[0],
        first_date=first_date,
        last_date=last_date,
        coldest_year=coldest_year,
        coldest_year_temp=coldest_year_temp,
        warmest_year=warmest_year,
        warmest_year_temp=warmest_year_temp,
        driest_year=driest_year,
        wettest_year=wettest_year,
    )
