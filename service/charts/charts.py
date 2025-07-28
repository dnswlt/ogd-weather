from datetime import date
from typing import Iterable
import altair as alt
import numpy as np
import pandas as pd
from . import db
from . import models
from .errors import NoDataError
from . import params

PERIOD_ALL = "all"

MEASUREMENT_LABELS = {
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

# Measurement colums to load, by chart type.
CHART_TYPE_COLUMNS = {
    "temperature": [db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX],
    "temperature_deviation": [db.TEMP_DAILY_MEAN],
    "precipitation": [db.PRECIP_DAILY_MM],
    "raindays": [db.PRECIP_DAILY_MM],
    "sunshine": [db.SUNSHINE_DAILY_MINUTES],
    "sunny_days": [db.SUNSHINE_DAILY_MINUTES],
    "summer_days": [db.TEMP_DAILY_MAX],
    "frost_days": [db.TEMP_DAILY_MIN],
}


def period_to_title(period: str) -> str:
    if period.isdigit():
        return MONTH_NAMES[int(period)]
    elif period in SEASON_NAMES:
        return SEASON_NAMES[period]
    elif period == PERIOD_ALL:
        return "Whole Year"
    return "Unknown Period"


def verify_columns(df: pd.DataFrame, columns: Iterable[str]):
    if not set(columns) <= set(df.columns):
        raise ValueError(
            f"DataFrame does not contain expected columns (want: {columns}, got: {df.columns})"
        )


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
    elif period == PERIOD_ALL:
        # All months are fine
        pass
    else:
        # Should not happen if validation is done before, but good to have
        raise ValueError(f"Unknown period: {period}")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames all well-known columns from their ODG SwissMetNet (smn) names to human-readable names.

    Example: tre200d0 -> temp_2m_mean. See `MEASUREMENT_LABELS` for the full list.
    """
    return df.rename(columns=MEASUREMENT_LABELS)


def annual_agg(df, func):
    """Returns a DataFrame with one row per year containing average values."""
    df_y = df.groupby(df.index.year).agg(func)
    df_y.index.name = "year"
    return df_y


def long_format(df):
    index_name = df.index.name
    return df.reset_index().melt(id_vars=index_name)


def rolling_mean(df: pd.DataFrame, window: int = 5):
    """Compute rolling mean. Drop initial rows that don't have enough periods."""
    return df.rolling(window=window, min_periods=window).mean()


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
    values_long: pd.DataFrame,
    trend_long: pd.DataFrame | None = None,
    typ: str = "line",
    y_label: str = "value",
    title: str = "Untitled chart",
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
    trend = None
    if trend_long is not None:
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

    layer_chart = values + trend if trend else alt.layer(values)
    return (
        layer_chart.add_params(highlight)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=title,
        )
        .interactive()
    )


def create_dynamic_baseline_bars(
    values_long: pd.DataFrame, title="Untitled chart"
) -> alt.LayerChart:
    df = values_long.copy()

    # Compute baseline
    baseline = df["value"].mean()
    df["anomaly"] = df["value"] - baseline
    df["sign"] = df["anomaly"].apply(lambda x: "Below mean" if x < 0 else "Above mean")

    # Color scale for below/above baseline
    color_scale = alt.Scale(
        domain=["Below mean", "Above mean"], range=["#2166ac", "#b2182b"]
    )

    # Bars for anomalies
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:Q", axis=alt.Axis(format="d", title=None)),
            y=alt.Y("anomaly:Q", title=f"Deviation from mean ({baseline:.2f} °C)"),
            color=alt.Color(
                "sign:N", scale=color_scale, legend=alt.Legend(title="Vs. baseline")
            ),
            tooltip=[
                alt.Tooltip("year:Q", title="Year"),
                alt.Tooltip("value:Q", title="Temp (°C)"),
                alt.Tooltip("anomaly:Q", title="Δ vs mean"),
            ],
        )
    )

    # Zero line
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color="black")
        .encode(y="y:Q")
    )

    chart_title = f"{title} (Baseline = {baseline:.2f} °C over {df['year'].min()}–{df['year'].max()})"

    return (
        (bars + zero_line)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=chart_title,
        )
        .interactive()
    )


def raindays_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No raindays data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["raindays"])

    raindays = df[[db.PRECIP_DAILY_MM]]
    raindays = raindays[raindays[db.PRECIP_DAILY_MM] >= 0.1]
    raindays = raindays.rename(columns={db.PRECIP_DAILY_MM: "# days"})

    raindays_m = annual_agg(raindays, "count")

    raindays_long = long_format(raindays_m).dropna()

    _, trend = polyfit_columns(raindays_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Number of rain days (≥ 0.1 mm precip.) in {period_to_title(period)}, by year".strip()
    return create_chart_trendline(
        raindays_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="# days",
    ).to_dict()


def sunny_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No sunshine data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["sunny_days"])

    data = pd.DataFrame(
        {"# days": (df[db.SUNSHINE_DAILY_MINUTES] >= 6 * 60).astype(int)}
    )

    data_m = annual_agg(data, "sum")

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Number of sunny days (≥ 6 h of sunshine) in {period_to_title(period)}, by year".strip()
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="# days",
    ).to_dict()


def frost_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No frost data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["frost_days"])

    data = pd.DataFrame({"# days": (df[db.TEMP_DAILY_MIN] < 0).astype(int)})

    data_m = annual_agg(data, "sum")

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Number of frost days (min. < 0 °C) in {period_to_title(period)}, by year".strip()
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="# days",
    ).to_dict()


def summer_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No hot days data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["summer_days"])

    data = pd.DataFrame({"# days": (df[db.TEMP_DAILY_MAX] >= 25).astype(int)})

    data_m = annual_agg(data, "sum")

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Number of summer days (max. ≥ 25 °C) in {period_to_title(period)}, by year".strip()
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="# days",
    ).to_dict()


def sunshine_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No sunshine data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["sunshine"])

    data = pd.DataFrame(
        {
            "sunshine (h)": df[db.SUNSHINE_DAILY_MINUTES] / 60.0,
        }
    )

    data_m = annual_agg(data, "mean")

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = (
        f"Mean daily hours of sunshine in {period_to_title(period)}, by year".strip()
    )
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="hours/d",
    ).to_dict()


def temperature_deviation_chart(
    df: pd.DataFrame,
    station_abbr: str,
    period: str = PERIOD_ALL,
    window: int | None = None,
):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No temperature data for {station_abbr}")
    verify_period(df, period)

    data = rename_columns(df[CHART_TYPE_COLUMNS["temperature_deviation"]])

    data_m = annual_agg(data, "mean")
    if window and window > 1:
        data_m = rolling_mean(data_m, window=window)

    data_long = long_format(data_m).dropna()

    if data_long.empty:
        raise NoDataError(f"No aggregate temperature data for {station_abbr}")

    return create_dynamic_baseline_bars(
        data_long,
        f"Temperature deviation from mean in {period_to_title(period)}",
    ).to_dict()


def temperature_chart(
    df: pd.DataFrame,
    station_abbr: str,
    period: str = PERIOD_ALL,
    window: int | None = None,
) -> alt.LayerChart:
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No temperature data for {station_abbr}")
    verify_period(df, period)

    data = rename_columns(df[CHART_TYPE_COLUMNS["temperature"]])

    data_m = annual_agg(data, "mean")
    if window and window > 1:
        data_m = rolling_mean(data_m, window=window)

    data_long = long_format(data_m).dropna()

    if data_long.empty:
        raise NoDataError(f"No aggregated temperature data for {station_abbr}")

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    window_info = f"({window}y rolling avg.)" if window else ""
    title = (
        f"Avg. temperatures in {period_to_title(period)}, by year {window_info}".strip()
    )
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="line",
        title=title,
        y_label="°C",
    ).to_dict()


def precipitation_chart(
    df: pd.DataFrame,
    station_abbr: str,
    period: str = PERIOD_ALL,
    window: int | None = None,
):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No precipitation data for {station_abbr}")
    verify_period(df, period)

    data = rename_columns(df[CHART_TYPE_COLUMNS["precipitation"]])

    data_m = annual_agg(data, "sum")
    if window and window > 1:
        data_m = rolling_mean(data_m, window=window)

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    window_info = f"({window}y rolling avg.)" if window else ""
    title = f"Total precipitation in {period_to_title(period)}, by year {window_info}".strip()
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="mm",
    ).to_dict()


def create_chart(
    chart_type: str,
    df: pd.DataFrame,
    station_abbr: str,
    period: str = PERIOD_ALL,
    window: int | None = None,
):
    if chart_type == "temperature":
        return temperature_chart(
            df, station_abbr=station_abbr, period=period, window=window
        )
    elif chart_type == "precipitation":
        return precipitation_chart(
            df, station_abbr=station_abbr, period=period, window=window
        )
    elif chart_type == "temperature_deviation":
        return temperature_deviation_chart(
            df, station_abbr=station_abbr, period=period, window=window
        )
    elif chart_type == "raindays":
        return raindays_chart(df, station_abbr=station_abbr, period=period)
    elif chart_type == "sunshine":
        return sunshine_chart(df, station_abbr=station_abbr, period=period)
    elif chart_type == "sunny_days":
        return sunny_days_chart(df, station_abbr=station_abbr, period=period)
    elif chart_type == "summer_days":
        return summer_days_chart(df, station_abbr=station_abbr, period=period)
    elif chart_type == "frost_days":
        return frost_days_chart(df, station_abbr=station_abbr, period=period)
    else:
        raise ValueError(f"Invalid chart type: {chart_type}")


def station_stats(
    df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL
) -> models.StationStats:
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No stats data for {station_abbr}")
    verify_period(df, period)

    df = df[
        [db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX, db.PRECIP_DAILY_MM]
    ]
    first_date = df.index.min().to_pydatetime().date()
    last_date = df.index.max().to_pydatetime().date()
    df_m = annual_agg(df, "mean")
    temp_dm = df_m[db.TEMP_DAILY_MEAN].dropna()
    coldest_year = temp_dm.idxmin() if not temp_dm.empty else None
    coldest_year_temp = temp_dm.min() if not temp_dm.empty else None
    warmest_year = temp_dm.idxmax() if not temp_dm.empty else None
    warmest_year_temp = temp_dm.max() if not temp_dm.empty else None
    precip = df_m[db.PRECIP_DAILY_MM].dropna()
    driest_year = precip.idxmin() if not precip.empty else None
    wettest_year = precip.idxmax() if not precip.empty else None

    annual_temp_increase = None
    if not temp_dm.empty:
        coeffs, _ = polyfit_columns(df_m[[db.TEMP_DAILY_MEAN]], deg=1)
        annual_temp_increase = coeffs[db.TEMP_DAILY_MEAN].iloc[0]

    return models.StationStats(
        first_date=first_date,
        last_date=last_date,
        period=period_to_title(period),
        annual_temp_increase=annual_temp_increase,
        coldest_year=coldest_year,
        coldest_year_temp=coldest_year_temp,
        warmest_year=warmest_year,
        warmest_year_temp=warmest_year_temp,
        driest_year=driest_year,
        wettest_year=wettest_year,
    )


def daily_measurements(
    df: pd.DataFrame, station_abbr: str
) -> models.StationMeasurementsData:
    rows = []
    measurements = df.select_dtypes(include=["number"]).astype("float")
    for t, d in measurements.iterrows():
        rows.append(
            models.MeasurementsRow(
                reference_timestamp=t.to_pydatetime(),
                measurements=list(d),
            )
        )
    return models.StationMeasurementsData(
        station_abbr=station_abbr,
        rows=rows,
        columns=[
            models.ColumnInfo(
                name=c,
                dtype=str(df[c].dtype),
            )
            for c in measurements.columns
        ],
    )


def station_period_stats(s: pd.Series) -> models.StationPeriodStats:
    def _vstats(var: str):
        v = s[var]
        return models.VariableStats(
            min_value=float(v["min_value"]),
            min_value_date=date.fromisoformat(v["min_value_date"]),
            mean_value=float(v["mean_value"]),
            max_value=float(v["max_value"]),
            max_value_date=date.fromisoformat(v["max_value_date"]),
            source_granularity=v["source_granularity"],
            value_sum=v["value_sum"],
            value_count=v["value_count"],
        )

    def _key(k: str) -> str:
        if d := db.VARIABLE_API_NAMES.get(k):
            return d
        return k

    variable_stats = {_key(v): _vstats(v) for v in s.index.get_level_values(0).unique()}
    return models.StationPeriodStats(
        start_date=date(1991, 1, 1),
        end_date=date(2020, 12, 31),
        variable_stats=variable_stats,
    )
