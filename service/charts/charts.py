from datetime import date
import datetime
from typing import Any, Iterable
import altair as alt
import numpy as np
import pandas as pd
from . import db
from . import models
from .errors import NoDataError

PERIOD_ALL = "all"

VEGA_LEGEND_LABELS = {
    db.TEMP_DAILY_MEAN: "temp mean",
    db.TEMP_DAILY_MAX: "temp max",
    db.TEMP_DAILY_MIN: "temp min",
    db.PRECIP_DAILY_MM: "precip mm",
    db.TEMP_HOURLY_MEAN: "temp mean",
    db.TEMP_HOURLY_MAX: "temp max",
    db.TEMP_HOURLY_MIN: "temp min",
    db.PRECIP_HOURLY_MM: "precip mm",
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
    "rainiest_day": [db.PRECIP_DAILY_MM],
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


_SEASONS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}


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
    elif period in _SEASONS:
        expected_months = set(_SEASONS[period])
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

    Example: tre200d0 -> "temp mean". See `VEGA_LEGEND_LABELS` for the full list.
    """
    return df.rename(columns=VEGA_LEGEND_LABELS)


def annual_agg(df, func):
    """Returns a DataFrame with one row per year containing average values."""
    df_y = df.groupby(df.index.year).agg(func)
    df_y.index.name = "year"
    return df_y


def long_format(df: pd.DataFrame, id_cols: list[str] | None = None) -> pd.DataFrame:
    index_name = df.index.name
    return df.reset_index().melt(id_vars=[index_name] + (id_cols or []))


def rolling_mean(df: pd.DataFrame, window: int = 5):
    """Compute rolling mean. Drop initial rows that don't have enough periods."""
    return df.rolling(window=window, min_periods=window).mean()


def polyfit_columns(
    df: pd.DataFrame, deg: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits a Polynomial (degree deg) to each column of df using numpy.polynomial.

    Returns:
        - A DataFrame of coefficients (per column, highest degree last)
        - A DataFrame of trend values (same shape as df)
    """
    trend = {}
    coeffs = {}

    for col in df.columns:
        vals = df[col].dropna()
        x = vals.index.values
        y = vals.values

        # Fit polynomial to column
        if len(x) < deg + 1:
            raise ValueError(
                f"Cannot fit degree-{deg} polynomial to only {len(x)} point(s) in column '{col}'"
            )
        p = np.polynomial.Polynomial.fit(x, y, deg=deg).convert()
        y_fit = p(x)

        trend[col] = pd.Series(y_fit, index=vals.index)
        coeffs[col] = p.coef

    coeffs_df = pd.DataFrame(coeffs)  # shape: (deg+1, num_columns)
    return coeffs_df, pd.concat(trend, axis=1)


def _year_to_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Plotting helper: turns the year:int column into year:nsdatetime.

    This allows us to use :T and timeUnit="year" for the x axis.
    """
    return df.assign(year=pd.to_datetime(df["year"], format="%Y"))


def create_chart_trendline(
    values_long: pd.DataFrame,
    trend_long: pd.DataFrame | None = None,
    typ: str = "line",
    y_label: str = "value",
    title: str = "Untitled chart",
) -> alt.LayerChart:
    highlight = alt.selection_point(fields=["variable"], bind="legend")

    values_long = _year_to_dt(values_long)
    # Actual data
    values = alt.Chart(values_long)
    if typ == "line":
        values = values.mark_line()
    elif typ == "bar":
        values = values.mark_bar()
    else:
        raise ValueError(f"Unsupported chart type {typ}")

    values = values.encode(
        x=alt.X("year:T", timeUnit="year"),
        y=alt.Y("value:Q", title=y_label),
        color=alt.Color("variable:N", legend=alt.Legend(title="Variable")),
        opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.1)),
        tooltip=[
            alt.Tooltip("year:T", title="Year", format="%Y"),
            "variable:N",
            "value:Q",
        ],
    )

    # Trendlines
    trend = None
    if trend_long is not None:
        trend_long = _year_to_dt(trend_long)
        trend = (
            alt.Chart(trend_long)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x=alt.X("year:T", timeUnit="year"),
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
    df = values_long.copy(deep=False)

    df["year"] = pd.to_datetime(df["year"], format="%Y")

    # Compute baseline
    baseline = df["value"].mean()
    df["anomaly"] = df["value"] - baseline
    df["sign"] = df["anomaly"].apply(lambda x: "below mean" if x < 0 else "above mean")

    # Color scale for below/above baseline
    color_scale = alt.Scale(
        domain=["below mean", "above mean"], range=["#2166ac", "#b2182b"]
    )

    # Bars for anomalies
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:T", timeUnit="year"),
            y=alt.Y("anomaly:Q", title=f"Deviation from mean ({baseline:.2f} °C)"),
            color=alt.Color(
                "sign:N", scale=color_scale, legend=alt.Legend(title="Vs. baseline")
            ),
            tooltip=[
                alt.Tooltip("year:T", title="Year", format="%Y"),
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

    year_range = f"{values_long['year'].min()} - {values_long['year'].max()}"
    chart_title = f"{title} (Baseline = {baseline:.2f} °C over {year_range})"

    return (
        (bars + zero_line)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=chart_title,
        )
        .interactive()
    )


def day_count_chart_data(predicate: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares the day count and trendline data for "# days" charts.

    Returns data and trendline in long format. The trendline can be None if
    no line can be fitted to the data.
    """
    if predicate.empty:
        raise NoDataError(f"No data for day count chart")

    data = pd.DataFrame({"# days": predicate.astype(int)})

    data_m = annual_agg(data, "sum")
    data_long = long_format(data_m).dropna()

    try:
        _, trend = polyfit_columns(data_m)
        trend_long = long_format(trend).dropna()
    except ValueError:
        trend_long = None

    return data_long, trend_long


def day_count_chart(
    predicate: pd.Series,
    period: str = PERIOD_ALL,
    title: str = "Untitled chart",
):
    """Creates a chart for "# days" of some boolean predicate."""
    data_long, trend_long = day_count_chart_data(predicate)
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=f"{title} in {period_to_title(period)}, by year".strip(),
        y_label="# days",
    ).to_dict()


def _verify_day_count_data(
    df: pd.DataFrame, station_abbr: str, period: str, column: str
):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No raindays data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, [column])


def raindays_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    """Creates a "# rain days" chart for the given station and period."""
    _verify_day_count_data(df, station_abbr, period, db.PRECIP_DAILY_MM)
    return day_count_chart(
        predicate=df[db.PRECIP_DAILY_MM] >= 0.1,
        period=period,
        title="Number of rain days (≥ 0.1 mm precip.)",
    )


def sunny_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    """Creates a "# sunny days" chart for the given station and period."""
    _verify_day_count_data(df, station_abbr, period, db.SUNSHINE_DAILY_MINUTES)
    return day_count_chart(
        predicate=df[db.SUNSHINE_DAILY_MINUTES] >= 6 * 60,
        period=period,
        title="Number of sunny days (≥ 6 h of sunshine)",
    )


def frost_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    """Creates a "# frost days" chart for the given station and period."""
    _verify_day_count_data(df, station_abbr, period, db.TEMP_DAILY_MIN)
    return day_count_chart(
        predicate=df[db.TEMP_DAILY_MIN] < 0,
        period=period,
        title="Number of frost days (min. < 0 °C)",
    )


def summer_days_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    """Creates a "# summer days" chart for the given station and period."""
    _verify_day_count_data(df, station_abbr, period, db.TEMP_DAILY_MAX)
    return day_count_chart(
        predicate=df[db.TEMP_DAILY_MAX] >= 25,
        period=period,
        title="Number of summer days (max. ≥ 25 °C)",
    )


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


def rainiest_day_chart(df: pd.DataFrame, station_abbr: str, period: str = PERIOD_ALL):
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No precipitation data for {station_abbr}")
    verify_period(df, period)
    verify_columns(df, CHART_TYPE_COLUMNS["rainiest_day"])

    data = pd.DataFrame(
        {
            "precip (mm)": df[db.PRECIP_DAILY_MM],
        }
    )

    data_m = annual_agg(data, "max")

    data_long = long_format(data_m).dropna()

    _, trend = polyfit_columns(data_m, deg=1)
    trend_long = long_format(trend).dropna()

    title = f"Max daily amount of rain in {period_to_title(period)}, by year".strip()
    return create_chart_trendline(
        data_long,
        trend_long,
        typ="bar",
        title=title,
        y_label="mm",
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


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    """Convert a #rrggbb hex color to (r, g, b) tuple in [0.0, 1.0]."""
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    return (r, g, b)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert an (r, g, b) tuple in [0.0, 1.0] to a #rrggbb hex color."""
    return "#{:02x}{:02x}{:02x}".format(
        round(rgb[0] * 255),
        round(rgb[1] * 255),
        round(rgb[2] * 255),
    )


def _renormalize_stops(
    stops: list[dict[str, Any]], y0_ref: float, y1_ref: float, y0: float, y1: float
):
    """Rescales a Vega gradient by sampling the original color gradient
    at new positions, effectively changing the mapping from [y0_ref, y1_ref]
    to [y0, y1].

    Returns:
        New stops with offsets in [0, 1] and colors matching the values that
        y0 and y1 would have had in the original gradient.
    """
    if y0_ref == y1_ref or y0 == y1:
        raise ValueError("y inputs must have different values for 0 and 1")

    offsets = [x["offset"] for x in stops]
    if min(offsets) != 0 or max(offsets) != 1:
        raise ValueError("stops must exactly cover the interval [0, 1]")
    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("stops must be sorted from 0 to 1")

    # Convert to RGB tuples
    colors = [_hex_to_rgb(x["color"]) for x in stops]

    # Interpolation function for RGB channels
    def interpolate_color(off: float) -> tuple[float, float, float]:
        if off <= 0:
            return colors[0]
        if off >= 1:
            return colors[-1]
        # stops is typically very short (< 10), so just do a linear scan.
        for i in range(1, len(stops)):
            left = stops[i - 1]["offset"]
            right = stops[i]["offset"]
            if left <= off <= right:
                t = (off - left) / (right - left)
                c0 = colors[i - 1]
                c1 = colors[i]
                return tuple(c0[j] + t * (c1[j] - c0[j]) for j in range(3))
        raise AssertionError("Should never get here")

    # Generate new stops at desired offsets (still [0, 1])
    new_stops = []
    for offset in offsets:
        # Map this new offset back to the actual y value
        y = y0 + offset * (y1 - y0)
        # Find the corresponding offset in the original gradient
        orig_offset = (y - y0_ref) / (y1_ref - y0_ref)
        # Get interpolated color
        rgb = interpolate_color(orig_offset)
        new_stops.append(
            {
                "offset": offset,
                "color": _rgb_to_hex(rgb),
            }
        )

    return new_stops


def daily_temp_precip_chart(
    df: pd.DataFrame, from_date: datetime.datetime, station_abbr: str
) -> dict[str, Any]:
    """
    Generates a combined, layered chart of hourly precipitation and temperature.

    This chart displays:
    - Precipitation as bars spanning the hour they represent.
    - Temperature as a line chart with points at the half-hour mark.
    - A shared, interactive legend to filter the series.
    - Independent Y-axes for precipitation and temperature.

    Args:
        df: DataFrame with a DatetimeIndex and columns 'station_abbr',
            TEMP_HOURLY_MEAN, and PRECIP_HOURLY_MM.
        from_date: The date for which the chart is generated (used for the title).
        station_abbr: The station abbreviation to filter/validate the data.

    Returns:
        A dictionary representing the Altair chart specification.
    """
    # --- 1. Data Validation and Preparation ---
    if not (df["station_abbr"] == station_abbr).all():
        raise ValueError(f"Not all rows are for station {station_abbr}")
    if df.empty:
        raise NoDataError(f"No precipitation data for {station_abbr}")

    # Work with a copy to avoid modifying the original DataFrame
    df_prep = df.copy().rename(columns=VEGA_LEGEND_LABELS)

    # The key is to create explicit start, end, and midpoint columns for our intervals.
    # We convert the index to timezone-naive 'Europe/Zurich' time as Vega/Altair
    # work best with naive datetimes, avoiding browser-local adjustments.
    df_prep["time_end"] = df_prep.index.tz_convert("Europe/Zurich").tz_localize(None)
    df_prep["time_start"] = df_prep["time_end"] - pd.Timedelta(hours=1)
    df_prep["time_midpoint"] = df_prep["time_start"] + pd.Timedelta(minutes=30)

    # Create a dedicated tooltip string for the intervals.
    df_prep["tooltip_interval"] = (
        df_prep["time_start"].dt.strftime("%H:%M")
        + " - "
        + df_prep["time_end"].dt.strftime("%H:%M")
    )

    # Melt the dataframe into a long format. This is a robust pattern for
    # creating charts with a shared legend and color scheme.
    id_vars = ["time_start", "time_end", "time_midpoint", "tooltip_interval"]
    value_vars = ["precip mm", "temp mean"]
    long_df = df_prep.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Measurement",
        value_name="Value",
    )

    # --- 2. Chart Configuration ---

    # Create an interactive legend that allows filtering by clicking on items.
    highlight = alt.selection_point(fields=["Measurement"], bind="legend")

    # Define a consistent color scheme for the variables.
    color_scale = alt.Scale(
        domain=["precip mm", "temp mean"], range=["steelblue", "firebrick"]
    )

    # Create a base chart to share data. Properties will be set on the final layered chart.
    base = alt.Chart(long_df)

    # --- 3. Create Chart Layers ---

    # y-axis lower and upper bounds. Used for both the
    # gradient "background" area and the temperature plot.
    # Always show 0, and at least a 30 degree interval.
    temp_ymin = min(df_prep["temp mean"].min() - 2, 0)
    temp_ymax = max(temp_ymin + 30, df_prep["temp mean"].max() + 2)

    # Layer 0: Gradient background
    # Constant data to get an "area" plot extending across the full range.
    gradient_df = pd.DataFrame(
        [
            {
                "gradient_time": df_prep["time_start"].min(),
                "gradient_min": temp_ymin,
                "gradient_max": temp_ymax,
            },
            {
                "gradient_time": df_prep["time_end"].max(),
                "gradient_min": temp_ymin,
                "gradient_max": temp_ymax,
            },
        ]
    )
    # Red-yellow-blue color gradient for the gradient area fill, from
    # https://vega.github.io/vega/docs/schemes/#redyellowblue
    gradient_stops = [
        {"offset": 0, "color": "#a50026"},
        {"offset": 0.1, "color": "#d4322c"},
        {"offset": 0.2, "color": "#f16e43"},
        {"offset": 0.3, "color": "#fcac64"},
        {"offset": 0.4, "color": "#fedd90"},
        {"offset": 0.5, "color": "#faf8c1"},
        {"offset": 0.6, "color": "#dcf1ec"},
        {"offset": 0.7, "color": "#abd6e8"},
        {"offset": 0.8, "color": "#75abd0"},
        {"offset": 0.9, "color": "#4a74b4"},
        {"offset": 1, "color": "#313695"},
    ]
    # Re-normalize s.t. stops 0..1 has colors for temp_ymax..temp_ymin,
    # assuming the original gradient defines colors for 25..-5 degrees C.
    gradient_stops = _renormalize_stops(gradient_stops, 25, -5, temp_ymax, temp_ymin)
    gradient_color = alt.Gradient(
        # All coordinates are defined in a normalized [0, 1] coordinate space,
        # relative to the bounding box of the item being colored.
        x1=1,
        y1=0,
        x2=1,
        y2=1,
        gradient="linear",
        stops=gradient_stops,
    )
    gradient_area = (
        alt.Chart(gradient_df)
        .mark_area(color=gradient_color)
        .encode(
            x=alt.X("gradient_time:T"),
            y=alt.Y("gradient_min:Q", title=None)
            # Force axis to be on the left, so it matches the location of the temperature axis.
            .axis(orient="left").scale(domain=[temp_ymin, temp_ymax]),
            y2=alt.Y2("gradient_max"),
            opacity=alt.value(0.35),  # 0.35 "looks good" for a background.
        )
    )

    # Layer 1: Precipitation Bars
    # We filter the long-form data to only include precipitation.
    precip_ymax = max(5, df_prep["precip mm"].max() + 1)
    precip_bars = (
        base.transform_filter(alt.datum.Measurement == "precip mm")
        .mark_bar()
        .encode(
            # Use 'time_start:T' for x and 'time_end:T' for x2 to define the bar's width.
            x=alt.X(
                "time_start:T",
                title="Hour of Day",
                axis=alt.Axis(format="%H:%M", labelAngle=0),
            ),
            x2=alt.X2("time_end:T"),
            # Use y for the top of the bar and y2=alt.value(0) to anchor the bottom to the baseline.
            y=alt.Y("Value:Q", title="Precipitation (mm)").scale(
                domain=[0, precip_ymax]
            ),
            y2=alt.value(0),
            color=alt.Color("Measurement:N", scale=color_scale),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("tooltip_interval:N", title="Interval"),
                alt.Tooltip("Value:Q", title="Precipitation", format=".1f"),
            ],
        )
    )

    # Layer 2: Temperature Line
    # Filter data to only include temperature.
    temp_line = (
        base.transform_filter(alt.datum.Measurement == "temp mean")
        .mark_line(interpolate="cardinal")
        .encode(
            # The temperature is a point measurement, so we only need 'x'.
            # We plot it at the midpoint of the hour interval.
            x=alt.X("time_midpoint:T", title="Hour of Day"),
            y=alt.Y("Value:Q", title="Temperature (°C)").scale(
                domain=[temp_ymin, temp_ymax]
            )
            # Force axis to be on the left to match the side of the gradient area.
            .axis(orient="left"),
            color=alt.Color("Measurement:N", scale=color_scale),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.2)),
        )
    )

    # Layer 3: Temperature Dots
    # Add points to the line for better visibility and tooltip interaction.
    temp_dots = temp_line.mark_point(filled=True, size=60).encode(
        tooltip=[
            alt.Tooltip("tooltip_interval:N", title="Interval"),
            alt.Tooltip("Value:Q", title="Temperature", format=".1f"),
        ]
    )

    # --- 4. Combine Layers and Finalize Chart ---
    date_str = from_date.strftime("%a, %d %b %Y")
    chart = (
        alt.layer(
            # gradient_area must come first, as the background.
            # precip_bars must come before temperature, so the temp. line
            # is not hidden.
            gradient_area,
            precip_bars,
            alt.layer(temp_line, temp_dots),
        )
        .resolve_scale(
            y="independent"  # Allow precipitation and temp to have separate y-axis scales
        )
        .add_params(highlight)
        .properties(
            width="container",
            autosize={"type": "fit", "contains": "padding"},
            title=f"Daily temperature and precipitation on {date_str}",
        )
    )

    return chart.to_dict()


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
    elif chart_type == "rainiest_day":
        return rainiest_day_chart(df, station_abbr=station_abbr, period=period)
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

    if not temp_dm.empty:
        try:
            coeffs, _ = polyfit_columns(df_m[[db.TEMP_DAILY_MEAN]], deg=1)
            annual_temp_increase = coeffs[db.TEMP_DAILY_MEAN].iloc[1]
        except ValueError:
            annual_temp_increase = None  # Could not fit a curve

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
                display_name=VEGA_LEGEND_LABELS.get(c, c),
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
