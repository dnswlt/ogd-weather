import pandas as pd
from pydantic import BaseModel


from service.charts import models
from service.charts.db import constants as dc

from service.charts.charts import transform as tf


class PerStationData(BaseModel):
    station: models.Station
    daily_measurements: pd.DataFrame
    daily_manual_measurements: pd.DataFrame
    monthly_wind_stats: pd.DataFrame

    class Config:
        # Needed for pd.DataFrame fields.
        arbitrary_types_allowed = True


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
                display_name=c,
                dtype=str(df[c].dtype),
            )
            for c in measurements.columns
        ],
    )


def year_highlights(
    df: pd.DataFrame, station_abbr: str, year: int
) -> models.StationYearHighlights:

    # Temp. range
    td = df[dc.TEMP_DAILY_MAX] - df[dc.TEMP_DAILY_MIN]
    max_daily_temp_range_date = td.idxmax()
    max_daily_temp_range_min = df.loc[max_daily_temp_range_date, dc.TEMP_DAILY_MIN]
    max_daily_temp_range_max = df.loc[max_daily_temp_range_date, dc.TEMP_DAILY_MAX]

    # Frost days
    fd = df[dc.TEMP_DAILY_MIN]
    fd = fd[fd < 0]

    h1 = fd[fd.index.month <= 6]
    h2 = fd[fd.index.month > 6]

    if not h1.empty:
        last_frost_day = h1.idxmax()
    else:
        last_frost_day = None

    if not h2.empty:
        first_frost_day = h2.idxmax()
    else:
        first_frost_day = None

    # Sunshine hours
    sh = df[dc.SUNSHINE_DAILY_MINUTES] / 60.0
    max_daily_sunshine_hours_date = sh.idxmax()
    max_daily_sunshine_hours = sh.loc[max_daily_sunshine_hours_date]

    # Snow depth
    sd = df[dc.SNOW_DEPTH_DAILY_CM].dropna()
    if len(sd) > 0:
        max_snow_depth_cm = sd.max()
        snow_days = (sd >= 1.0).sum()
    else:
        max_snow_depth_cm = None
        snow_days = None

    return models.StationYearHighlights(
        first_frost_day=first_frost_day,
        last_frost_day=last_frost_day,
        max_daily_temp_range_min=max_daily_temp_range_min,
        max_daily_temp_range_max=max_daily_temp_range_max,
        max_daily_temp_range_date=max_daily_temp_range_date,
        max_daily_sunshine_hours=max_daily_sunshine_hours,
        max_daily_sunshine_hours_date=max_daily_sunshine_hours_date,
        snow_days=snow_days,
        max_snow_depth_cm=max_snow_depth_cm,
    )


def _to_wind_stats(df: pd.DataFrame) -> models.WindStats | None:
    if df.empty:
        return None
    value_count = df[dc.DX_VALUE_COUNT]
    if value_count.sum() == 0:
        return None

    gust_factor = (
        df[dc.DX_GUST_FACTOR_MONTHLY_MEAN] * value_count
    ).sum() / value_count.sum()

    total_count = df[dc.DX_WIND_DIR_TOTAL_COUNT].sum()
    if total_count > 0:
        dirs = list(dc.DX_WIND_DIR_COUNT_MAP.keys())
        dir_cols = list(dc.DX_WIND_DIR_COUNT_MAP.values())
        dir_sums = df[dir_cols].sum()
        max_dir_col = dir_sums.idxmax()
        main_wind_dir = dirs[dir_cols.index(max_dir_col)]
        wind_dir_pct = {
            d: df[c].sum() / total_count * 100
            for d, c in dc.DX_WIND_DIR_COUNT_MAP.items()
        }
    else:
        main_wind_dir = None
        wind_dir_pct = None

    days_y = (
        df[
            [
                dc.DX_MODERATE_BREEZE_DAYS_MONTHLY_COUNT,
                dc.DX_STRONG_BREEZE_DAYS_MONTHLY_COUNT,
            ]
        ]
        .groupby(df.index.year)
        .agg("sum")
    ).mean()

    return models.WindStats(
        moderate_breeze_days=float(days_y[dc.DX_MODERATE_BREEZE_DAYS_MONTHLY_COUNT]),
        strong_breeze_days=float(days_y[dc.DX_STRONG_BREEZE_DAYS_MONTHLY_COUNT]),
        gust_factor=float(gust_factor),
        main_wind_dir=main_wind_dir,
        wind_dir_percent=wind_dir_pct,
        measurement_count=total_count,
    )


def compare_stations(
    per_station_data: list[PerStationData],
) -> models.StationComparisonData:

    def nn(f):
        if pd.isna(f):
            return None
        return f

    stations = [d.station for d in per_station_data]
    dfs = [d.daily_measurements for d in per_station_data]
    dfs_manual = [d.daily_manual_measurements for d in per_station_data]
    dfs_wind = [d.monthly_wind_stats for d in per_station_data]
    rows = []

    # Station elevation (m a.s.l.)
    heights_masl = [s.height_masl for s in stations]
    rows.append(
        models.StationComparisonRow(
            label="Station elevation (m a.s.l.)",
            values=heights_masl,
            lower_bound=0,
            upper_bound=1500,
        )
    )

    # Mean daily max. temperature
    mdx_temps = [nn(df[dc.TEMP_DAILY_MAX].mean()) for df in dfs]
    rows.append(
        models.StationComparisonRow(
            label="Avg. daily max. temperature (°C)",
            values=mdx_temps,
            lower_bound=0,
        )
    )

    # Mean daily temperature variation
    d_temps = [nn((df[dc.TEMP_DAILY_MAX] - df[dc.TEMP_DAILY_MIN]).mean()) for df in dfs]
    rows.append(
        models.StationComparisonRow(
            label="Avg. daily temp. range (∆ °C)",
            values=d_temps,
            lower_bound=0,
            upper_bound=10,
        )
    )

    # Min. temperature measured in period
    min_temps = [nn(df[dc.TEMP_DAILY_MIN].min()) for df in dfs]

    rows.append(
        models.StationComparisonRow(
            label="Min. temperature (°C)",
            values=min_temps,
            lower_bound=-25,
            upper_bound=0,
        )
    )

    # Max. temperature measured in period
    max_temps = [nn(df[dc.TEMP_DAILY_MAX].max()) for df in dfs]

    rows.append(
        models.StationComparisonRow(
            label="Max. temperature (°C)",
            values=max_temps,
            lower_bound=20,
            upper_bound=40,
        )
    )

    # Avg. number of frost days (< 0°C)
    frost_days = []
    for df in dfs:
        days = pd.DataFrame({"days": (df[dc.TEMP_DAILY_MIN] < 0).astype(int)})
        days_y = tf.annual_agg(days, "sum")
        frost_days.append(days_y["days"].mean())

    rows.append(
        models.StationComparisonRow(
            label="Avg. number of frost days (min. < 0 °C)",
            values=frost_days,
            lower_bound=0,
        )
    )

    # Avg. number of summer days (≥ 25°C)
    summer_days = []
    for df in dfs:
        days = pd.DataFrame({"days": (df[dc.TEMP_DAILY_MAX] >= 25).astype(int)})
        days_y = tf.annual_agg(days, "sum")
        summer_days.append(days_y["days"].mean())

    rows.append(
        models.StationComparisonRow(
            label="Avg. number of summer days (max. ≥ 25 °C)",
            values=summer_days,
            lower_bound=0,
        )
    )

    # Avg. number of tropical nights (min ≥ 20°C)
    tropical_nights = []
    for df in dfs:
        days = pd.DataFrame({"days": (df[dc.TEMP_DAILY_MIN] >= 20).astype(int)})
        days_y = tf.annual_agg(days, "sum")
        tropical_nights.append(days_y["days"].mean())

    rows.append(
        models.StationComparisonRow(
            label="Avg. number of tropical nights (min. ≥ 20 °C)",
            values=tropical_nights,
            lower_bound=0,
        )
    )

    # Avg. number of daily sunshine hours
    sunshine_hours = [nn(df[dc.SUNSHINE_DAILY_MINUTES].mean() / 60) for df in dfs]
    rows.append(
        models.StationComparisonRow(
            label="Avg. daily hours of sunshine (h)",
            values=sunshine_hours,
            lower_bound=0,
            upper_bound=6,
        )
    )

    # Avg. total annual precipitation
    precip_means = []
    for df in dfs:
        df_p = tf.annual_agg(df[[dc.PRECIP_DAILY_MM]], "sum")
        precip_means.append(nn(df_p[dc.PRECIP_DAILY_MM].mean()))

    rows.append(
        models.StationComparisonRow(
            label="Avg. annual precipitation (mm)",
            values=precip_means,
            lower_bound=0,
        )
    )

    # Avg. number of rain days (≥ 1 mm)
    rain_days = []
    for df in dfs:
        days = pd.DataFrame({"days": (df[dc.PRECIP_DAILY_MM] >= 1).astype(int)})
        days_y = tf.annual_agg(days, "sum")
        rain_days.append(days_y["days"].mean())

    rows.append(
        models.StationComparisonRow(
            label="Avg. number of rain days (≥ 1 mm precip.)",
            values=rain_days,
            lower_bound=0,
        )
    )

    # Mean atmospheric pressure (QFE)
    atm_p = [nn(df[dc.ATM_PRESSURE_DAILY_MEAN].mean()) for df in dfs]
    rows.append(
        models.StationComparisonRow(
            label="Avg. atmospheric pressure (hPa)",
            values=atm_p,
            lower_bound=850,
            upper_bound=1014,
        )
    )

    # Snow days (≥ 1cm snow depth)
    snow_days = []
    for df in dfs_manual:
        if len(df) == 0:
            # No manual measurement data for this station.
            snow_days.append(None)
            continue
        days = pd.DataFrame({"days": (df[dc.SNOW_DEPTH_MAN_DAILY_CM] >= 1).astype(int)})
        days_y = tf.annual_agg(days, "sum")
        snow_days.append(days_y["days"].mean())

    rows.append(
        models.StationComparisonRow(
            label="Avg. number of snow days (≥ 1 cm snow depth)",
            values=snow_days,
            lower_bound=0,
        )
    )

    # Wind
    wind_stats = [_to_wind_stats(df) for df in dfs_wind]
    moderate_breeze_days = [
        ws.moderate_breeze_days if ws else None for ws in wind_stats
    ]
    rows.append(
        models.StationComparisonRow(
            label="Avg. days with Moderate Breeze hour (≥ 20 km/h)",
            values=moderate_breeze_days,
            lower_bound=0,
        )
    )
    strong_breeze_days = [ws.strong_breeze_days if ws else None for ws in wind_stats]
    rows.append(
        models.StationComparisonRow(
            label="Avg. days with Strong Breeze gusts (≥ 39 km/h)",
            values=strong_breeze_days,
            lower_bound=0,
        )
    )

    gust_factors = [ws.gust_factor if ws else None for ws in wind_stats]
    rows.append(
        models.StationComparisonRow(
            label="Avg. hourly gust factor",
            values=gust_factors,
            lower_bound=1,
        )
    )

    return models.StationComparisonData(
        stations=stations, rows=rows, wind_stats=wind_stats
    )
