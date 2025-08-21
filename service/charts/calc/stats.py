import pandas as pd


from service.charts import models
from service.charts.db import constants as dc


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
