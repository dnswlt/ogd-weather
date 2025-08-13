import pandas as pd


from . import db
from . import models


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
    td = df[db.TEMP_DAILY_MAX] - df[db.TEMP_DAILY_MIN]
    max_daily_temp_range_date = td.idxmax()
    max_daily_temp_range = td.loc[max_daily_temp_range_date]

    # Frost days
    fd = df[db.TEMP_DAILY_MIN]
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
    sh = df[db.SUNSHINE_DAILY_MINUTES] / 60.0
    max_daily_sunshine_hours_date = sh.idxmax()
    max_daily_sunshine_hours = sh.loc[max_daily_sunshine_hours_date]

    # Snow depth
    sd = df[db.SNOW_DEPTH_DAILY_CM].dropna()
    if len(sd) > 0:
        max_snow_depth_cm = sd.max()
        snow_days = (sd > 1.0).sum()
    else:
        max_snow_depth_cm = None
        snow_days = None

    return models.StationYearHighlights(
        first_frost_day=first_frost_day,
        last_frost_day=last_frost_day,
        max_daily_temp_range=max_daily_temp_range,
        max_daily_temp_range_date=max_daily_temp_range_date,
        max_daily_sunshine_hours=max_daily_sunshine_hours,
        max_daily_sunshine_hours_date=max_daily_sunshine_hours_date,
        snow_days=snow_days,
        max_snow_depth_cm=max_snow_depth_cm,
    )
