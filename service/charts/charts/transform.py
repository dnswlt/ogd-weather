import numpy as np
import pandas as pd

from service.charts.db import constants as dc


# Column names for "timeless" measurements, i.e. where the
# time dimension ("daily", "monthly", etc.) is dropped.
# These names are useful for functions that deal with data
# of multiple time granularities without having to case switch
# on the time dimension all the time.
TEMP_MIN = "temp_min"
TEMP_MEAN_OF_DAILY_MIN = "temp_mean_of_daily_min"
TEMP_MEAN = "temp_mean"
TEMP_MAX = "temp_max"
TEMP_MEAN_OF_DAILY_MAX = "temp_mean_of_daily_max"
PRECIP_MM = "precip_mm"
SUMMER_DAYS = "summer_days"
FROST_DAYS = "frost_days"
ICE_DAYS = "ice_days"
TROPICAL_NIGHTS = "tropical_nights"
HEAT_DAYS = "heat_days"


def annual_agg(df, func):
    """Returns a DataFrame with one row per year containing average values."""
    df_y = df.groupby(df.index.year).agg(func)
    df_y.index.name = "year"
    return df_y


def long_format(df: pd.DataFrame, id_cols: list[str] | None = None) -> pd.DataFrame:
    index_name = df.index.name
    return df.reset_index().melt(
        id_vars=[index_name] + (id_cols or []),
        var_name="measurement",
        value_name="value",
    )


def rolling_mean(df: pd.DataFrame, window: int):
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


def timeline_years_chart_data(
    df: pd.DataFrame, agg_func, window: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Prepares long data for a timeline chart with a trendline.

    Both returned data frames use a "year":int index.

    Args:
        df: The (possibly multi-column) input data. Must have a datetime index.
        agg_func: the aggregation function to use, e.g. "sum", or np.sum.
        window: the sliding window of years to smooth data. 1 means no smoothing.
    Returns:
        The long-format data aggregated from df.
        The long-format trendline.
    """

    data = annual_agg(df, agg_func)
    if window > 1:
        # Rolling window for smoothing the data.
        data = rolling_mean(data, window)
    data_long = long_format(data).dropna()

    try:
        _, trend = polyfit_columns(data, deg=1)
        trend_long = long_format(trend).dropna()
    except ValueError:
        trend_long = None

    return data_long, trend_long


def timeless_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns in df from database column names to their matching timeless names."""
    DISPLAY_COLUMN_NAMES = {
        dc.TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MIN: TEMP_MEAN_OF_DAILY_MIN,
        dc.TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MIN: TEMP_MEAN_OF_DAILY_MIN,
        dc.TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MAX: TEMP_MEAN_OF_DAILY_MAX,
        dc.TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MAX: TEMP_MEAN_OF_DAILY_MAX,
        dc.TEMP_HOM_ANNUAL_MEAN: TEMP_MEAN,
        dc.TEMP_HOM_MONTHLY_MEAN: TEMP_MEAN,
        dc.PRECIP_HOM_MONTHLY_MM: PRECIP_MM,
        dc.PRECIP_HOM_ANNUAL_MM: PRECIP_MM,
        dc.SUMMER_DAYS_HOM_MONTHLY_COUNT: SUMMER_DAYS,
        dc.SUMMER_DAYS_HOM_ANNUAL_COUNT: SUMMER_DAYS,
        dc.FROST_DAYS_HOM_MONTHLY_COUNT: FROST_DAYS,
        dc.FROST_DAYS_HOM_ANNUAL_COUNT: FROST_DAYS,
        dc.ICE_DAYS_HOM_MONTHLY_COUNT: ICE_DAYS,
        dc.ICE_DAYS_HOM_ANNUAL_COUNT: ICE_DAYS,
        dc.TROPICAL_NIGHTS_HOM_MONTHLY_COUNT: TROPICAL_NIGHTS,
        dc.TROPICAL_NIGHTS_HOM_ANNUAL_COUNT: TROPICAL_NIGHTS,
        dc.HEAT_DAYS_HOM_MONTHLY_COUNT: HEAT_DAYS,
        dc.HEAT_DAYS_HOM_ANNUAL_COUNT: HEAT_DAYS,
    }
    return df.rename(columns=DISPLAY_COLUMN_NAMES)
