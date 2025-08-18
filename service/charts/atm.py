import datetime
import numpy as np
import pandas as pd


def _normal_range_qff(date: datetime.datetime) -> tuple[float, float]:
    if date.month in (12, 1, 2):
        return (1000, 1035)
    elif date.month in (6, 7, 8):
        return (1010, 1025)
    else:
        # Spring, Autumn: coarse linear interpolation
        return (1005, 1030)


def _normal_range_z850(date: datetime.datetime) -> tuple[float, float]:
    if date.month in (12, 1, 2):
        return (1400, 1520)
    elif date.month in (6, 7, 8):
        return (1480, 1570)
    else:
        # Spring, Autumn: coarse linear interpolation
        return (1440, 1545)


def _normal_range_z700(date: datetime.datetime) -> tuple[float, float]:
    if date.month in (12, 1, 2):
        return (2900, 3040)
    elif date.month in (6, 7, 8):
        return (3000, 3120)
    else:
        # Spring, Autumn: coarse linear interpolation
        return (2950, 3080)


def pressure_normals_qfe(
    temp_mean: pd.Series,
    height: float,
    date: datetime.datetime,
) -> pd.DataFrame:
    """Calculates the "normal" atmospheric pressure band based on the given parameters.

    The returned values are expressed in atmospheric pressure at barometric altitude (QFE).
    The bands considered normal are defined in the _normal_range_* functions above.

    Returns:
        a DataFrame with two columns: ["low", "high"].
    """

    G = 9.81  # Acceleration due to gravity
    R = 287.05  # specific gas constant for dry air
    H = height  # station elevation
    K_mean = temp_mean + 273.15  # mean hourly temp in Kelvin

    if height < 600:
        # Summer: 1010 - 1025
        # Winter normals: 1000 - 1035
        low, high = _normal_range_qff(date)
        F = np.exp(-(H * G) / (R * K_mean))
        return pd.DataFrame(
            {
                "low": np.round(low * F, 1),
                "high": np.round(high * F, 1),
            }
        )

    if height < 2250:
        Z = 850
        low, high = _normal_range_z850(date)
    else:
        Z = 700
        low, high = _normal_range_z700(date)

    return pd.DataFrame(
        {
            "low": Z * np.exp((G * (low - H)) / (R * K_mean)),
            "high": Z * np.exp((G * (high - H)) / (R * K_mean)),
        }
    )
