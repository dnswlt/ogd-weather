import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from service.charts.db import constants as dc

from . import atm


def test_atm_pressure_z850():
    here = Path(__file__).resolve().parent
    csv_file = here.joinpath("../testdata/ogd-smn_vis_h_recent.csv")
    df = pd.read_csv(
        csv_file,
        sep=";",
        encoding="cp1252",
    )
    p_qfe = dc.ATM_PRESSURE_HOURLY_MEAN
    p_z850 = "ppz850h0"

    H_stn = 639  # station elevation
    G = 9.81  # Acceleration due to gravity
    R = 287.05  # specific gas constant for dry air
    K_mean = df[dc.TEMP_HOURLY_MEAN] + 273.15  # mean hourly temp in Kelvin

    z850 = df[p_z850]
    qfe_approx = 850 * np.exp((G * (z850 - H_stn)) / (R * K_mean))

    err_abs = np.abs(df[p_qfe] - qfe_approx)

    assert err_abs.max() < 1  # Not more than 1 hPa absolute error
    assert err_abs.std() < 0.25  # With low variance
    assert err_abs.mean() < 0.6  # Not more than 0.6 hPa mean abs error


def test_atm_pressure_qff():
    # This test validates that our formulas for deriving the
    # QFE (station level) air pressure from given QFF (sea level) values
    # yield accurate values.
    here = Path(__file__).resolve().parent
    csv_file = here.joinpath("../testdata/ogd-smn_ber_h_recent.csv")
    df = pd.read_csv(
        csv_file,
        sep=";",
        encoding="cp1252",
    )
    p_qfe = dc.ATM_PRESSURE_HOURLY_MEAN
    p_qff = "pp0qffh0"

    H_stn = 553  # station elevation
    G = 9.81  # Acceleration due to gravity
    R = 287.05  # specific gas constant for dry air
    qff = df[p_qff]
    K_mean = df[dc.TEMP_HOURLY_MEAN] + 273.15  # mean hourly temp in Kelvin
    qfe_approx = np.round(qff * np.exp(-(H_stn * G) / (R * K_mean)), 1)

    err_abs = np.abs(df[p_qfe] - qfe_approx)

    assert err_abs.max() < 1  # Not more than 1 hPa absolute error
    assert err_abs.std() < 0.1  # With low variance
    assert err_abs.mean() < 0.6  # Not more than 0.6 hPa mean abs error


def test_pressure_normals_qfe():
    temp = pd.Series(np.arange(-10, 30, 1))
    date = datetime.datetime(2025, 1, 1, 0, 0, 0)
    height = 500

    pn = atm.pressure_normals_qfe(temp, height, date)

    assert list(pn.columns) == ["low", "high"]
    low = pn["low"]
    high = pn["high"]

    assert ((low > 930) & (low < 950)).all()
    assert ((high > 960) & (high < 980)).all()


def test_pressure_normals_z850():
    temp = pd.Series(np.arange(-10, 30, 1))
    date = datetime.datetime(2025, 7, 1, 0, 0, 0)
    height = 1500

    pn = atm.pressure_normals_qfe(temp, height, date)

    assert list(pn.columns) == ["low", "high"]
    low = pn["low"]
    high = pn["high"]

    assert ((low > 847) & (low < 849)).all()
    assert ((high > 856) & (high < 858)).all()


def test_pressure_normals_z700():
    temp = pd.Series(np.arange(-10, 30, 1))
    date = datetime.datetime(2025, 7, 1, 0, 0, 0)
    height = 3000

    pn = atm.pressure_normals_qfe(temp, height, date)

    assert list(pn.columns) == ["low", "high"]
    low = pn["low"]
    high = pn["high"]

    assert ((low > 699) & (low < 701)).all()
    assert ((high > 709) & (high < 711)).all()
