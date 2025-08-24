import datetime
import pytest
import sqlalchemy as sa

from service.charts.db import schema as ds
from service.charts.db import constants as dc
from service.charts import db

from . import ogd


# A list of test cases, each with a URL and the expected result object.
TEST_CASES = [
    (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_m.csv",
        ogd.HrefMatch(interval="m", frequency="historical"),
    ),
    (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_d_historical.csv",
        ogd.HrefMatch(interval="d", frequency="historical"),
    ),
    (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_d_recent.csv",
        ogd.HrefMatch(interval="d", frequency="recent"),
    ),
    (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_h_historical_2010-2019.csv",
        ogd.HrefMatch(interval="h", frequency="historical", years=(2010, 2019)),
    ),
]


@pytest.mark.parametrize("url, expected_match", TEST_CASES)
def test_match_csv_resource(url, expected_match):
    """
    Tests that match_csv_resource correctly parses various URL formats.
    """
    match = ogd.match_csv_resource(url, None)

    assert match is not None
    assert match == expected_match


def test_match_csv_resource_filter():
    """
    Tests that match_csv_resource correctly parses various URL formats.
    """
    # "ber" is not a substring, so the match should fail.
    match = ogd.match_csv_resource(
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_h_historical_2010-2019.csv",
        "ber",
    )

    assert match is None


############################################################
# Integration tests
############################################################


@pytest.mark.integration
def test_integration_import_ogd_data(tmp_path):
    engine = sa.create_engine("sqlite:///:memory:")
    ds.metadata.create_all(engine)

    base_dir = tmp_path / "data"
    base_dir.mkdir()

    insert_mode = "insert_missing"
    csv_filter = "_ber_"

    ogd.run_import(
        engine=engine,
        base_dir=str(base_dir),
        insert_mode=insert_mode,
        force_update=False,
        csv_filter=csv_filter,
    )

    with engine.begin() as conn:
        df_daily = db.read_daily_measurements(
            conn, "BER", from_date=datetime.date(2020, 1, 1)
        )
    # Expect rows
    assert len(df_daily) > 100
    # Expect columns
    assert set(
        [dc.TEMP_DAILY_MAX, dc.PRECIP_DAILY_MM, dc.SUNSHINE_DAILY_MINUTES]
    ).issubset(set(df_daily.columns))
    # Expect no row has only empty data
    assert (
        (df_daily[ds.TABLE_DAILY_MEASUREMENTS.measurements] > 0).any(axis=1).all()
    ), "Found measurement rows with only zero values"
