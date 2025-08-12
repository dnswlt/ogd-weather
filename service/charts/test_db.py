import os
import unittest
import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from . import db
from . import models
from .errors import StationNotFoundError
from .testhelpers import PandasTestCase


def _testdata_dir():
    return os.path.join(os.path.dirname(__file__), "testdata")


class TestDb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sa.create_engine("sqlite:///:memory:", future=True)

        # Create all tables for tests
        db.metadata.create_all(cls.engine)


class TestDbStations(TestDb):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._insert_station_test_data()

    def setUp(self):
        self.conn = self.engine.connect()

    def tearDown(self):
        self.conn.close()

    @classmethod
    def _insert_station_test_data(cls):
        stations_data = [
            {
                "station_abbr": "ABO",
                "station_name": "Arosa",
                "station_canton": "GR",
                "tre200d0_min_date": "2000-01-01",
                "tre200d0_max_date": "2020-12-31",
                "rre150d0_min_date": "2000-01-01",
                "rre150d0_max_date": "2020-12-31",
                "tre200d0_count": 100,
                "rre150d0_count": 100,
                "station_exposition_de": "Ebene",
                "station_exposition_fr": "Plaine",
                "station_exposition_it": "Pianura",
                "station_exposition_en": "plain",
            },
            {
                "station_abbr": "BAS",
                "station_name": "Basel / Binningen",
                "station_canton": "BL",
                "tre200d0_min_date": "1990-01-01",
                "tre200d0_max_date": "2010-12-31",
                "rre150d0_min_date": None,
                "rre150d0_max_date": None,
                "tre200d0_count": 50,
                "rre150d0_count": 0,
                "station_exposition_de": "Ebene",
                "station_exposition_fr": None,
                "station_exposition_it": None,
                "station_exposition_en": None,
            },  # Missing precip dates
            {
                "station_abbr": "BER",
                "station_name": "Bern / Zollikofen",
                "station_canton": "BE",
                "tre200d0_min_date": None,
                "tre200d0_max_date": None,
                "rre150d0_min_date": "1980-01-01",
                "rre150d0_max_date": "2022-12-31",
                "tre200d0_count": 0,
                "rre150d0_count": 50,
                "station_exposition_de": "Ebene",
                "station_exposition_fr": "Plaine",
                "station_exposition_it": "Pianura",
                "station_exposition_en": "plain",
            },  # Missing temp dates
            {
                "station_abbr": "LUG",
                "station_name": "Lugano",
                "station_canton": "TI",
                "tre200d0_min_date": None,
                "tre200d0_max_date": None,
                "rre150d0_min_date": None,
                "rre150d0_max_date": None,
                "tre200d0_count": 0,
                "rre150d0_count": 0,
                "station_exposition_de": None,
                "station_exposition_fr": None,
                "station_exposition_it": None,
                "station_exposition_en": None,
            },  # All dates missing
        ]

        with cls.engine.begin() as conn:
            conn.execute(
                sa.insert(db.sa_table_x_station_data_summary),
                stations_data,
            )

    def test_read_station_found(self):
        station = db.read_station(self.conn, "ABO")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "ABO")
        self.assertEqual(station.name, "Arosa")
        self.assertEqual(station.canton, "GR")
        self.assertEqual(station.precipitation_min_date, datetime.date(2000, 1, 1))
        self.assertEqual(station.precipitation_max_date, datetime.date(2020, 12, 31))

    def test_read_station_not_found(self):
        with self.assertRaises(StationNotFoundError) as cm:
            db.read_station(self.conn, "XYZ")
        self.assertEqual(str(cm.exception), "No station found with abbr=XYZ")

    def test_read_station_partial_dates(self):
        station = db.read_station(self.conn, "BAS")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "BAS")
        self.assertEqual(station.temperature_min_date, datetime.date(1990, 1, 1))
        self.assertEqual(station.temperature_max_date, datetime.date(2010, 12, 31))

        station = db.read_station(self.conn, "BER")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "BER")
        self.assertEqual(station.precipitation_min_date, datetime.date(1980, 1, 1))
        self.assertEqual(station.precipitation_max_date, datetime.date(2022, 12, 31))

    def test_read_station_no_dates(self):
        station = db.read_station(self.conn, "LUG")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "LUG")
        self.assertIsNone(station.temperature_min_date)
        self.assertIsNone(station.temperature_max_date)
        self.assertIsNone(station.precipitation_min_date)
        self.assertIsNone(station.precipitation_max_date)

    def test_read_station_localized_string(self):
        station = db.read_station(self.conn, "BER")
        self.assertIsNotNone(station.exposition)
        self.assertEqual(station.exposition.de, "Ebene")
        self.assertEqual(station.exposition.fr, "Plaine")
        self.assertEqual(station.exposition.it, "Pianura")
        self.assertEqual(station.exposition.en, "plain")

    def test_read_station_localized_string_empty(self):
        station = db.read_station(self.conn, "BAS")
        self.assertIsNotNone(station.exposition)
        self.assertEqual(station.exposition.de, "Ebene")
        self.assertEqual(station.exposition.fr, "")
        self.assertEqual(station.exposition.it, "")
        self.assertEqual(station.exposition.en, "")

    def test_read_stations_no_filters(self):
        stations = db.read_stations(self.conn)
        # Default exclude_empty=True, so BAS (no precip) and BER (no temp) and LUG (no data) are excluded
        self.assertEqual(len(stations), 1)  # Only ABO has both temp and precip data
        self.assertEqual(stations[0].abbr, "ABO")

    def test_read_stations_localized_string(self):
        stations = db.read_stations(self.conn, exclude_empty=False)
        station = next((s for s in stations if s.abbr == "BER"), None)
        self.assertIsNotNone(station)
        self.assertEqual(station.exposition.de, "Ebene")
        self.assertEqual(station.exposition.it, "Pianura")

    def test_read_stations_exclude_empty_false(self):
        stations = db.read_stations(self.conn, exclude_empty=False)
        self.assertEqual(len(stations), 4)  # All stations except LUG (no data at all)
        self.assertEqual(stations[0].abbr, "ABO")
        self.assertEqual(stations[1].abbr, "BAS")
        self.assertEqual(stations[2].abbr, "BER")
        self.assertEqual(stations[3].abbr, "LUG")

    def test_read_stations_with_canton_filter(self):
        stations = db.read_stations(self.conn, cantons=["GR"])
        self.assertEqual(len(stations), 1)
        self.assertEqual(stations[0].abbr, "ABO")

        stations = db.read_stations(self.conn, cantons=["BL", "BE"])
        self.assertEqual(
            len(stations), 0
        )  # BAS (no precip) and BER (no temp) are excluded by default

    def test_read_stations_canton_and_exclude_empty_false(self):
        stations = db.read_stations(
            self.conn, cantons=["BL", "BE"], exclude_empty=False
        )
        self.assertEqual(len(stations), 2)
        self.assertEqual(stations[0].abbr, "BAS")
        self.assertEqual(stations[1].abbr, "BER")

    def test_read_stations_canton_and_exclude_empty_true(self):
        stations = db.read_stations(
            self.conn, cantons=["GR", "BL", "BE"], exclude_empty=True
        )
        self.assertEqual(len(stations), 1)  # Only ABO has both temp and precip data
        self.assertEqual(stations[0].abbr, "ABO")


class TestDbDaily(TestDb):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._insert_daily_test_data()

    def setUp(self):
        self.conn = self.engine.connect()

    def tearDown(self):
        self.conn.close()

    @classmethod
    def _insert_daily_test_data(cls):
        historical_data = [
            ("ABO", "2023-01-01", 1.0, 0.5, 1.5, 10.0),
            ("ABO", "2023-02-01", 2.0, 1.5, 2.5, 12.0),
            ("ABO", "2023-03-01", 3.0, 2.5, 3.5, 15.0),
            ("ABO", "2023-04-01", 4.0, 3.5, 4.5, 18.0),
            ("ABO", "2023-05-01", 5.0, 4.5, 5.5, 20.0),
            ("ABO", "2023-06-01", 6.0, 5.5, 6.5, 22.0),
            ("ABO", "2023-07-01", 7.0, 6.5, 7.5, 25.0),
            ("ABO", "2023-08-01", 8.0, 7.5, 8.5, 28.0),
            ("ABO", "2023-09-01", 9.0, 8.5, 9.5, 30.0),
            ("ABO", "2023-10-01", 10.0, 9.5, 10.5, 32.0),
            ("ABO", "2023-11-01", 11.0, 10.5, 11.5, 35.0),
            ("ABO", "2023-12-01", 12.0, 11.5, 12.5, 38.0),
            ("ABO", "2024-01-01", 1.5, 1.0, 2.0, 10.5),
            ("GEN", "2023-06-15", 20.0, 19.0, 21.0, 5.0),
            ("GEN", "2023-07-15", 22.0, 21.0, 23.0, 7.0),
        ]
        records = [
            dict(
                station_abbr=station_abbr,
                reference_timestamp=reference_timestamp,
                tre200d0=tre200d0,
                tre200dn=tre200dn,
                tre200dx=tre200dx,
                rre150d0=rre150d0,
            )
            for (
                station_abbr,
                reference_timestamp,
                tre200d0,
                tre200dn,
                tre200dx,
                rre150d0,
            ) in historical_data
        ]
        with cls.engine.begin() as conn:
            conn.execute(
                sa.insert(db.TABLE_DAILY_MEASUREMENTS.sa_table),
                records,
            )

    def test_read_daily_historical_no_filters(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 13)  # 12 months in 2023 + 1 in 2024
        self.assertIn(db.TEMP_DAILY_MEAN, df.columns)
        self.assertIn(db.PRECIP_DAILY_MM, df.columns)
        self.assertEqual(df.index.name, "reference_timestamp")
        self.assertEqual(df.index.dtype, "datetime64[ns]")

    def test_read_daily_historical_with_period_month(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO", period="1")
        self.assertEqual(len(df), 2)  # Jan 2023, Jan 2024
        self.assertTrue(all(df.index.month == 1))

    def test_read_daily_historical_with_period_season(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO", period="spring")
        self.assertEqual(len(df), 3)  # Mar, Apr, May 2023
        self.assertTrue(all(df.index.month.isin([3, 4, 5])))

    def test_read_daily_historical_with_period_all(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO", period="all")
        self.assertEqual(len(df), 13)  # All data for ABO

    def test_read_daily_historical_with_from_year(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO", from_year=2024)
        self.assertEqual(len(df), 1)  # Only 2024-01-01
        self.assertTrue(all(df.index.year >= 2024))

    def test_read_daily_historical_with_to_year(self):
        df = db.read_daily_measurements(self.conn, station_abbr="ABO", to_year=2023)
        self.assertEqual(len(df), 12)  # All of 2023
        self.assertTrue(all(df.index.year <= 2023))

    def test_read_daily_historical_with_from_and_to_year(self):
        df = db.read_daily_measurements(
            self.conn, station_abbr="ABO", from_year=2023, to_year=2023
        )
        self.assertEqual(len(df), 12)  # All of 2023
        self.assertTrue(all(df.index.year == 2023))

    def test_read_daily_historical_no_data_for_station(self):
        df = db.read_daily_measurements(self.conn, station_abbr="LUG")
        self.assertTrue(df.empty)

    def test_read_daily_historical_invalid_columns(self):
        with self.assertRaises(ValueError):
            db.read_daily_measurements(
                self.conn, station_abbr="ABO", columns=["invalid;column"]
            )

    def test_read_daily_historical_specific_columns(self):
        df = db.read_daily_measurements(
            self.conn, station_abbr="ABO", columns=[db.TEMP_DAILY_MEAN]
        )
        self.assertIn(db.TEMP_DAILY_MEAN, df.columns)
        self.assertNotIn(db.PRECIP_DAILY_MM, df.columns)
        self.assertEqual(set(df.columns), {db.TEMP_DAILY_MEAN, "station_abbr"})


class TestDbHourly(TestDb):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._insert_hourly_test_data()

    def setUp(self):
        self.conn = self.engine.connect()

    def tearDown(self):
        self.conn.close()

    @classmethod
    def _insert_hourly_test_data(cls):
        historical_data = [
            ("ABO", "2023-01-01 00:00:00Z", 1.0, 0.5, 1.5, 10.0),
            ("ABO", "2023-01-01 01:00:00Z", 2.0, 1.5, 2.5, 12.0),
            ("ABO", "2023-01-01 02:00:00Z", 3.0, 2.5, 3.5, 15.0),
            ("ABO", "2023-01-01 03:00:00Z", 4.0, 3.5, 4.5, 18.0),
            ("ABO", "2023-01-01 04:00:00Z", 5.0, 4.5, 5.5, 20.0),
            ("ABO", "2023-01-01 05:00:00Z", 6.0, 5.5, 6.5, 22.0),
            ("ABO", "2024-01-02 00:00:00Z", 1.5, 1.0, 2.0, 10.5),
            ("GEN", "2023-08-15 12:00:00Z", 23.0, 22.0, 24.5, 7.0),
            ("GEN", "2023-07-15 12:00:00Z", 22.0, 21.0, 23.0, 7.0),
            ("GEN", "2023-06-15 12:00:00Z", 20.0, 19.0, 21.0, 5.0),
        ]
        records = [
            dict(
                station_abbr=station_abbr,
                reference_timestamp=reference_timestamp,
                tre200h0=tre200h0,
                tre200hn=tre200hn,
                tre200hx=tre200hx,
                rre150h0=rre150h0,
            )
            for (
                station_abbr,
                reference_timestamp,
                tre200h0,
                tre200hn,
                tre200hx,
                rre150h0,
            ) in historical_data
        ]
        with cls.engine.begin() as conn:
            conn.execute(
                sa.insert(db.TABLE_HOURLY_MEASUREMENTS.sa_table),
                records,
            )

    def test_read_hourly_recent_time_range(self):
        from_date = datetime.datetime(2023, 1, 1, 0, 0, 0, 0, datetime.UTC)
        to_date = from_date + datetime.timedelta(hours=4)
        df = db.read_hourly_measurements(self.conn, "ABO", from_date, to_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 4)  # to_date is exclusive
        self.assertIn(db.TEMP_HOURLY_MEAN, df.columns)
        self.assertIn(db.TEMP_HOURLY_MAX, df.columns)
        self.assertIn(db.TEMP_HOURLY_MIN, df.columns)
        self.assertIn(db.PRECIP_HOURLY_MM, df.columns)
        self.assertEqual(df.index.name, "reference_timestamp")
        self.assertEqual(df.index.dtype, "datetime64[ns, UTC]")
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_read_hourly_recent_multiple_days(self):
        from_date = datetime.datetime(2023, 1, 1, 0, 0, 0, 0, datetime.UTC)
        to_date = from_date + datetime.timedelta(days=365)
        df = db.read_hourly_measurements(self.conn, "GEN", from_date, to_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # all 3 rows should be returned
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_read_hourly_recent_invalid_column(self):
        from_date = datetime.datetime(2023, 1, 1, 0, 0, 0, 0, datetime.UTC)
        to_date = from_date + datetime.timedelta(hours=4)
        with self.assertRaises(ValueError):
            db.read_hourly_measurements(
                self.conn,
                station_abbr="ABO",
                columns=["invalid;column"],
                from_date=from_date,
                to_date=to_date,
            )

    def test_read_hourly_recent_no_data_in_range(self):
        from_date = datetime.datetime(1980, 1, 1, 0, 0, 0, 0, datetime.UTC)
        to_date = from_date + datetime.timedelta(hours=4)
        df = db.read_hourly_measurements(self.conn, "ABO", from_date, to_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


class TestCreateDb(unittest.TestCase):

    def test_create_daily(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()
        db.insert_csv_data(
            _testdata_dir(),
            engine,
            db.TABLE_DAILY_MEASUREMENTS,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_vis_d_recent.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
        )
        with engine.connect() as conn:
            columns = [
                db.TEMP_DAILY_MAX,
                db.PRECIP_DAILY_MM,
                db.ATM_PRESSURE_DAILY_MEAN,
            ]
            df = db.read_daily_measurements(
                conn,
                "VIS",
                columns=columns,
            )
        self.assertEqual(len(df), 202)
        self.assertTrue(
            (df[columns].sum() > 0).all(),
            "All measurement columns should have some nonzero values.",
        )

    def test_create_daily_append(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()
        db.insert_csv_data(
            _testdata_dir(),
            engine,
            db.TABLE_DAILY_MEASUREMENTS,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_vis_d_recent.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
            insert_mode="append",
        )

    def test_create_daily_append_twice_failure(self):
        # Inserting the same data twice should fail in "append" mode.
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()

        def _insert(update_id, mode):
            db.insert_csv_data(
                _testdata_dir(),
                engine,
                db.TABLE_DAILY_MEASUREMENTS,
                db.UpdateStatus(
                    id=update_id,
                    href="file:///ogd-smn_vis_d_recent.csv",
                    resource_updated_time=now,
                    table_updated_time=now,
                ),
                insert_mode=mode,
            )

        _insert(None, "append")
        # Need to get UpdateStatus ID for second round.
        updates = db.read_update_status(engine)
        self.assertEqual(len(updates), 1)
        update_id = updates[0].id
        with self.assertRaises(IntegrityError):
            _insert(update_id, "append")

        # should just do nothing
        _insert(update_id, "insert_missing")

    def test_create_daily_merge_failure(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()
        # insert_mode="merge" only works for Postgres and should fail for sqlite.
        with self.assertRaises(ValueError):
            db.insert_csv_data(
                _testdata_dir(),
                engine,
                db.TABLE_DAILY_MEASUREMENTS,
                db.UpdateStatus(
                    id=None,
                    href="file:///ogd-smn_vis_d_recent.csv",
                    resource_updated_time=now,
                    table_updated_time=now,
                ),
                insert_mode="merge",
            )

    def test_create_hourly(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()
        db.insert_csv_data(
            _testdata_dir(),
            engine,
            db.TABLE_HOURLY_MEASUREMENTS,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_vis_h_recent.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
        )
        columns = [db.TEMP_HOURLY_MAX, db.PRECIP_HOURLY_MM, db.GUST_PEAK_HOURLY_MAX]
        with engine.connect() as conn:
            df = db.read_hourly_measurements(
                conn,
                "VIS",
                from_date=datetime.datetime(2020, 1, 1),
                to_date=datetime.datetime(2026, 1, 1),
                columns=columns,
            )
        self.assertEqual(len(df), 500)
        self.assertTrue(
            (df[columns].sum() > 0).all(),
            "All measurement columns should have some nonzero values.",
        )

    def test_create_monthly(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now()
        db.insert_csv_data(
            _testdata_dir(),
            engine,
            db.TABLE_MONTHLY_MEASUREMENTS,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_vis_m.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
        )
        columns = [
            db.TEMP_MONTHLY_MEAN,
            db.PRECIP_MONTHLY_MM,
            db.REL_HUMIDITY_MONTHLY_MEAN,
        ]
        with engine.connect() as conn:
            df = db.read_monthly_measurements(
                conn,
                "VIS",
                from_date=datetime.datetime(2020, 1, 1),
                to_date=datetime.datetime(2021, 1, 1),
                columns=columns,
            )
        self.assertEqual(len(df), 12)
        self.assertTrue(
            (df[columns].sum() > 0).all(),
            "All measurement columns should have some nonzero values.",
        )

    def test_create_metadata_stations(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now(datetime.timezone.utc)
        db.insert_csv_metadata(
            _testdata_dir(),
            engine,
            db.sa_table_meta_stations,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_meta_stations.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
        )
        # No query methods for metadata, so not much to check in the metadata table itself.
        updates = db.read_update_status(engine)
        self.assertEqual(len(updates), 1)
        u = updates[0]
        self.assertEqual(u.href, "file:///ogd-smn_meta_stations.csv")
        self.assertEqual(len(u.id), 36)  # Should have a uuid4
        self.assertEqual(u.resource_updated_time, now)
        self.assertEqual(u.table_updated_time, now)

    def test_recreate_nearby_stations(self):
        engine = sa.create_engine("sqlite:///:memory:")
        db.metadata.create_all(engine)
        now = datetime.datetime.now(datetime.timezone.utc)
        db.insert_csv_metadata(
            _testdata_dir(),
            engine,
            db.sa_table_meta_stations,
            db.UpdateStatus(
                id=None,
                href="file:///ogd-smn_meta_stations.csv",
                resource_updated_time=now,
                table_updated_time=now,
            ),
        )
        # Need this table to create nearby stations.
        db.recreate_station_data_summary(engine)
        # Now create nearby stations
        db.recreate_nearby_stations(engine, max_neighbors=4, exclude_empty=False)

        # Validate
        with engine.begin() as conn:
            ns = db.read_nearby_stations(conn, "BER")

        self.assertEqual(len(ns), 4)
        nsmap = {n.abbr: n for n in ns}
        self.assertIn("GRE", nsmap)
        gre = nsmap["GRE"]
        self.assertAlmostEqual(gre.distance_km, 21.267, delta=0.01)
        self.assertAlmostEqual(gre.height_diff, -125, delta=0.01)


class TestDbRefPeriod1991_2020(unittest.TestCase):
    def setUp(self):
        self.engine = sa.create_engine("sqlite:///:memory:")

        # Create all tables for tests
        db.metadata.create_all(self.engine)

    def test_recreate_empty(self):
        db.recreate_reference_period_stats_all(self.engine)

    def test_create_select(self):
        # Insert daily data.
        def p(stn, dt, t, p):
            return {
                "station_abbr": stn,
                "reference_timestamp": dt,
                db.TEMP_DAILY_MIN: t,
                db.PRECIP_DAILY_MM: p,
            }

        conn = self.engine.connect()
        conn.execute(
            sa.insert(db.TABLE_DAILY_MEASUREMENTS.sa_table),
            [
                p("BER", "1991-01-01", -3, 0.5),
                p("BER", "1991-01-02", -4, 1.5),
                p("BER", "2001-06-03", 22, None),
                p("XXX", "2001-06-03", 30, 12),
            ],
        )
        conn.commit()
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        df = db.read_var_summary_stats_all(
            conn,
            db.AGG_NAME_REF_1991_2020,
            station_abbr="BER",
            variables=[db.TEMP_DAILY_MIN, db.PRECIP_DAILY_MM],
        )

        self.assertEqual(len(df), 2)
        self.assertEqual(
            df.columns.to_list(),
            [
                "source_granularity",
                "min_value",
                "min_value_date",
                "mean_value",
                "max_value",
                "max_value_date",
                "p10_value",
                "p25_value",
                "median_value",
                "p75_value",
                "p90_value",
                "value_sum",
                "value_count",
            ],
        )
        t = df.loc["BER", db.TEMP_DAILY_MIN]
        self.assertAlmostEqual(t["mean_value"], 5.0)
        self.assertEqual(t["min_value_date"], "1991-01-02")
        self.assertEqual(t["max_value_date"], "2001-06-03")
        p = df.loc["BER", db.PRECIP_DAILY_MM]
        self.assertEqual(p["min_value"], 0.5)
        self.assertEqual(p["max_value_date"], "1991-01-02")
        self.assertEqual(p["value_sum"], 2.0)
        self.assertEqual(p["value_count"], 2)  # One row has None for rre150d0
        self.assertEqual(p["source_granularity"], "daily")

    def _insert_var(self, var_col: str, params: list[tuple]) -> str:
        with self.engine.begin() as conn:
            records = [
                {
                    "station_abbr": stn,
                    "reference_timestamp": dt,
                    var_col: val,
                }
                for (stn, dt, val) in params
            ]
            conn.execute(sa.insert(db.TABLE_DAILY_MEASUREMENTS.sa_table), records)

    def _insert_vars(self, var_cols: list[str], params: list[tuple]) -> str:
        with self.engine.begin() as conn:
            records = [
                {
                    "station_abbr": stn,
                    "reference_timestamp": dt,
                    **dict(zip(var_cols, vals)),
                }
                for (stn, dt, *vals) in params
            ]
            conn.execute(sa.insert(db.TABLE_DAILY_MEASUREMENTS.sa_table), records)

    def test_derived_summer_days(self):
        # Insert daily data.
        self._insert_var(
            db.TEMP_DAILY_MAX,
            [
                ("BER", "1991-07-01", 24.9),
                ("BER", "1991-07-02", 25.1),
                ("BER", "2001-06-03", 25),
                ("BER", "2001-06-04", 28),
                ("BER", "2001-06-05", 27),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        self.assertGreaterEqual(len(df), 1)

        s = df.loc[("BER", db.DX_SUMMER_DAYS_ANNUAL_COUNT)]
        self.assertAlmostEqual(
            s["mean_value"], 2.0, msg="2 years with data, 4 summer days."
        )
        self.assertEqual(s["min_value"], 1.0)
        self.assertEqual(s["max_value"], 3.0)
        # Dates for annual granularity are stored as 01 Jan.
        self.assertEqual(s["min_value_date"], "1991-01-01")
        self.assertEqual(s["max_value_date"], "2001-01-01")
        self.assertEqual(s["source_granularity"], "annual")
        self.assertEqual(s["value_count"], 2)

    def test_derived_frost_days(self):
        # Insert daily data.
        self._insert_var(
            db.TEMP_DAILY_MIN,
            [
                ("BER", "1991-01-01", 0.5),
                ("BER", "1992-02-02", -1),
                ("BER", "1992-02-03", -0.5),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["BER", db.DX_FROST_DAYS_ANNUAL_COUNT]
        self.assertEqual(fd["min_value"], 0.0)
        self.assertEqual(fd["max_value"], 2.0)

    def test_derived_tropical_nights(self):
        # Insert daily data.
        self._insert_var(
            db.TEMP_DAILY_MIN,
            [
                ("LUG", "1991-07-01", 21),
                ("LUG", "1992-07-01", 19),
                ("LUG", "1992-07-02", 18),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["LUG", db.DX_TROPICAL_NIGHTS_ANNUAL_COUNT]
        self.assertEqual(fd["min_value"], 0.0)
        self.assertEqual(fd["max_value"], 1.0)
        self.assertEqual(fd["max_value_date"], "1991-01-01")

    def test_extra_column_date_range(self):
        # Test that the "virtual" column "date_range" is included
        # and has the total min and max date as extremal points.
        self._insert_var(
            db.TEMP_DAILY_MIN,
            [
                ("LUG", "1991-01-01", 21),
                ("LUG", "1992-07-01", 19),
                ("LUG", "2020-12-31", 18),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["LUG", db.DX_SOURCE_DATE_RANGE]
        self.assertAlmostEqual(fd["min_value"], 7670.0)  # Days since epoch
        self.assertEqual(fd["min_value_date"], "1991-01-01")
        self.assertAlmostEqual(fd["max_value"], 18627.0)
        self.assertEqual(fd["max_value_date"], "2020-12-31")

    def test_derived_sunny_days(self):
        # Insert daily data.
        self._insert_var(
            db.SUNSHINE_DAILY_MINUTES,
            [
                ("BER", "1991-06-01", 8 * 60),
                ("BER", "1991-06-02", 5.8 * 60),
                ("BER", "1991-06-03", 6.0 * 60),
                ("BER", "1991-06-04", 8 * 60),
                ("BER", "1992-06-01", 8 * 60),
                ("BER", "1993-06-01", 8 * 60),
                ("BER", "1994-06-01", 8 * 60),
                ("BER", "1995-06-01", 8 * 60),
                # No sunny days in 1996 and 1997
                ("BER", "1996-06-01", 30),
                ("BER", "1997-06-01", 30),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["BER", db.DX_SUNNY_DAYS_ANNUAL_COUNT]
        self.assertEqual(fd["min_value"], 0.0)
        self.assertEqual(fd["max_value"], 3.0)
        # This is annual granularity, so expect 1 value per year
        self.assertEqual(fd["value_count"], 7)
        self.assertEqual(fd["source_granularity"], "annual")
        self.assertAlmostEqual(fd["mean_value"], 1.0)
        self.assertEqual(fd["max_value_date"], "1991-01-01")

    def test_derived_sunny_days_no_data(self):
        # This tests that derived data "preserves NaNs":
        # If there was not a single day matching the criteria,
        # we want 0. But if there was simply no data for the variable
        # from which the derived metric is ... derived, then we
        # should not have 0, but NA.
        self._insert_var(
            db.SUNSHINE_DAILY_MINUTES,
            [
                # BER has zeros sunny days
                ("BER", "1991-06-01", 0),
                ("BER", "1991-06-02", 0),
                # GES does not have data at all
                ("GES", "1991-06-01", None),
                ("GES", "1991-06-02", None),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(conn, db.AGG_NAME_REF_1991_2020)

        # BER
        self.assertTrue("BER" in df.index)
        ber = df.loc["BER", db.DX_SUNNY_DAYS_ANNUAL_COUNT]
        self.assertEqual(ber["min_value"], 0.0)
        self.assertEqual(ber["max_value"], 0.0)
        self.assertEqual(ber["value_count"], 1)  # Data for 1 year
        self.assertEqual(ber["source_granularity"], "annual")
        # GES shouldn't even be there
        self.assertTrue("GET" not in df.index)

    def test_derived_precipitation_total(self):
        # Insert daily data.
        self._insert_var(
            db.PRECIP_DAILY_MM,
            [
                ("BER", "1991-06-01", 15),
                ("BER", "1991-06-02", 0),
                ("BER", "1991-06-03", 10),
                ("BER", "1991-06-04", 100),
                ("BER", "1992-06-01", 0),
                ("BER", "1993-06-01", 0),
                ("BER", "1994-06-01", 40),
                ("BER", "1995-06-01", 30),
                ("BER", "1996-06-01", 30),
                ("BER", "1997-06-01", 30),
            ],
        )
        # Recreate derived tables.
        db.recreate_reference_period_stats_month(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_month(conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        pt = df.loc["BER", db.DX_PRECIP_TOTAL, db.ts_month(6)]
        self.assertEqual(pt["min_value"], 0.0)
        # Sum of precipitation in given month, so should be 125 from 1991:
        self.assertEqual(pt["max_value"], 125.0)
        self.assertEqual(pt["max_value_date"], "1991-01-01")

    def test_create_select_agg_not_exist(self):
        self._insert_var(db.TEMP_DAILY_MIN, [("BER", "1991-01-01", -3)])
        db.recreate_reference_period_stats_all(self.engine)
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(
                conn, "AGG_DOES_NOT_EXIST", station_abbr="BER"
            )
        self.assertTrue(df.empty)

    def test_create_select_nan(self):
        # Insert daily data. PRECIP_DAILY_MM is always empty.
        self._insert_vars(
            [db.TEMP_DAILY_MIN, db.PRECIP_DAILY_MM],
            [
                ("BER", "1991-01-01", 10, None),
                ("BER", "1991-01-02", 10, None),
                ("BER", "2001-06-03", 17, None),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_all(self.engine)
        # Read data
        with self.engine.begin() as conn:
            df = db.read_var_summary_stats_all(
                conn, db.AGG_NAME_REF_1991_2020, station_abbr="BER"
            )

        # Should return data for TEMP_DAILY_MIN and derived metrics.
        self.assertGreaterEqual(len(df), 1)

        vars = df.loc["BER"].index.get_level_values(0).unique().to_list()
        self.assertIn(db.TEMP_DAILY_MIN, vars)
        self.assertNotIn(db.PRECIP_DAILY_MM, vars)

    def test_create_select_single_month(self):
        self._insert_vars(
            [db.TEMP_DAILY_MIN, db.PRECIP_DAILY_MM],
            [
                ("BER", "1991-01-01", 10, None),
                ("BER", "1991-01-02", 11, None),
                ("BER", "1999-01-31", 12, None),
                ("BER", "2001-06-03", 17, None),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_month(self.engine)

        with self.engine.begin() as conn:
            df = db.read_summary_stats(
                conn,
                table=db.var_summary_stats_month.sa_table,
                agg_name=db.AGG_NAME_REF_1991_2020,
                station_abbr="BER",
                variables=[db.TEMP_DAILY_MIN],
                time_slices=["01"],
            )
        self.assertEqual(len(df), 1)
        var = df.loc[("BER", db.TEMP_DAILY_MIN, db.ts_month(1))]
        self.assertEqual(var["min_value"], 10)
        self.assertEqual(var["max_value"], 12)
        self.assertEqual(var["value_count"], 3)

    def test_create_select_all_months(self):
        self._insert_vars(
            [db.TEMP_DAILY_MIN, db.PRECIP_DAILY_MM],
            [
                ("BER", "1980-01-01", 10, None),
                ("ABO", "1991-12-31", 10, 100),
                ("BER", "1991-01-01", 10, None),
                ("BER", "1991-01-02", 11, None),
                ("BER", "1999-01-31", 12, None),
                ("BER", "2001-06-03", 17, None),
                ("BER", "2025-01-01", None, 100),
            ],
        )
        # Recreate derived table.
        db.recreate_reference_period_stats_month(self.engine)

        with self.engine.begin() as conn:
            df = db.read_summary_stats(
                conn,
                table=db.var_summary_stats_month.sa_table,
                agg_name=db.AGG_NAME_REF_1991_2020,
                variables=[db.TEMP_DAILY_MIN, db.PRECIP_DAILY_MM],
            )
        self.assertEqual(len(df), 4)
        var = df.loc[("ABO", db.PRECIP_DAILY_MM, db.ts_month(12))]
        self.assertEqual(var["min_value"], 100)
        self.assertEqual(var["max_value"], 100)
        self.assertEqual(var["value_count"], 1)
        # The three other results are (choosing min_value arbitrarily):
        self.assertIn("min_value", df.loc[("BER", db.TEMP_DAILY_MIN, db.ts_month(1))])
        self.assertIn("min_value", df.loc[("BER", db.TEMP_DAILY_MIN, db.ts_month(6))])
        self.assertIn("min_value", df.loc[("ABO", db.TEMP_DAILY_MIN, db.ts_month(12))])


class TestHelpers(unittest.TestCase):

    def test_column_to_dtype(self):
        def _col(name):
            return db.TABLE_DAILY_MEASUREMENTS.sa_table.columns[name]

        self.assertEqual(db._column_to_dtype(_col(db.PRECIP_DAILY_MM)), float)
        self.assertEqual(db._column_to_dtype(_col("station_abbr")), str)
        int_col = db.sa_table_meta_parameters.columns["parameter_decimals"]
        self.assertEqual(db._column_to_dtype(int_col), int)
        # Unsupported type should raise ValueError:
        with self.assertRaises(ValueError):
            db._column_to_dtype(sa.Column(name="test", type_=sa.DATE))


class TestVarSummaryStats(PandasTestCase):

    def test_simple(self):
        df = pd.DataFrame(
            [
                ("2025-01-01", "BER", "01", 1.0, 2.0),
                ("2025-01-01", "ABO", "01", 5.0, 6.0),
                ("2025-01-02", "BER", "01", 3.0, 4.0),
                ("2025-01-03", "BER", "01", 5.0, 6.0),
            ],
            columns=["date", "station_abbr", "time_slice", "x1", "x2"],
        )
        df["date"] = pd.to_datetime(df["date"])

        result = db._var_summary_stats(
            df,
            agg_name="test_agg",
            date_col="date",
            var_cols=["x1", "x2"],
            granularity="daily",
        )

        self.assertEqual(len(result), 4)
        res_df = pd.DataFrame(result)

        # Should have one row per "primary key"
        pk_cols = ["station_abbr", "variable", "time_slice"]
        counts = res_df[pk_cols].value_counts(pk_cols).reset_index()
        self.assertFrameEqual(
            counts,
            pd.DataFrame(
                [
                    ("ABO", "x1", "01", 1),
                    ("ABO", "x2", "01", 1),
                    ("BER", "x1", "01", 1),
                    ("BER", "x2", "01", 1),
                ],
                columns=pk_cols + ["count"],
            ),
        )

        self.assertTrue((res_df["agg_name"] == "test_agg").all())

    def test_all_nan(self):
        df = pd.DataFrame(
            [
                ("2025-01-01", "BER", "01", np.nan, 2.0),
                ("2025-01-01", "ABO", "01", 5.0, 6.0),
                ("2025-01-02", "BER", "01", np.nan, 4.0),
            ],
            columns=["date", "station_abbr", "time_slice", "x1", "x2"],
        )
        df["date"] = pd.to_datetime(df["date"])

        result = db._var_summary_stats(
            df,
            agg_name="test_agg",
            date_col="date",
            var_cols=["x1", "x2"],
            granularity="daily",
        )

        res_df = pd.DataFrame(result)
        self.assertEqual(len(res_df), 3)

        # Should have one row per "primary key", but
        # (BER, x1) should be missing, since it has all nan values
        pk_cols = ["station_abbr", "variable", "time_slice"]
        counts = res_df[pk_cols].value_counts(pk_cols).reset_index()
        self.assertFrameEqual(
            counts,
            pd.DataFrame(
                [
                    ("ABO", "x1", "01", 1),
                    ("ABO", "x2", "01", 1),
                    ("BER", "x2", "01", 1),
                ],
                columns=pk_cols + ["count"],
            ),
        )

        self.assertTrue((res_df["agg_name"] == "test_agg").all())

    def test_partial_nan(self):
        df = pd.DataFrame(
            [
                ("2025-01-01", "BER", "01", np.nan, 2.0),
                ("2025-01-02", "BER", "01", 1.0, 6.0),
                ("2025-01-03", "BER", "01", np.nan, 4.0),
            ],
            columns=["date", "station_abbr", "time_slice", "x1", "x2"],
        )
        df["date"] = pd.to_datetime(df["date"])

        result = db._var_summary_stats(
            df,
            agg_name="test_agg",
            date_col="date",
            var_cols=["x1", "x2"],
            granularity="daily",
        )

        self.assertEqual(len(result), 2)
        res_df = pd.DataFrame(result)

        # Should have one row per "primary key", but
        # (BER, x1) should be missing, since it has all nan values
        cols = ["station_abbr", "variable", "time_slice", "value_count", "mean_value"]
        expected = res_df[cols]
        self.assertFrameEqual(
            expected,
            pd.DataFrame(
                [
                    ("BER", "x1", "01", 1, 1.0),
                    ("BER", "x2", "01", 3, 4.0),
                ],
                columns=cols,
            ),
        )
