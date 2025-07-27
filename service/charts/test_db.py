import os
import unittest
import sqlite3
import datetime
import pandas as pd

from . import db
from . import models
from .errors import StationNotFoundError


def _testdata_dir():
    return os.path.join(os.path.dirname(__file__), "testdata")


class TestDb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(":memory:")
        cls.conn.row_factory = sqlite3.Row
        cls._create_all_tables()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    @classmethod
    def _create_all_tables(cls):
        conn = cls.conn
        db.prepare_sql_table_from_spec(
            _testdata_dir(), conn, db.TABLE_DAILY_MEASUREMENTS, []
        )
        db.prepare_sql_table_from_spec(
            _testdata_dir(), conn, db.TABLE_HOURLY_MEASUREMENTS, []
        )
        conn.execute(
            """
            CREATE TABLE ogd_smn_meta_stations (
                station_abbr TEXT,
                station_name TEXT,
                station_canton TEXT,
                station_wigos_id TEXT,
                station_type_de TEXT,
                station_type_fr TEXT,
                station_type_it TEXT,
                station_type_en TEXT,
                station_dataowner TEXT,
                station_data_since TEXT,
                station_height_masl REAL,
                station_height_barometer_masl REAL,
                station_coordinates_lv95_east REAL,
                station_coordinates_lv95_north REAL,
                station_coordinates_wgs84_lat REAL,
                station_coordinates_wgs84_lon REAL,
                station_exposition_de TEXT,
                station_exposition_fr TEXT,
                station_exposition_it TEXT,
                station_exposition_en TEXT,
                station_url_de TEXT,
                station_url_fr TEXT,
                station_url_it TEXT,
                station_url_en TEXT,
                PRIMARY KEY (station_abbr)
            );
            """
        )
        db.recreate_station_data_summary(conn)


class TestDbStations(TestDb):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._insert_station_test_data()

    @classmethod
    def _insert_station_test_data(cls):
        cursor = cls.conn.cursor()
        stations_data = [
            (
                "ABO",
                "Arosa",
                "GR",
                "2000-01-01",
                "2020-12-31",
                "2000-01-01",
                "2020-12-31",
                100,  # tre200d0_count
                100,  # rre150d0_count
                "Ebene",
                "Plaine",
                "Pianura",
                "plain",
            ),
            (
                "BAS",
                "Basel / Binningen",
                "BL",
                "1990-01-01",
                "2010-12-31",
                None,
                None,
                50,  # tre200d0_count
                0,  # rre150d0_count
                "Ebene",
                None,
                None,
                None,
            ),  # Missing precip dates
            (
                "BER",
                "Bern / Zollikofen",
                "BE",
                None,
                None,
                "1980-01-01",
                "2022-12-31",
                0,  # tre200d0_count
                50,  # rre150d0_count
                "Ebene",
                "Plaine",
                "Pianura",
                "plain",
            ),  # Missing temp dates
            (
                "LUG",
                "Lugano",
                "TI",
                None,
                None,
                None,
                None,
                0,
                0,
                None,
                None,
                None,
                None,
            ),  # All dates missing
        ]
        cursor.executemany(
            f"""
            INSERT INTO {db.TABLE_NAME_X_STATION_DATA_SUMMARY} (
                station_abbr, station_name, station_canton,
                tre200d0_min_date, tre200d0_max_date,
                rre150d0_min_date, rre150d0_max_date,
                tre200d0_count, rre150d0_count,
                station_exposition_de, station_exposition_fr,
                station_exposition_it, station_exposition_en
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            stations_data,
        )
        cls.conn.commit()

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

    @classmethod
    def _insert_daily_test_data(cls):
        cursor = cls.conn.cursor()
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
        cursor.executemany(
            f"""
            INSERT INTO {db.TABLE_DAILY_MEASUREMENTS.name}
            (station_abbr, reference_timestamp, tre200d0, tre200dn, tre200dx, rre150d0)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            historical_data,
        )
        cls.conn.commit()

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

    @classmethod
    def _insert_hourly_test_data(cls):
        cursor = cls.conn.cursor()
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
        cursor.executemany(
            f"""
            INSERT INTO {db.TABLE_HOURLY_MEASUREMENTS.name} 
            (station_abbr, reference_timestamp, tre200h0, tre200hn, tre200hx, rre150h0)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            historical_data,
        )
        cls.conn.commit()

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
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_create_daily(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(),
            self.conn,
            db.TABLE_DAILY_MEASUREMENTS,
            ["ogd-smn_vis_d_recent.csv"],
        )
        columns = [db.TEMP_DAILY_MAX, db.PRECIP_DAILY_MM, db.ATM_PRESSURE_DAILY_MEAN]
        df = db.read_daily_measurements(
            self.conn,
            "VIS",
            columns=columns,
        )
        self.assertEqual(len(df), 202)
        self.assertTrue(
            (df[columns].sum() > 0).all(),
            "All measurement columns should have some nonzero values.",
        )

    def test_create_hourly(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(),
            self.conn,
            db.TABLE_HOURLY_MEASUREMENTS,
            ["ogd-smn_vis_h_recent.csv"],
        )
        columns = [db.TEMP_HOURLY_MAX, db.PRECIP_HOURLY_MM, db.GUST_PEAK_HOURLY_MAX]
        df = db.read_hourly_measurements(
            self.conn,
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


class TestDbRefPeriod1991_2020(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_create_empty(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(), self.conn, db.TABLE_DAILY_MEASUREMENTS, []
        )
        db.recreate_station_var_summary_stats(self.conn)

    def test_create_select(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(), self.conn, db.TABLE_DAILY_MEASUREMENTS, []
        )
        # Insert daily data.
        self.conn.executemany(
            f"""
            INSERT INTO {db.TABLE_DAILY_MEASUREMENTS.name} (
                station_abbr,
                reference_timestamp,
                tre200dn,
                rre150d0
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                ("BER", "1991-01-01", -3, 0),
                ("BER", "1991-01-02", -4, 1.5),
                ("BER", "2001-06-03", 22, None),
                ("XXX", "2001-06-03", 30, 12),
            ],
        )
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(
            self.conn, db.AGG_NAME_REF_1991_2020, station_abbr="BER"
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(
            df.columns.levels[1].tolist(),
            [
                "source_granularity",
                "min_value",
                "min_value_date",
                "mean_value",
                "max_value",
                "max_value_date",
                "value_count",
            ],
        )
        t = df.loc["BER", "tre200dn"]
        self.assertAlmostEqual(t["mean_value"], 5.0)
        self.assertEqual(t["min_value_date"], "1991-01-02")
        self.assertEqual(t["max_value_date"], "2001-06-03")
        p = df.loc["BER", "rre150d0"]
        self.assertEqual(p["min_value"], 0)
        self.assertEqual(p["max_value_date"], "1991-01-02")
        self.assertEqual(p["value_count"], 2)  # One row has None for rre150d0
        self.assertEqual(p["source_granularity"], "daily")

    def _insert_var(self, var_col: str, params: list[tuple]) -> str:
        db.prepare_sql_table_from_spec(
            _testdata_dir(), self.conn, db.TABLE_DAILY_MEASUREMENTS
        )
        sql = f"""
            INSERT INTO {db.TABLE_DAILY_MEASUREMENTS.name} (
                station_abbr,
                reference_timestamp,
                {var_col}
            )
            VALUES (?, ?, ?)
            """
        self.conn.executemany(sql, params)

    def test_derived_summer_days(self):
        # Insert daily data.
        self._insert_var(
            "tre200dx",
            [
                ("BER", "1991-07-01", 24.9),
                ("BER", "1991-07-02", 25.1),
                ("BER", "2001-06-03", 25),
                ("BER", "2001-06-04", 28),
                ("BER", "2001-06-05", 27),
            ],
        )
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(self.conn, db.AGG_NAME_REF_1991_2020)

        self.assertEqual(len(df), 1)

        s = df.loc["BER", db.DX_SUMMER_DAYS]
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
            "tre200dn",
            [
                ("BER", "1991-01-01", 0.5),
                ("BER", "1992-02-02", -1),
                ("BER", "1992-02-03", -0.5),
            ],
        )
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(self.conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["BER", db.DX_FROST_DAYS]
        self.assertEqual(fd["min_value"], 0.0)
        self.assertEqual(fd["max_value"], 2.0)

    def test_derived_sunny_days(self):
        # Insert daily data.
        self._insert_var(
            "sre000d0",
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
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(self.conn, db.AGG_NAME_REF_1991_2020)

        # Check summary stats
        fd = df.loc["BER", db.DX_SUNNY_DAYS]
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
            "sre000d0",
            [
                # BER has zeros sunny days
                ("BER", "1991-06-01", 0),
                ("BER", "1991-06-02", 0),
                # GES does not have data at all
                ("GES", "1991-06-01", None),
                ("GES", "1991-06-02", None),
            ],
        )
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(self.conn, db.AGG_NAME_REF_1991_2020)

        # BER
        self.assertTrue("BER" in df.index)
        ber = df.loc["BER", db.DX_SUNNY_DAYS]
        self.assertEqual(ber["min_value"], 0.0)
        self.assertEqual(ber["max_value"], 0.0)
        self.assertEqual(ber["value_count"], 1)  # Data for 1 year
        self.assertEqual(ber["source_granularity"], "annual")
        # GES shouldn't even be there
        self.assertTrue("GET" not in df.index)

    def test_create_select_agg_not_exist(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(), self.conn, db.TABLE_DAILY_MEASUREMENTS, []
        )
        self.conn.executemany(
            f"""
            INSERT INTO {db.TABLE_DAILY_MEASUREMENTS.name} (
                station_abbr, reference_timestamp, tre200dn, rre150d0
            )
            VALUES (?, ?, ?, ?)
            """,
            [("BER", "1991-01-01", -3, 20)],
        )
        db.recreate_station_var_summary_stats(self.conn)
        df = db.read_station_var_summary_stats(
            self.conn, "AGG_DOES_NOT_EXIST", station_abbr="BER"
        )
        self.assertTrue(df.empty)

    def test_create_select_nan(self):
        db.prepare_sql_table_from_spec(
            _testdata_dir(), self.conn, db.TABLE_DAILY_MEASUREMENTS, []
        )
        # Insert daily data. rre150d0 is always empty.
        self.conn.executemany(
            f"""
            INSERT INTO {db.TABLE_DAILY_MEASUREMENTS.name} (
                station_abbr,
                reference_timestamp,
                tre200dn,
                rre150d0
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                ("BER", "1991-01-01", -3, None),
                ("BER", "1991-01-02", -4, None),
                ("BER", "2001-06-03", 22, None),
            ],
        )
        self.conn.commit()
        # Recreate derived table.
        db.recreate_station_var_summary_stats(self.conn)
        # Read data
        df = db.read_station_var_summary_stats(
            self.conn, db.AGG_NAME_REF_1991_2020, station_abbr="BER"
        )

        self.assertEqual(len(df), 1)

        # Should not have data for rre150d0:
        with self.assertRaises(KeyError):
            df.loc["BER", "rre150d0"]
