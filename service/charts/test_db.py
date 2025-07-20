import unittest
import sqlite3
import datetime

from . import db
from . import models


class TestDb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(":memory:")
        cls.conn.row_factory = sqlite3.Row
        cls._create_tables()
        cls._insert_test_data()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    @classmethod
    def _create_tables(cls):
        cursor = cls.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE ogd_smn_station_data_summary (
                station_abbr TEXT PRIMARY KEY,
                station_name TEXT,
                station_canton TEXT,
                tre200d0_min_date TEXT,
                tre200d0_max_date TEXT,
                rre150d0_min_date TEXT,
                rre150d0_max_date TEXT
            )
        """
        )
        cls.conn.commit()

    @classmethod
    def _insert_test_data(cls):
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
            ),
            (
                "BAS",
                "Basel / Binningen",
                "BL",
                "1990-01-01",
                "2010-12-31",
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
            ),  # Missing temp dates
            ("LUG", "Lugano", "TI", None, None, None, None),  # All dates missing
        ]
        cursor.executemany(
            """
            INSERT INTO ogd_smn_station_data_summary (
                station_abbr, station_name, station_canton,
                tre200d0_min_date, tre200d0_max_date,
                rre150d0_min_date, rre150d0_max_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
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
        self.assertEqual(station.first_available_date, datetime.date(2000, 1, 1))
        self.assertEqual(station.last_available_date, datetime.date(2020, 12, 31))

    def test_read_station_not_found(self):
        with self.assertRaises(ValueError) as cm:
            db.read_station(self.conn, "XYZ")
        self.assertEqual(str(cm.exception), "No station found with abbr='XYZ'")

    def test_read_station_partial_dates(self):
        station = db.read_station(self.conn, "BAS")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "BAS")
        self.assertEqual(station.first_available_date, datetime.date(1990, 1, 1))
        self.assertEqual(station.last_available_date, datetime.date(2010, 12, 31))

        station = db.read_station(self.conn, "BER")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "BER")
        self.assertEqual(station.first_available_date, datetime.date(1980, 1, 1))
        self.assertEqual(station.last_available_date, datetime.date(2022, 12, 31))

    def test_read_station_no_dates(self):
        station = db.read_station(self.conn, "LUG")
        self.assertIsInstance(station, models.Station)
        self.assertEqual(station.abbr, "LUG")
        self.assertIsNone(station.first_available_date)
        self.assertIsNone(station.last_available_date)
