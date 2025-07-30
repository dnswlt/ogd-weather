import pandas as pd
import unittest
from . import db
from . import charts
from datetime import date


class TestStationPeriodStats(unittest.TestCase):
    def test_variable_stats_dict(self):
        # Wide DataFrame with two variables
        df = pd.DataFrame(
            [
                {
                    "variable": db.TEMP_DAILY_MIN,
                    "min_value": -15.0,
                    "min_value_date": "1995-01-10",
                    "mean_value": 5.0,
                    "max_value": 25.0,
                    "max_value_date": "1995-07-01",
                    "source_granularity": "daily",
                    "value_sum": 5000.0,
                    "value_count": 1000,
                },
                {
                    "variable": db.TEMP_DAILY_MAX,
                    "min_value": -5.0,
                    "min_value_date": "1995-02-02",
                    "mean_value": 15.0,
                    "max_value": 35.0,
                    "max_value_date": "1995-08-15",
                    "source_granularity": "daily",
                    "value_sum": 15000.0,
                    "value_count": 1000,
                },
            ]
        ).set_index("variable")

        # Convert to MultiIndex Series
        s = df.stack()

        # Call function
        result = charts.station_period_stats(s)

        # Check we got both variables in the dict
        self.assertSetEqual(
            set(result.variable_stats.keys()),
            {"temperature_daily_min", "temperature_daily_max"},
        )

        # --- Check TEMP_DAILY_MIN ---
        vmin = result.variable_stats["temperature_daily_min"]
        self.assertEqual(vmin.min_value, -15.0)
        self.assertEqual(vmin.min_value_date, date(1995, 1, 10))
        self.assertEqual(vmin.mean_value, 5.0)
        self.assertEqual(vmin.max_value, 25.0)
        self.assertEqual(vmin.max_value_date, date(1995, 7, 1))
        self.assertEqual(vmin.source_granularity, "daily")
        self.assertEqual(vmin.value_sum, 5000.0)
        self.assertEqual(vmin.value_count, 1000)

        # --- Check TEMP_DAILY_MAX ---
        vmax = result.variable_stats["temperature_daily_max"]
        self.assertEqual(vmax.min_value, -5.0)
        self.assertEqual(vmax.min_value_date, date(1995, 2, 2))
        self.assertEqual(vmax.mean_value, 15.0)
        self.assertEqual(vmax.max_value, 35.0)
        self.assertEqual(vmax.max_value_date, date(1995, 8, 15))
        self.assertEqual(vmax.source_granularity, "daily")
        self.assertEqual(vmax.value_sum, 15000.0)
        self.assertEqual(vmax.value_count, 1000)


class TestNormalizeStops(unittest.TestCase):
    def test_renormalize_stops_identity(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 100, 0, 100)
        self.assertListEqual([s["color"] for s in newstops], ["#000000", "#ffffff"])

    def test_renormalize_stops_invert(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 100, 100, 0)
        self.assertListEqual([s["color"] for s in newstops], ["#ffffff", "#000000"])

    def test_renormalize_stops_negative(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, -1, -1, 0)
        self.assertListEqual([s["color"] for s in newstops], ["#ffffff", "#000000"])

    def test_renormalize_stops_above(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 1, 2, 3)
        self.assertListEqual([s["color"] for s in newstops], ["#ffffff", "#ffffff"])

    def test_renormalize_stops_below(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 1, -2, -3)
        self.assertListEqual([s["color"] for s in newstops], ["#000000", "#000000"])

    def test_renormalize_stops_half(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 1.0, "color": "#ffffff"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 100, 0, 50)
        self.assertListEqual([s["color"] for s in newstops], ["#000000", "#808080"])

    def test_renormalize_stops_multi_red(self):
        stops = [
            {"offset": 0.0, "color": "#000000"},
            {"offset": 0.25, "color": "#400000"},
            {"offset": 0.5, "color": "#800000"},
            {"offset": 0.75, "color": "#c00000"},
            {"offset": 1.0, "color": "#ff0000"},
        ]
        newstops = charts._renormalize_stops(stops, 0, 1, 0.5, 1.5)
        self.assertListEqual(
            [s["color"] for s in newstops],
            ["#800000", "#c00000", "#ff0000", "#ff0000", "#ff0000"],
        )
