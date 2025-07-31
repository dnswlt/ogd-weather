from datetime import date
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import unittest

from . import db
from . import charts


import unittest
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


class PandasTestCase(unittest.TestCase):
    def assertSeriesValuesEqual(self, series, expected_values):
        """Check only the values of a Series (ignore index, dtype, name)."""
        self.assertEqual(series.tolist(), expected_values)

    def assertSeriesEqual(self, actual, expected, **kwargs):
        """Wrapper around assert_series_equal with relaxed defaults.

        If expected is a list, it will be converted to an unnamed pd.Series.
        """
        kwargs.setdefault("check_dtype", False)
        kwargs.setdefault("check_names", False)
        kwargs.setdefault("check_index_type", False)

        if isinstance(expected, list):
            expected = pd.Series(expected)
        assert_series_equal(actual, expected, **kwargs)

    def assertFrameEqual(self, actual, expected, **kwargs):
        """Wrapper around assert_frame_equal with relaxed defaults."""
        kwargs.setdefault("check_dtype", False)
        kwargs.setdefault("check_column_type", False)
        kwargs.setdefault("check_index_type", False)
        assert_frame_equal(actual, expected, **kwargs)

    def assertColumnNames(self, df, expected_names):
        self.assertEqual(df.columns.to_list(), expected_names)


class TestStationNumDays(PandasTestCase):
    def test_day_count_chart_data(self):
        times = pd.date_range("2022-01-01", periods=2 * 365, freq="d")

        ser = pd.Series(
            data=list(range(len(times))),
            index=times,
        )
        pred = ser >= 100
        data, trend = charts.day_count_chart_data(pred)

        # Want data for two years
        self.assertSeriesEqual(data["year"], [2022, 2023])
        # First year has 100 days that don't satisfy the predicate.
        self.assertSeriesEqual(data["value"], [265, 365])
        # Structural assertions.
        self.assertTrue(set(data["variable"].unique()) == set(["# days"]))
        self.assertColumnNames(data, ["year", "variable", "value"])
        self.assertColumnNames(trend, ["year", "variable", "value"])

        # Values lie on a straight line, trend should be approx. equal.
        self.assertEqual(len(trend), 2)
        self.assertSeriesEqual(trend["value"], [265, 365], check_exact=False)

    def test_day_count_chart(self):
        # Smoke test to verify charts can be generated.
        times = pd.date_range("2022-01-01", periods=2 * 365, freq="d")

        temp = pd.Series(
            data=list(range(len(times))),
            index=times,
        )
        chart = charts.frost_days_chart(
            pd.DataFrame(
                {
                    "station_abbr": ["BER"] * len(times),
                    db.TEMP_DAILY_MIN: temp,
                }
            ),
            "BER",
            period=charts.PERIOD_ALL,
        )

        self.assertIsInstance(chart, dict)
        # Dict should have some typical Vega-Lite fields
        self.assertIn("layer", chart.keys())
        self.assertIn("datasets", chart.keys())

    def test_day_count_chart_data_all_false(self):
        """Tests that years don't get removed if they have a zero count."""
        times = pd.date_range("2022-01-01", periods=2 * 365, freq="d")

        ser = pd.Series(
            data=[0] * len(times),
            index=times,
        )
        pred = ser > 0
        data, _ = charts.day_count_chart_data(pred)
        # Want 0 data for two years
        self.assertSeriesEqual(data["year"], [2022, 2023])
        self.assertSeriesEqual(data["value"], [0, 0])


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
