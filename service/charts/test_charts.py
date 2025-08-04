import altair as alt
from datetime import date
import datetime
import numpy as np
import pandas as pd
import unittest


from . import charts
from . import db
from .testhelpers import PandasTestCase


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
        self.assertSeriesValuesEqual(data["year"], [2022, 2023])
        # First year has 100 days that don't satisfy the predicate.
        self.assertSeriesValuesEqual(data["value"], [265, 365])
        # Structural assertions.
        self.assertTrue(set(data["measurement"].unique()) == set(["# days"]))
        self.assertColumnNames(data, ["year", "measurement", "value"])
        self.assertColumnNames(trend, ["year", "measurement", "value"])

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

        self.assertIsInstance(chart, alt.LayerChart)

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
        self.assertSeriesValuesEqual(data["year"], [2022, 2023])
        self.assertSeriesValuesEqual(data["value"], [0, 0])


class TestPolyfitColumns(PandasTestCase):

    def test_polyfit_colums_exact(self):
        # polyfit_columns with deg=1 should fit an actual line accurately.
        x = np.linspace(0, 10, 11)
        df = pd.DataFrame(
            dict(
                y1=3 * x + 2,
                y2=-1 * x,
            ),
            index=x,
        )
        coeffs, trend = charts.polyfit_columns(df, deg=1)
        # Should recover the original line parameters.
        self.assertFrameEqual(
            coeffs,
            pd.DataFrame([[2.0, 0], [3.0, -1.0]], columns=["y1", "y2"]),
            check_exact=False,
        )
        # Trend line should be equal to actual line.
        self.assertFrameEqual(
            trend,
            pd.DataFrame(
                {
                    "y1": 3 * x + 2,
                    "y2": -x,
                }
            ),
            check_exact=False,
        )

    def test_polyfit_colums_single_value(self):
        # Fitting a curve to just a single point is... hard.
        df = pd.DataFrame(
            dict(
                y1=[1],
            ),
            index=[0],
        )
        with self.assertRaises(ValueError):
            charts.polyfit_columns(df, deg=1)


class TestStationStats(PandasTestCase):

    def test_station_stats_temp_increase(self):
        df = pd.DataFrame(
            [
                ["2025-01-01", "BER", -2, -1, 0, 0],
                ["2025-01-02", "BER", -2, -1, 0, 0],
                ["2026-01-01", "BER", -10, 1, 3, 2.5],
            ],
            columns=[
                "reference_timestamp",
                "station_abbr",
                db.TEMP_DAILY_MIN,
                db.TEMP_DAILY_MEAN,
                db.TEMP_DAILY_MAX,
                db.PRECIP_DAILY_MM,
            ],
        )
        df = df.set_index("reference_timestamp")
        df.index = pd.to_datetime(df.index)
        s = charts.station_stats(df, "BER", period="1")
        self.assertEqual(s.first_date, datetime.date(2025, 1, 1))
        self.assertEqual(s.last_date, datetime.date(2026, 1, 1))
        # Cannot calculate temp increase from a single year of data:
        self.assertAlmostEqual(s.annual_temp_increase, 2.0)
        self.assertEqual(s.period, "January")

    def test_station_stats_single_year(self):
        df = pd.DataFrame(
            [
                ["2025-01-01", "BER", -2, -1, 0, 0],
                ["2025-01-02", "BER", -2, -1, 0, 0],
            ],
            columns=[
                "reference_timestamp",
                "station_abbr",
                db.TEMP_DAILY_MIN,
                db.TEMP_DAILY_MEAN,
                db.TEMP_DAILY_MAX,
                db.PRECIP_DAILY_MM,
            ],
        )
        df = df.set_index("reference_timestamp")
        df.index = pd.to_datetime(df.index)
        s = charts.station_stats(df, "BER", period="1")
        # Cannot calculate temp increase from a single year of data:
        self.assertIsNone(s.annual_temp_increase)


class TestTimelineYearsChartData(PandasTestCase):

    def test_timeline_years_chart_data_precip(self):
        df = pd.DataFrame(
            [
                ["2024-01-01", 0],
                ["2024-01-02", 2],
                ["2024-01-03", 4],
                ["2025-01-01", 0],
                ["2025-01-02", 0],
                ["2025-01-03", 0],
                ["2026-01-01", 12],
            ],
            columns=[
                "reference_timestamp",
                db.PRECIP_DAILY_MM,
            ],
        )
        df = df.set_index("reference_timestamp")
        df.index = pd.to_datetime(df.index)
        data_long, trend_long = charts.timeline_years_chart_data(df, "sum", window=1)
        self.assertIsNotNone(data_long)
        self.assertIsNotNone(trend_long)

        self.assertColumnNames(data_long, ["year", "measurement", "value"])
        self.assertSetEqual(
            set(data_long["measurement"].unique()),
            set([db.PRECIP_DAILY_MM]),
        )
        values = data_long[data_long["measurement"] == db.PRECIP_DAILY_MM]["value"]
        self.assertSeriesValuesEqual(values, [6, 0, 12])

        trend = trend_long[trend_long["measurement"] == db.PRECIP_DAILY_MM]["value"]
        self.assertSeriesValuesEqual(trend, [3, 6, 9])


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
                    "p10_value": 0.1,
                    "p25_value": 0.25,
                    "median_value": 0.50,
                    "p75_value": 0.75,
                    "p90_value": 0.9,
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
                    "p10_value": 0.1,
                    "p25_value": 0.25,
                    "median_value": 0.50,
                    "p75_value": 0.75,
                    "p90_value": 0.9,
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
        self.assertEqual(vmin.p10_value, 0.1)
        self.assertEqual(vmin.p25_value, 0.25)
        self.assertEqual(vmin.median_value, 0.5)
        self.assertEqual(vmin.p75_value, 0.75)
        self.assertEqual(vmin.p90_value, 0.9)

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
