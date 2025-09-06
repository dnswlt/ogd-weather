import altair as alt
from datetime import date
import datetime
import numpy as np
import pandas as pd
import unittest


from service.charts.db import constants as dc
from service.charts.base.errors import NoDataError
from service.charts.testutils import PandasTestCase

from . import charts
from . import transform as tf


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
                    dc.TEMP_DAILY_MIN: temp,
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


class TestStationStats(PandasTestCase):

    def test_station_stats_temp_increase(self):
        df = pd.DataFrame(
            [
                ["2025-01-01", "BER", -1, 0],
                ["2025-01-02", "BER", -1, 0],
                ["2026-01-01", "BER", 1, 2.5],
            ],
            columns=[
                "reference_timestamp",
                "station_abbr",
                tf.TEMP_MEAN,
                tf.PRECIP_MM,
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
                ["2025-01-01", "BER", -1, 0],
                ["2025-01-02", "BER", -1, 0],
            ],
            columns=[
                "reference_timestamp",
                "station_abbr",
                tf.TEMP_MEAN,
                tf.PRECIP_MM,
            ],
        )
        df = df.set_index("reference_timestamp")
        df.index = pd.to_datetime(df.index)
        s = charts.station_stats(df, "BER", period="1")
        # Cannot calculate temp increase from a single year of data:
        self.assertIsNone(s.annual_temp_increase)


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


class TestDryWetSpellsChartData(PandasTestCase):

    def _input_df(self, *args):
        df = pd.DataFrame(
            args,
            columns=[
                "reference_timestamp",
                "station_abbr",
                dc.PRECIP_DAILY_MM,
                dc.SUNSHINE_DAILY_PCT_OF_MAX,
            ],
        )
        df["reference_timestamp"] = pd.to_datetime(df["reference_timestamp"])
        df = df.set_index("reference_timestamp")
        return df

    def test_drywet_spells_bar_chart_data_empty(self):
        df = self._input_df()

        with self.assertRaises(NoDataError):
            charts.drywet_spells_bar_chart_data(df, "BER", 2025)

    def test_drywet_spells_bar_chart_single_date(self):
        df = self._input_df(
            ("2025-01-01", "BER", 17.0, 2.0),
        )

        ts = charts.drywet_spells_bar_chart_data(df, "BER", 2025, min_days=1)
        self.assertSeriesValuesEqual(ts["category"], ["wet"])
        self.assertSeriesValuesEqual(ts["duration_days"], [1])
        self.assertSeriesValuesEqual(ts.index, [pd.to_datetime("2025-01-01")])

    def test_drywet_spells_bar_chart_data_last_spell(self):
        df = self._input_df(
            ("2025-01-01", "BER", 17.0, 2.0),
            ("2025-01-02", "BER", 0.19, 80.0),
            ("2025-01-03", "BER", 0.19, 80.0),
            ("2025-01-04", "BER", 0.19, 80.0),
        )

        top_spells = charts.drywet_spells_bar_chart_data(df, "BER", 2025, min_days=3)
        self.assertEqual(len(top_spells), 1)

    def test_drywet_spells_bar_chart_data_correctness(self):
        df = self._input_df(
            ("2025-01-01", "BER", 0.19, 80.0),
            ("2025-01-02", "BER", 0.19, 80.0),
            ("2025-01-03", "BER", 0.19, 80.0),
        )
        top_spells = charts.drywet_spells_bar_chart_data(df, "BER", 2025, min_days=1)
        expected = pd.DataFrame(
            [
                {
                    "reference_timestamp": pd.to_datetime("2025-01-01"),
                    "category": "dry",
                    "duration_days": 3,
                }
            ]
        ).set_index("reference_timestamp")

        self.assertFrameEqual(
            top_spells,
            expected,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
        )

    def test_drywet_spells_bar_chart_data_fill_dates(self):
        # Should fill in missing dates with NAN, interrupting spells.
        df = self._input_df(
            ("2025-01-01", "BER", 25, 0),
            ("2025-01-02", "BER", 25, 0),
            # -03 is missing
            ("2025-01-04", "BER", 25, 0),
            ("2025-01-05", "BER", 25, 0),
        )

        top_spells = charts.drywet_spells_bar_chart_data(df, "BER", 2025, min_days=3)

        self.assertTrue(top_spells.empty)

    def test_drywet_spells_bar_chart_data_unsorted(self):
        # Should deal with unsorted index
        df = self._input_df(
            ("2025-01-03", "BER", 25, 0),
            ("2025-01-02", "BER", 25, 0),
            ("2025-01-01", "BER", 25, 0),
        )

        top_spells = charts.drywet_spells_bar_chart_data(df, "BER", 2025, min_days=3)

        self.assertEqual(len(top_spells), 1)

        index_sorted = all(
            a <= b for a, b in zip(top_spells.index, top_spells.index[1:])
        )
        self.assertTrue(index_sorted)


class TestFindSpells(PandasTestCase):

    def _input(self, dates_vals):
        if not dates_vals:
            return pd.Series([], index=pd.DatetimeIndex([]))
        dates, vals = zip(*dates_vals)
        return pd.Series(vals, index=pd.to_datetime(dates))

    def _expected(self, inputs):
        dates, cats, durations = zip(*inputs)
        return pd.DataFrame(
            {
                "category": cats,
                "duration_days": durations,
            },
            index=pd.to_datetime(dates),
        )

    def test_find_spells_edge_cases(self):
        # Empty
        sp = charts.find_spells(pd.Series([], index=pd.DatetimeIndex([])))
        self.assertTrue(sp.empty)

        # Single value
        r1 = self._input([("2025-12-31", 0)])
        sp = charts.find_spells(r1, min_days=1)
        self.assertFrameEqual(
            sp,
            self._expected([("2025-12-31", 0, 1)]),
        )

    def test_find_spells_all_equal(self):

        # All values identical
        r = self._input(
            [
                ("2025-12-30", 1),
                ("2025-12-31", 1),
            ]
        )
        self.assertFrameEqual(
            charts.find_spells(r, min_days=2),
            self._expected(
                [
                    ("2025-12-30", 1, 2),
                ]
            ),
        )

    def test_find_spells_all_zero(self):

        # All values identically zero
        dates = pd.date_range("2012-01-01", "2012-12-31", freq="D")
        n = len(dates)
        self.assertEqual(n, 366)

        r = self._input(zip(dates, [0] * n))
        self.assertFrameEqual(
            charts.find_spells(r, min_days=n),
            self._expected(
                [
                    ("2012-01-01", 0, n),
                ]
            ),
        )

    def test_find_spells_all_different(self):

        # All values different
        dates = pd.date_range("2012-01-01", "2012-01-31", freq="D")
        vals = list(range(len(dates)))

        r = self._input(zip(dates, vals))

        # Each spell should have length 1:
        self.assertFrameEqual(
            charts.find_spells(r, min_days=1),
            self._expected(zip(dates, vals, [1] * len(dates))),
        )
        # With min_days=2, we should not get anything.
        self.assertTrue(charts.find_spells(r, min_days=2).empty)

    def test_find_spells_gaps(self):

        # Gaps in the date index.
        r = self._input(
            [
                ("2025-01-01", 0),
                ("2025-01-02", 1),
                ("2025-01-03", 1),
                ("2025-01-10", 1),
                ("2025-01-15", 2),
                ("2025-01-16", 2),
                ("2025-01-17", 2),
            ]
        )
        self.assertFrameEqual(
            charts.find_spells(r, min_days=2),
            self._expected(
                [
                    ("2025-01-02", 1, 2),
                    ("2025-01-15", 2, 3),
                ]
            ),
        )

    def test_find_spells_index(self):
        # Should retain the index name

        # Empty
        runs = self._input([])
        runs.index.name = "reference_timestamp"

        sp = charts.find_spells(runs, min_days=100)
        self.assertTrue(sp.index.name, "reference_timestamp")

        # Non empty
        runs = self._input(
            [
                ("2023-01-01", 1),
                ("2023-01-02", 1),
                ("2023-01-03", 1),
            ]
        )
        runs.index.name = "reference_timestamp"

        sp = charts.find_spells(runs, min_days=3)

        self.assertIsInstance(sp.index, pd.DatetimeIndex)
        self.assertTrue(sp.index.name, "reference_timestamp")
