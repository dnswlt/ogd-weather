import numpy as np
import pandas as pd

from service.charts.testutils.testhelpers import PandasTestCase
from service.charts.db import constants as dc

from . import transform as tf


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
                dc.PRECIP_DAILY_MM,
            ],
        )
        df = df.set_index("reference_timestamp")
        df.index = pd.to_datetime(df.index)
        data_long, trend_long = tf.timeline_years_chart_data(df, "sum", window=1)
        self.assertIsNotNone(data_long)
        self.assertIsNotNone(trend_long)

        self.assertColumnNames(data_long, ["year", "measurement", "value"])
        self.assertSetEqual(
            set(data_long["measurement"].unique()),
            set([dc.PRECIP_DAILY_MM]),
        )
        values = data_long[data_long["measurement"] == dc.PRECIP_DAILY_MM]["value"]
        self.assertSeriesValuesEqual(values, [6, 0, 12])

        trend = trend_long[trend_long["measurement"] == dc.PRECIP_DAILY_MM]["value"]
        self.assertSeriesValuesEqual(trend, [3, 6, 9])


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
        coeffs, trend = tf.polyfit_columns(df, deg=1)
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
            tf.polyfit_columns(df, deg=1)
