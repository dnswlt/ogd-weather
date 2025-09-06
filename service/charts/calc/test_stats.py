import datetime
import pandas as pd
import unittest

from service.charts.db import constants as dc

from . import stats


class TestStationPeriodStats(unittest.TestCase):
    def test_variable_stats_dict(self):
        # Wide DataFrame with two variables
        df = pd.DataFrame(
            [
                {
                    "variable": dc.TEMP_DAILY_MIN,
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
                    "variable": dc.TEMP_DAILY_MAX,
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
        result = stats.station_period_stats(s)

        # Check we got both variables in the dict
        self.assertSetEqual(
            set(result.variable_stats.keys()),
            {"temperature_daily_min", "temperature_daily_max"},
        )

        # --- Check TEMP_DAILY_MIN ---
        vmin = result.variable_stats["temperature_daily_min"]
        self.assertEqual(vmin.min_value, -15.0)
        self.assertEqual(vmin.min_value_date, datetime.date(1995, 1, 10))
        self.assertEqual(vmin.mean_value, 5.0)
        self.assertEqual(vmin.max_value, 25.0)
        self.assertEqual(vmin.max_value_date, datetime.date(1995, 7, 1))
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
        self.assertEqual(vmax.min_value_date, datetime.date(1995, 2, 2))
        self.assertEqual(vmax.mean_value, 15.0)
        self.assertEqual(vmax.max_value, 35.0)
        self.assertEqual(vmax.max_value_date, datetime.date(1995, 8, 15))
        self.assertEqual(vmax.source_granularity, "daily")
        self.assertEqual(vmax.value_sum, 15000.0)
        self.assertEqual(vmax.value_count, 1000)
