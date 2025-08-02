from typing import Any
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import unittest


class PandasTestCase(unittest.TestCase):
    def assertSeriesValuesEqual(self, series: pd.Series, expected_values: list[Any]):
        """Check only the values of a Series (ignore index, dtype, name)."""
        self.assertEqual(series.tolist(), expected_values)

    def assertSeriesEqual(self, actual: pd.Series, expected: pd.Series, **kwargs):
        """Wrapper around assert_series_equal with relaxed defaults.

        If expected is a list, it will be converted to an unnamed pd.Series.
        """
        kwargs.setdefault("check_dtype", False)
        kwargs.setdefault("check_names", False)
        kwargs.setdefault("check_index_type", False)

        if isinstance(expected, list):
            expected = pd.Series(expected)
        assert_series_equal(actual, expected, **kwargs)

    def assertFrameEqual(self, actual: pd.DataFrame, expected: pd.DataFrame, **kwargs):
        """Wrapper around assert_frame_equal with relaxed defaults."""
        kwargs.setdefault("check_dtype", False)
        kwargs.setdefault("check_column_type", False)
        kwargs.setdefault("check_index_type", False)
        assert_frame_equal(actual, expected, **kwargs)

    def assertColumnNames(self, df: pd.DataFrame, expected_names):
        self.assertEqual(df.columns.to_list(), expected_names)
