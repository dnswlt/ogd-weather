import unittest
from . import smn


class TestApp(unittest.TestCase):

    def test_match_csv_resource_real_urls(self):

        m = smn.match_csv_resource(
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_m.csv",
            None,
        )
        self.assertIsNotNone(m)
        self.assertEqual(
            m,
            smn.HrefMatch(
                interval="m",
                frequency="historical",
            ),
        )

        m = smn.match_csv_resource(
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_d_historical.csv",
            None,
        )
        self.assertIsNotNone(m)
        self.assertEqual(
            m,
            smn.HrefMatch(
                interval="d",
                frequency="historical",
            ),
        )

        m = smn.match_csv_resource(
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_d_recent.csv",
            None,
        )
        self.assertIsNotNone(m)
        self.assertEqual(
            m,
            smn.HrefMatch(
                interval="d",
                frequency="recent",
            ),
        )

        m = smn.match_csv_resource(
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ban/ogd-smn_ban_h_historical_2010-2019.csv",
            None,
        )
        self.assertIsNotNone(m)
        self.assertEqual(
            m,
            smn.HrefMatch(
                interval="h",
                frequency="historical",
                years=(2010, 2019),
            ),
        )
