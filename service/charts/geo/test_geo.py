import pytest
from pathlib import Path

from . import geo


def test_from_geonames():
    here = Path(__file__).resolve().parent
    places = geo.Places.from_geonames(here / "../datafiles/geo/CH.txt")

    assert len(places) >= 1000

    p = places.get_postal_code("1001")
    assert p is not None
    assert p.name == "Lausanne"
    assert p.lat == pytest.approx(46.516, abs=1e-6)
    assert p.lon == pytest.approx(6.6328, abs=1e-6)


def test_from_swisstopo():
    here = Path(__file__).resolve().parent
    places = geo.Places.from_swisstopo(here / "../datafiles/geo/AMTOVZ_CSV_WGS84.csv")

    assert len(places) >= 1000

    p = places.get_postal_code("3011")
    assert p is not None
    assert p.name == "Bern"
    assert p.lat == pytest.approx(46.9472, abs=1e-3)
    assert p.lon == pytest.approx(7.4480, abs=1e-3)


def test_places_find_prefix():
    here = Path(__file__).resolve().parent
    places = geo.Places.from_geonames(here / "../datafiles/geo/CH.txt")

    zuri = set(p.name for p in places.find_prefix("zuri"))
    assert set(["Zürich"]).issubset(zuri)

    wohlen = set(p.name for p in places.find_prefix("wohlen b."))
    assert "Wohlen b. Bern" in wohlen
