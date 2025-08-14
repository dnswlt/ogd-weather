import math
import pandas as pd
from pydantic import BaseModel
from pyproj import Geod

from . import models
from . import prefix

# Create the geodesic object once (WGS84 ellipsoid)
_GEOD = Geod(ellps="WGS84")


def station_distance_meters(
    s1: models.Station, s2: models.Station, include_height: bool = True
) -> float:
    """
    Compute distance (in meters) between two stations.
    Uses ellipsoidal geodesic for horizontal distance and
    optionally includes height difference.
    """
    if None in (
        s1.coordinates_wgs84_lon,
        s1.coordinates_wgs84_lat,
        s2.coordinates_wgs84_lon,
        s2.coordinates_wgs84_lat,
    ):
        raise ValueError("Both stations must have latitude and longitude set.")

    _, _, dist_horiz = _GEOD.inv(
        s1.coordinates_wgs84_lon,
        s1.coordinates_wgs84_lat,
        s2.coordinates_wgs84_lon,
        s2.coordinates_wgs84_lat,
    )

    if not include_height:
        return dist_horiz

    if None in (s1.height_masl, s2.height_masl):
        raise ValueError("Both stations must have height_masl if include_height=True.")

    height_diff = s1.height_masl - s2.height_masl

    return math.hypot(dist_horiz, height_diff)


class Places:

    def __init__(self, places: list[models.Place]):
        """Creates a new PostalCodes instance from the given list of (postal_code, place_name, lat, lon) values."""
        self._codes = {place.postal_code: place for place in places}
        self._lookup = prefix.PrefixLookup(places, key=lambda p: p.name)

    def __len__(self):
        return len(self._codes)

    def get_postal_code(self, postal_code: str) -> models.Place | None:
        return self._codes.get(postal_code)

    def find_prefix(self, prefix: str, limit: int = 10) -> list[models.Place]:
        return self._lookup.find_prefix(prefix, limit)

    @classmethod
    def from_geonames(cls, path: str) -> "Places":
        """Creates a new instance from a geonames TSV file.

        Duplicate postal codes in the file are ignored (first one wins).
        """
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "country_code",
                "postal_code",
                "place_name",
                "canton_name",
                "canton_code",
                "district_name",
                "district_code",
                "community_name",
                "community_code",
                "latitude_wgs84",
                "longitude_wgs84",
                "accuracy",
            ],
        )
        places = []
        seen = set()
        for t in df.itertuples():
            if t.postal_code in seen:
                continue
            seen.add(t.postal_code)

            places.append(
                models.Place(
                    postal_code=str(t.postal_code),
                    name=t.place_name,
                    lon=t.longitude_wgs84,
                    lat=t.latitude_wgs84,
                )
            )

        return cls(places)


class StationLookup:

    def __init__(self, stations: list[models.Station]):
        self._stations = stations

    def find_nearest(
        self, place: models.Place, limit: int = 3, max_distance_km: float = 100
    ) -> list[models.PlaceNearestStations]:
        """Returns the limit stations closest to the given (lon, lat) coordinates."""

        dists: list[models.StationDistance] = []
        for s in self._stations:
            if None in (s.coordinates_wgs84_lat, s.coordinates_wgs84_lon):
                continue
            _, _, dist_m = _GEOD.inv(
                s.coordinates_wgs84_lon, s.coordinates_wgs84_lat, place.lon, place.lat
            )
            dist_km = dist_m / 1e3
            if dist_km <= max_distance_km:
                dists.append(models.StationDistance(station=s, distance_km=dist_km))

        dists.sort(key=lambda s: (s.distance_km, s.station.name))
        if len(dists) > limit:
            dists = dists[:limit]

        return models.PlaceNearestStations(
            place=place,
            stations=dists,
        )
