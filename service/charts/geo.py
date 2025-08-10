from pyproj import Geod
import math

from . import models

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
        s1.coordinates_wgs84_lat,
        s1.coordinates_wgs84_lon,
        s2.coordinates_wgs84_lat,
        s2.coordinates_wgs84_lon,
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
