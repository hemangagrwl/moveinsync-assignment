from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_M = 6_371_008.8
METRES_PER_DEGREE = EARTH_RADIUS_M * math.pi / 180.0

FloatArray = NDArray[np.float64]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def haversine_np(lat1: FloatArray, lon1: FloatArray, lat2: FloatArray, lon2: FloatArray) -> FloatArray:
    """Vectorised great-circle distance in metres."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def is_valid_coordinate(lat: float | None, lon: float | None) -> bool:
    """Finite, inside WGS84 bounds and not the (0, 0) "null island" that broken devices emit."""
    if lat is None or lon is None:
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    return not (abs(lat) < 1e-6 and abs(lon) < 1e-6)


def valid_coordinates_np(lat: FloatArray, lon: FloatArray) -> NDArray[np.bool_]:
    finite = np.isfinite(lat) & np.isfinite(lon)
    with np.errstate(invalid="ignore"):
        in_range = (np.abs(lat) <= 90.0) & (np.abs(lon) <= 180.0)
        null_island = (np.abs(lat) < 1e-6) & (np.abs(lon) < 1e-6)
    return finite & in_range & ~null_island


def speed_kmh(distance_m: float, dt_s: float, min_dt_s: float) -> float:
    return distance_m / 1000.0 / (max(dt_s, min_dt_s) / 3600.0)


def speed_kmh_np(distance_m: FloatArray, dt_s: FloatArray, min_dt_s: float) -> FloatArray:
    return distance_m / 1000.0 / (np.maximum(dt_s, min_dt_s) / 3600.0)
