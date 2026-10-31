from __future__ import annotations

import math

import numpy as np
import pytest

from backfill.core.cleaning import PingCleaner, RejectReason
from backfill.core.config import RuleConfig
from backfill.core.geo import haversine_m
from backfill.core.models import GeoPoint, PingBatch
from tests.helpers import LAT0, LON0, T0, Drive, east


@pytest.fixture
def cleaner() -> PingCleaner:
    return PingCleaner(RuleConfig(min_move_m=0))


def batch(rows: list[tuple[int, float, float, float | None]]) -> PingBatch:
    return PingBatch.from_rows([(T0 + ts, lat, lon, acc) for ts, lat, lon, acc in rows])


def test_haversine_known_distance() -> None:
    # One degree of latitude on the mean-radius sphere.
    assert haversine_m(12.0, 77.0, 13.0, 77.0) == pytest.approx(111_195.08, abs=0.01)


def test_sorts_out_of_order_pings(cleaner: PingCleaner) -> None:
    p = east(GeoPoint(LAT0, LON0), 40)
    out = cleaner.clean(
        batch([(8000, p.lat, p.lon, 5), (0, LAT0, LON0, 5), (4000, LAT0, (LON0 + p.lon) / 2, 5)])
    )
    assert out.ts_ms.tolist() == [T0, T0 + 4000, T0 + 8000]


def test_duplicate_timestamp_keeps_most_accurate(cleaner: PingCleaner) -> None:
    near = east(GeoPoint(LAT0, LON0), 3)
    out = cleaner.clean(batch([(0, near.lat, near.lon, 20), (0, LAT0, LON0, 4), (0, LAT0, LON0, 4)]))
    assert len(out) == 1
    assert out.lon[0] == pytest.approx(LON0)
    assert out.rejected[RejectReason.DUPLICATE] == 2


def test_duplicate_survivor_does_not_depend_on_arrival_order(cleaner: PingCleaner) -> None:
    a = east(GeoPoint(LAT0, LON0), 3)
    rows = [(0, LAT0, LON0, 5.0), (0, a.lat, a.lon, 5.0)]
    first = cleaner.clean(batch(rows))
    second = cleaner.clean(batch(rows[::-1]))
    assert first.lon.tolist() == second.lon.tolist()


def test_drops_low_accuracy_but_keeps_unreported_accuracy(cleaner: PingCleaner) -> None:
    out = cleaner.clean(batch([(0, LAT0, LON0, 80), (4000, LAT0, LON0, None), (8000, LAT0, LON0, 50)]))
    assert len(out) == 2
    assert out.rejected[RejectReason.LOW_ACCURACY] == 1


@pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (math.nan, 77.0), (95.0, 77.0), (12.0, 190.0)])
def test_drops_invalid_coordinates(cleaner: PingCleaner, lat: float, lon: float) -> None:
    out = cleaner.clean(batch([(0, LAT0, LON0, 5), (4000, lat, lon, 5)]))
    assert len(out) == 1
    assert out.rejected[RejectReason.INVALID_COORDINATE] == 1


def test_drops_pings_outside_trip_window(cleaner: PingCleaner) -> None:
    out = cleaner.clean(
        batch([(-5000, LAT0, LON0, 5), (0, LAT0, LON0, 5), (9000, LAT0, LON0, 5)]), (T0, T0 + 5000)
    )
    assert len(out) == 1
    assert out.rejected[RejectReason.OUT_OF_WINDOW] == 2


def test_bad_fix_burst_9km_away_is_dropped(cleaner: PingCleaner) -> None:
    """Spec example: a 4-second burst of pings 9 km away implies ~8,000 km/h and is discarded."""
    d = Drive(GeoPoint(LAT0, LON0), speed_ms=10, every_s=4).leg(2000)
    clean_pings = d.pings()
    far = east(GeoPoint(LAT0, LON0), 9000)
    burst_ts = clean_pings.ts_ms[20] + np.array([500, 1500, 2500, 3500])
    burst = PingBatch.from_arrays(burst_ts, np.full(4, far.lat), np.full(4, far.lon), np.full(4, 5.0))
    out = cleaner.clean(clean_pings.concat(burst))
    assert out.rejected[RejectReason.TELEPORT] == 4
    assert len(out) == len(clean_pings)


def test_bad_first_fix_does_not_discard_the_whole_trip(cleaner: PingCleaner) -> None:
    d = Drive(GeoPoint(LAT0, LON0), speed_ms=10, every_s=4).leg(1000)
    good = d.pings()
    far = east(GeoPoint(LAT0, LON0), 20_000)
    bad_first = PingBatch.from_arrays([good.ts_ms[0] - 1000], [far.lat], [far.lon], [5.0])
    out = cleaner.clean(bad_first.concat(good))
    assert len(out) >= len(good)


def test_stationary_jitter_collapses() -> None:
    rng = np.random.default_rng(1)
    n = 60
    lat = LAT0 + rng.normal(0, 2.0, n) / 111_195
    lon = LON0 + rng.normal(0, 2.0, n) / 111_195
    pings = PingBatch.from_arrays(T0 + np.arange(n) * 2000, lat, lon, np.full(n, 5.0))
    out = PingCleaner(RuleConfig(min_move_m=8)).clean(pings)
    assert len(out) <= 4
    assert out.rejected[RejectReason.STATIONARY_JITTER] >= n - 4


def test_jitter_filter_keeps_a_moving_track() -> None:
    d = Drive(GeoPoint(LAT0, LON0), speed_ms=8, every_s=4).leg(3000)
    out = PingCleaner(RuleConfig(min_move_m=8)).clean(d.pings())
    assert out.rejected[RejectReason.STATIONARY_JITTER] == 0
