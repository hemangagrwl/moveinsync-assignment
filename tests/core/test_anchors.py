from __future__ import annotations

import numpy as np
import pytest

from backfill.core.anchors import (
    Anchor,
    AnchorBuilder,
    AnchorKind,
    AnchorPlacer,
    AnchorSource,
    AnchorStatus,
)
from backfill.core.config import RuleConfig
from backfill.core.models import GeoPoint, PingBatch, Rider, RiderStatus, RiderType, TripType
from tests.helpers import LAT0, LON0, T0, east, evidence, north

OFFICE = GeoPoint(LAT0, LON0)
HOME = north(OFFICE, 5000)
END = T0 + 3_600_000


def build(riders: tuple[Rider, ...], cfg: RuleConfig | None = None, **kw: object):  # type: ignore[no-untyped-def]
    ev = evidence(PingBatch.empty(), end_ms=END, office=OFFICE, riders=riders, **kw)  # type: ignore[arg-type]
    return AnchorBuilder(cfg or RuleConfig()).build(ev)


def rider_anchor(result) -> Anchor:  # type: ignore[no-untyped-def]
    found = [a for a in result.anchors if a.ref.startswith("E")]
    assert len(found) == 1
    return found[0]


def test_valid_actual_drop_is_used() -> None:
    actual = east(HOME, 20)
    r = Rider(
        "E1",
        planned_location=HOME,
        planned_time_ms=T0 + 1_000_000,
        actual_location=actual,
        actual_time_ms=T0 + 1_100_000,
    )
    a = rider_anchor(build((r,)))
    assert (a.kind, a.source, a.ts_ms) == (AnchorKind.DROP, AnchorSource.ACTUAL, T0 + 1_100_000)


@pytest.mark.parametrize(
    "actual",
    [None, GeoPoint(0.0, 0.0), north(HOME, 3000)],  # missing, null island, too far from plan
    ids=["missing", "null-island", "too-far"],
)
def test_planned_rider_falls_back_to_planned_point(actual: GeoPoint | None) -> None:
    r = Rider(
        "E1",
        planned_location=HOME,
        planned_time_ms=T0 + 1_000_000,
        actual_location=actual,
        actual_time_ms=T0 + 1_100_000,
    )
    a = rider_anchor(build((r,)))
    assert (a.source, a.lat, a.ts_ms) == (AnchorSource.PLANNED, HOME.lat, T0 + 1_000_000)


def test_adhoc_rider_falls_back_to_profile_home_and_sign_in_time() -> None:
    r = Rider(
        "E1",
        rider_type=RiderType.ADHOC,
        profile_home=HOME,
        sign_in_time_ms=T0 + 900_000,
        actual_location=None,
    )
    a = rider_anchor(build((r,)))
    assert (a.source, a.ts_ms) == (AnchorSource.PROFILE_HOME, T0 + 900_000)


def test_adhoc_far_actual_is_checked_against_profile_home() -> None:
    r = Rider(
        "E1",
        rider_type=RiderType.ADHOC,
        profile_home=HOME,
        sign_in_time_ms=T0 + 900_000,
        actual_location=north(HOME, 6000),
        actual_time_ms=T0 + 950_000,
    )
    assert rider_anchor(build((r,))).source is AnchorSource.PROFILE_HOME


def test_rider_with_no_valid_point_is_reported() -> None:
    r = Rider("E1", planned_location=None, actual_location=GeoPoint(0, 0), actual_time_ms=T0)
    result = build((r,))
    assert result.riders_without_anchor == ("E1",)


def test_event_time_far_outside_trip_is_invalid() -> None:
    r = Rider(
        "E1",
        planned_location=HOME,
        planned_time_ms=T0 + 1_000_000,
        actual_location=HOME,
        actual_time_ms=END + 3_600_000,
    )
    assert rider_anchor(build((r,))).source is AnchorSource.PLANNED


def test_no_show_anchor_is_behind_a_flag() -> None:
    r = Rider(
        "E1",
        status=RiderStatus.NO_SHOW,
        planned_location=HOME,
        planned_time_ms=T0 + 600_000,
        no_show_location=east(HOME, 30),
        no_show_time_ms=T0 + 650_000,
    )
    assert [a for a in build((r,)).anchors if a.ref == "E1"] == []
    a = rider_anchor(build((r,), RuleConfig(include_no_show_anchors=True)))
    assert (a.kind, a.source, a.ts_ms) == (AnchorKind.NO_SHOW, AnchorSource.NO_SHOW_REPORTED, T0 + 650_000)


def test_invalid_no_show_location_uses_planned_location() -> None:
    r = Rider(
        "E1",
        status=RiderStatus.NO_SHOW,
        planned_location=HOME,
        planned_time_ms=T0 + 600_000,
        no_show_location=GeoPoint(0, 0),
        no_show_time_ms=T0 + 650_000,
    )
    a = rider_anchor(build((r,), RuleConfig(include_no_show_anchors=True)))
    assert (a.source, a.lat) == (AnchorSource.PLANNED, HOME.lat)


@pytest.mark.parametrize(
    "offset_m,source",
    [(399, AnchorSource.OFFICE_ACTUAL), (401, AnchorSource.OFFICE_PLANNED)],
)
def test_login_office_anchor_uses_trip_end_only_within_400m(offset_m: float, source: AnchorSource) -> None:
    end_point = north(OFFICE, offset_m)
    result = build((), trip_type=TripType.LOGIN, end_location=end_point)
    office = next(a for a in result.anchors if a.kind is AnchorKind.OFFICE)
    assert office.source is source
    assert office.ts_ms == END
    expected_lat = end_point.lat if source is AnchorSource.OFFICE_ACTUAL else OFFICE.lat
    assert office.lat == pytest.approx(expected_lat)


def test_logout_office_anchor_is_at_trip_start() -> None:
    result = build((), trip_type=TripType.LOGOUT, start_location=north(OFFICE, 50))
    office = next(a for a in result.anchors if a.kind is AnchorKind.OFFICE)
    assert (office.source, office.ts_ms) == (AnchorSource.OFFICE_ACTUAL, T0)


# -- placement --------------------------------------------------------------------------------


def track(points: list[tuple[int, GeoPoint]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([T0 + t for t, _ in points], dtype=np.int64),
        np.array([p.lat for _, p in points]),
        np.array([p.lon for _, p in points]),
    )


def anchor(source: AnchorSource, point: GeoPoint, t: int) -> Anchor:
    return Anchor(AnchorKind.DROP, source, point.lat, point.lon, T0 + t, "E1")


def test_anchor_on_the_gps_track_is_redundant() -> None:
    ts, lat, lon = track([(0, OFFICE), (30_000, HOME), (60_000, east(HOME, 300))])
    d = AnchorPlacer(RuleConfig()).place(
        (anchor(AnchorSource.PLANNED, east(HOME, 50), 200_000),), ts, lat, lon
    )
    assert d[0].status is AnchorStatus.REDUNDANT


def test_fallback_anchor_is_retimed_into_the_gap_proportionally() -> None:
    a_pt, b_pt = OFFICE, east(OFFICE, 4000)
    drop = east(OFFICE, 1000)  # a quarter of the way along the gap
    ts, lat, lon = track([(0, a_pt), (400_000, b_pt)])
    d = AnchorPlacer(RuleConfig()).place((anchor(AnchorSource.PLANNED, drop, 450_000),), ts, lat, lon)
    assert d[0].status is AnchorStatus.RETIMED
    assert d[0].anchor.ts_ms == pytest.approx(T0 + 100_000, abs=1_000)
    assert d[0].original_ts_ms == T0 + 450_000


def test_trusted_actual_anchor_keeps_its_time() -> None:
    ts, lat, lon = track([(0, OFFICE), (400_000, east(OFFICE, 4000))])
    a = anchor(AnchorSource.ACTUAL, north(OFFICE, 900), 123_000)
    d = AnchorPlacer(RuleConfig()).place((a,), ts, lat, lon)
    assert (d[0].status, d[0].anchor.ts_ms) == (AnchorStatus.USED, T0 + 123_000)
