from __future__ import annotations

import numpy as np
import pytest

from backfill.core.config import RuleConfig
from backfill.core.geo import speed_kmh_np
from backfill.core.rules import (
    ParkedRule,
    RoutedGapPolicy,
    RuleCode,
    SegmentBatch,
    ShortGapAerialRule,
)

CFG = RuleConfig(speed_tolerance_m=0.0)


def seg(dt_s: float, aerial_m: float) -> SegmentBatch:
    dt = np.array([dt_s])
    d = np.array([aerial_m])
    return SegmentBatch(dt, d, speed_kmh_np(d, dt, CFG.min_speed_dt_s))


@pytest.mark.parametrize(
    "dt,dist,expected",
    [
        (69.9, 500.0, True),  # just under 70 s, 25.8 km/h
        (70.0, 500.0, False),  # 70 s is no longer a short gap
        (10.0, 400.0, False),  # 144 km/h is not plausible
        (10.0, 333.0, True),  # 119.9 km/h is
    ],
)
def test_short_gap_aerial(dt: float, dist: float, expected: bool) -> None:
    assert bool(ShortGapAerialRule(CFG).claims(seg(dt, dist))[0]) is expected


@pytest.mark.parametrize(
    "dt,dist,expected",
    [
        (600.0, 100.0, True),  # 0.6 km/h over 10 minutes: parked
        (600.0, 833.0, True),  # 4.998 km/h
        (600.0, 834.0, False),  # 5.004 km/h: moving, needs routing
        (60.0, 10.0, False),  # short gaps are never PARKED
    ],
)
def test_parked(dt: float, dist: float, expected: bool) -> None:
    assert bool(ParkedRule(CFG).claims(seg(dt, dist))[0]) is expected


def test_provider_failure_uses_aerial_times_1_4() -> None:
    v = RoutedGapPolicy(CFG).judge(aerial_m=1000.0, dt_s=300.0, routed_m=None)
    assert v.code is RuleCode.DEGRADED_AERIAL
    assert v.distance_m == pytest.approx(1400.0)


def test_route_over_100km_is_rejected() -> None:
    v = RoutedGapPolicy(CFG).judge(aerial_m=60_000.0, dt_s=7200.0, routed_m=100_001.0)
    assert (v.code, v.distance_m) == (RuleCode.REJECTED_MAX_DISTANCE, 0.0)


def test_route_rejected_when_routed_and_fallback_both_too_fast() -> None:
    # 10 km in 60 s: routed 720 km/h, aerial*1.4 = 11.2 km -> 672 km/h.
    v = RoutedGapPolicy(CFG).judge(aerial_m=8000.0, dt_s=60.0, routed_m=10_000.0)
    assert (v.code, v.distance_m) == (RuleCode.REJECTED_SPEED, 0.0)


def test_route_kept_when_only_routed_is_too_fast() -> None:
    # routed 5 km in 120 s = 150 km/h, but aerial*1.4 = 2.8 km -> 84 km/h is plausible.
    v = RoutedGapPolicy(CFG).judge(aerial_m=2000.0, dt_s=120.0, routed_m=5000.0)
    assert (v.code, v.distance_m) == (RuleCode.ROUTED, 5000.0)


def test_normal_route_is_accepted() -> None:
    v = RoutedGapPolicy(CFG).judge(aerial_m=2000.0, dt_s=600.0, routed_m=2600.0)
    assert (v.code, v.distance_m) == (RuleCode.ROUTED, 2600.0)


def test_rule_version_changes_with_any_threshold() -> None:
    assert RuleConfig().rule_version != RuleConfig(short_gap_s=60).rule_version
    assert RuleConfig().rule_version == RuleConfig().rule_version
