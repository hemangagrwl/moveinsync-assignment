from __future__ import annotations

import numpy as np
import pytest

from backfill.core.config import RuleConfig
from backfill.core.engine import TripStatus, metres_to_invoice_km
from backfill.core.models import GeoPoint, PingBatch, Rider
from backfill.core.rules import RuleCode
from backfill.providers.breaker import CircuitBreaker
from backfill.providers.cache import RouteCache
from backfill.sim.generator import GenConfig, TripGenerator, naive_distance_m
from tests.helpers import LAT0, LON0, T0, Drive, east, evidence, make_engine


def logout_trip_with_gps_loss() -> tuple[Drive, list[tuple[float, float]], tuple[Rider, ...]]:
    """~31 km logout trip on a street grid. GPS drops out for 22 minutes in total
    (an 8-minute underpass and a 14-minute basement), and two drops happen in the dark."""
    d = Drive(GeoPoint(LAT0, LON0), speed_ms=8.5, every_s=4)
    d.leg(5000).leg(2500, "north")  # drop A is up a side street
    drop_a, t_a = d.at()
    d.wait(90).leg(-2500, "north").leg(6000).leg(4000, "north")
    drop_b, t_b = d.at()
    d.wait(90).leg(2000).leg(-4000, "north").leg(5000)
    drop_c, t_c = d.at()
    d.wait(60)
    riders = (
        Rider(
            "E-A", planned_location=drop_a, planned_time_ms=t_a, actual_location=drop_a, actual_time_ms=t_a
        ),
        Rider(
            "E-B", planned_location=drop_b, planned_time_ms=t_b, actual_location=drop_b, actual_time_ms=t_b
        ),
        Rider(
            "E-C", planned_location=drop_c, planned_time_ms=t_c, actual_location=drop_c, actual_time_ms=t_c
        ),
    )
    t_a_s, t_b_s = (t_a - T0) / 1000, (t_b - T0) / 1000
    gaps = [(t_a_s - 200, t_a_s + 280), (t_b_s - 380, t_b_s + 460)]
    return d, gaps, riders


def test_spec_example_31km_trip_with_22_minute_gap() -> None:
    d, gaps, riders = logout_trip_with_gps_loss()
    assert sum(b - a for a, b in gaps) == pytest.approx(22 * 60)
    pings = d.pings(gaps)
    ev = evidence(pings, end_ms=d.end_ms, riders=riders, start_location=GeoPoint(LAT0, LON0))
    engine, provider = make_engine()

    result = engine.compute(ev)

    truth = d.driven_m
    naive = naive_distance_m(pings)
    assert truth == pytest.approx(31_000)
    assert naive < 0.8 * truth  # straight lines across the gaps skip both side streets
    assert result.total_m == pytest.approx(truth, rel=0.01)
    routed = [li for li in result.line_items if li.rule is RuleCode.ROUTED]
    assert len(routed) == 4  # into and out of each of the two anchored drops
    assert all("DROP:E-" in li.note for li in routed)
    assert result.status is TripStatus.COMPLETE
    assert provider.calls == 1  # all gaps of the trip go to the provider in one batch


def test_spec_example_bad_fix_burst_leaves_distance_unchanged() -> None:
    d = Drive(GeoPoint(LAT0, LON0), speed_ms=10, every_s=4).leg(5000).leg(3000, "north")
    clean = d.pings()
    far = east(GeoPoint(LAT0, LON0), 9000)
    burst = PingBatch.from_arrays(
        clean.ts_ms[50] + np.arange(4) * 1000 + 500, np.full(4, far.lat), np.full(4, far.lon), np.full(4, 5.0)
    )
    engine, _ = make_engine()

    base = engine.compute(evidence(clean, end_ms=d.end_ms))
    noisy = engine.compute(evidence(clean.concat(burst), end_ms=d.end_ms))

    assert noisy.total_m == base.total_m
    assert noisy.input_fingerprint == base.input_fingerprint
    assert noisy.rejected_pings["TELEPORT"] == 4


def test_total_is_exact_sum_of_line_items_and_rounding_is_documented() -> None:
    trip = TripGenerator(config=GenConfig(target_pings=400)).generate(3)
    engine, _ = make_engine()
    r = engine.compute(trip.evidence)
    assert r.total_m == sum(li.distance_m for li in r.line_items)
    assert r.total_m == sum(r.rule_totals_m.values())
    assert all(li.distance_m >= 0 for li in r.line_items)
    assert metres_to_invoice_km(30_605) == metres_to_invoice_km(30_605)
    assert str(metres_to_invoice_km(30_605)) == "30.61"  # half-up at 2 decimals
    assert str(metres_to_invoice_km(30_604)) == "30.60"


def test_provider_down_marks_trip_degraded_and_uses_fallback() -> None:
    d, gaps, riders = logout_trip_with_gps_loss()
    ev = evidence(d.pings(gaps), end_ms=d.end_ms, riders=riders)
    engine, provider = make_engine()
    provider.set_down(True)

    r = engine.compute(ev)

    assert r.status is TripStatus.DEGRADED
    degraded = [li for li in r.line_items if li.rule is RuleCode.DEGRADED_AERIAL]
    assert len(degraded) == 4
    assert all("provider unavailable" in li.note for li in degraded)


def test_pinned_routes_reproduce_the_invoice_without_calling_the_provider() -> None:
    d, gaps, riders = logout_trip_with_gps_loss()
    ev = evidence(d.pings(gaps), end_ms=d.end_ms, riders=riders)
    engine, provider = make_engine()
    first = engine.compute(ev)
    calls = provider.calls

    again = engine.compute(ev, pinned_routes=first.route_evidence)

    assert provider.calls == calls
    assert again.total_m == first.total_m
    assert again.pinned_hits == first.routed_pairs


def test_shared_cache_does_not_make_results_depend_on_trip_order() -> None:
    gen = TripGenerator(config=GenConfig(target_pings=300))
    trips = [gen.generate(i).evidence for i in range(6)]
    e1, _ = make_engine(cache=RouteCache())
    e2, _ = make_engine(cache=RouteCache())
    forward = {t.trip_id: e1.compute(t).total_m for t in trips}
    backward = {t.trip_id: e2.compute(t).total_m for t in reversed(trips)}
    assert forward == backward


def test_cache_cuts_provider_calls_for_repeat_routes() -> None:
    gen = TripGenerator(config=GenConfig(target_pings=300))
    trip = gen.generate(1).evidence
    engine, provider = make_engine(cache=RouteCache())
    engine.compute(trip)
    first_calls = provider.calls
    r = engine.compute(trip)
    assert provider.calls == first_calls
    assert r.cache_hits == r.routed_pairs > 0


def test_rule_version_is_part_of_the_fingerprint() -> None:
    trip = TripGenerator(config=GenConfig(target_pings=200)).generate(5).evidence
    a, _ = make_engine(RuleConfig())
    b, _ = make_engine(RuleConfig(short_gap_s=60))
    assert a.compute(trip).input_fingerprint != b.compute(trip).input_fingerprint


def test_empty_trip_needs_review_but_does_not_crash() -> None:
    engine, _ = make_engine()
    r = engine.compute(evidence(PingBatch.empty(), end_ms=T0 + 60_000, start_location=None))
    assert r.total_m == 0
    assert r.review_required


def test_provider_fails_halfway_through_a_batch_then_recovers() -> None:
    """Failure injection: the provider dies after half the trips. Those trips are DEGRADED,
    never wrong silently, and recomputing after recovery gives exactly the healthy totals."""
    gen = TripGenerator(config=GenConfig(target_pings=300))
    trips = [gen.generate(i).evidence for i in range(30)]
    healthy, _ = make_engine()
    expected = [healthy.compute(t).total_m for t in trips]

    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_s=30, clock=lambda: now[0])
    engine, provider = make_engine(breaker=breaker)
    results = []
    for i, t in enumerate(trips):
        if i == len(trips) // 2:
            provider.set_down(True)
        results.append(engine.compute(t))

    assert all(r.status is TripStatus.COMPLETE for r in results[:15])
    degraded = [i for i, r in enumerate(results) if r.status is TripStatus.DEGRADED]
    assert degraded and min(degraded) >= 15
    calls_while_down = provider.calls
    assert calls_while_down < 15 + 3 + 1  # breaker opened after 3 failures and stopped the calls

    provider.set_down(False)
    now[0] += 31  # breaker half-opens and lets a trial call through
    for i in degraded:
        results[i] = engine.compute(trips[i])

    assert [r.total_m for r in results] == expected
    assert all(r.status is TripStatus.COMPLETE for r in results)
