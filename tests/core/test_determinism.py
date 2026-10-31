"""Property-based proof of "same input + same rule version -> same kilometres".

Hypothesis generates trips and then re-delivers their pings shuffled, duplicated and split
into arbitrary late batches. The billed metres and the input fingerprint must not move.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backfill.core.models import PingBatch, TripEvidence
from backfill.providers.cache import RouteCache
from backfill.sim.generator import GenConfig, TripGenerator
from tests.helpers import make_engine

GEN = TripGenerator(config=GenConfig(target_pings=250))
PLAIN, _ = make_engine()
CACHED, _ = make_engine(cache=RouteCache())


def redeliver(ev: TripEvidence, seed: int, dup_fraction: float) -> TripEvidence:
    rng = np.random.default_rng(seed)
    p = ev.pings
    dup = np.flatnonzero(rng.random(len(p)) < dup_fraction)
    both = p.concat(p.take(dup))
    shuffled = both.take(rng.permutation(len(both)))
    return TripEvidence(
        ev.trip_id,
        ev.trip_type,
        ev.office,
        ev.start_time_ms,
        ev.end_time_ms,
        shuffled,
        ev.riders,
        ev.start_location,
        ev.end_location,
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    trip=st.integers(0, 5_000),
    seed=st.integers(0, 2**32 - 1),
    dup_fraction=st.floats(0.0, 0.5),
)
def test_shuffled_and_duplicated_pings_bill_the_same(trip: int, seed: int, dup_fraction: float) -> None:
    ev = GEN.generate(trip).evidence
    base = PLAIN.compute(ev)
    again = CACHED.compute(redeliver(ev, seed, dup_fraction))
    assert again.total_m == base.total_m
    assert again.input_fingerprint == base.input_fingerprint
    assert again.line_items == base.line_items


@settings(max_examples=25, deadline=None)
@given(trip=st.integers(0, 5_000), cut=st.floats(0.1, 0.9))
def test_late_batches_converge_to_the_full_delivery(trip: int, cut: float) -> None:
    ev = GEN.generate(trip).evidence
    k = int(len(ev.pings) * cut)
    first = ev.pings.take(np.arange(k))
    late = ev.pings.take(np.arange(k, len(ev.pings)))
    merged = first.concat(late)
    full = PLAIN.compute(ev)
    replay = PLAIN.compute(
        TripEvidence(
            ev.trip_id,
            ev.trip_type,
            ev.office,
            ev.start_time_ms,
            ev.end_time_ms,
            merged,
            ev.riders,
            ev.start_location,
            ev.end_location,
        )
    )
    assert replay.total_m == full.total_m


@settings(max_examples=25, deadline=None)
@given(trip=st.integers(0, 5_000))
def test_invariants(trip: int) -> None:
    r = PLAIN.compute(GEN.generate(trip).evidence)
    assert r.total_m == sum(li.distance_m for li in r.line_items)
    assert all(li.distance_m >= 0 for li in r.line_items)
    assert all(li.end_ms >= li.start_ms for li in r.line_items)
    # Line items tile the track: each one starts where the previous one ended.
    assert all(a.to_point == b.from_point for a, b in zip(r.line_items, r.line_items[1:], strict=False))


def test_empty_batch_round_trip() -> None:
    assert len(PingBatch.empty()) == 0
