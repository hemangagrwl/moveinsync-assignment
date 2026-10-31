from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from backfill.core.routing import GridSnapper, ProviderError, RouteQuery, pair_key, parse_pair_key
from backfill.providers.breaker import BreakerState, CircuitBreaker
from backfill.providers.cache import RouteCache
from backfill.providers.mock import ManhattanRoadNetwork, MockRoutingProvider
from backfill.providers.resilient import ResilientRouter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# -- circuit breaker ---------------------------------------------------------------------------


def test_breaker_opens_after_threshold_and_half_opens_after_timeout() -> None:
    clock = Clock()
    b = CircuitBreaker(failure_threshold=3, reset_timeout_s=10, clock=clock)
    for _ in range(2):
        b.record_failure()
    assert b.state is BreakerState.CLOSED
    b.record_failure()
    assert b.state is BreakerState.OPEN
    assert not b.allow_request()

    clock.now = 10
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow_request()  # exactly one trial
    assert not b.allow_request()
    b.record_success()
    assert b.state is BreakerState.CLOSED


def test_failed_trial_reopens_the_breaker() -> None:
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, reset_timeout_s=5, clock=clock)
    b.record_failure()
    clock.now = 5
    assert b.allow_request()
    b.record_failure()
    assert b.state is BreakerState.OPEN
    clock.now = 9
    assert not b.allow_request()


def test_success_resets_the_failure_count() -> None:
    b = CircuitBreaker(failure_threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.state is BreakerState.CLOSED


def test_listeners_see_every_transition() -> None:
    clock = Clock()
    seen: list[tuple[BreakerState, BreakerState]] = []
    b = CircuitBreaker(failure_threshold=1, reset_timeout_s=1, clock=clock)
    b.add_listener(lambda old, new: seen.append((old, new)))
    b.record_failure()
    clock.now = 1
    b.allow_request()
    b.record_success()
    assert seen == [
        (BreakerState.CLOSED, BreakerState.OPEN),
        (BreakerState.OPEN, BreakerState.HALF_OPEN),
        (BreakerState.HALF_OPEN, BreakerState.CLOSED),
    ]


def test_breaker_lets_exactly_one_trial_through_under_concurrency() -> None:
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, reset_timeout_s=1, clock=clock)
    b.record_failure()
    clock.now = 2
    allowed: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def worker() -> None:
        barrier.wait()
        ok = b.allow_request()
        with lock:
            allowed.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert allowed.count(True) == 1


# -- cache -------------------------------------------------------------------------------------


def test_cache_evicts_least_recently_used() -> None:
    c = RouteCache(max_entries=2)
    c.put((0, 0, 0, 1), 1.0)
    c.put((0, 0, 0, 2), 2.0)
    assert c.get((0, 0, 0, 1)) == 1.0  # touch -> most recent
    c.put((0, 0, 0, 3), 3.0)
    assert c.get((0, 0, 0, 2)) is None
    assert c.get((0, 0, 0, 1)) == 1.0
    assert c.stats.evictions == 1


def test_cache_entries_expire_after_ttl() -> None:
    clock = Clock()
    c = RouteCache(ttl_s=60, clock=clock)
    c.put((1, 1, 2, 2), 500.0)
    clock.now = 60
    assert c.get((1, 1, 2, 2)) == 500.0
    clock.now = 61
    assert c.get((1, 1, 2, 2)) is None
    assert c.stats.expirations == 1


def test_pair_key_round_trip() -> None:
    p = (-3, 155_200, 25_940, -7)
    assert parse_pair_key(pair_key(p)) == p


# -- router ------------------------------------------------------------------------------------


def pairs(n: int) -> list[tuple[int, int, int, int]]:
    return [(25_940, 155_200, 25_940 + i + 1, 155_200) for i in range(n)]


def test_router_batches_and_counts_calls() -> None:
    prov = MockRoutingProvider()
    r = ResilientRouter(prov, GridSnapper(0.0005), CircuitBreaker(), RouteCache(), batch_size=50)
    res = r.resolve(pairs(120))
    assert res.provider_calls == 3
    assert prov.pairs == 120
    again = r.resolve(pairs(120))
    assert again.provider_calls == 0
    assert again.count("cache") == 120


def test_router_deduplicates_pairs_within_a_trip() -> None:
    prov = MockRoutingProvider()
    r = ResilientRouter(prov, GridSnapper(0.0005), CircuitBreaker())
    res = r.resolve(pairs(5) * 4)
    assert prov.pairs == 5
    assert len(res.distances) == 5


def test_router_returns_none_instead_of_raising_when_provider_fails() -> None:
    prov = MockRoutingProvider()
    prov.set_down(True)
    breaker = CircuitBreaker(failure_threshold=2)
    r = ResilientRouter(prov, GridSnapper(0.0005), breaker, batch_size=10)
    res = r.resolve(pairs(50))
    assert all(v is None for v in res.distances.values())
    assert res.provider_calls == 2  # third batch onwards short-circuited by the open breaker
    assert breaker.state is BreakerState.OPEN
    assert res.count("breaker_open") == 30


class Malformed:
    def route_many(self, queries: Sequence[RouteQuery]) -> list[float]:
        return [float("nan")] * len(queries)


def test_malformed_answers_are_treated_as_failures_and_never_cached() -> None:
    cache = RouteCache()
    r = ResilientRouter(Malformed(), GridSnapper(0.0005), CircuitBreaker(), cache)
    res = r.resolve(pairs(3))
    assert all(v is None for v in res.distances.values())
    assert len(cache) == 0


def test_mock_provider_is_manhattan_distance() -> None:
    net = ManhattanRoadNetwork()
    q = RouteQuery(12.97, 77.60, 12.98, 77.61)
    d = net.distance_m(q)
    assert d == pytest.approx(1111.95 + 1083.3, rel=1e-3)


def test_mock_provider_fails_after_configured_calls() -> None:
    prov = MockRoutingProvider(fail_after_calls=1)
    q = [RouteQuery(12.97, 77.60, 12.98, 77.61)]
    prov.route_many(q)
    with pytest.raises(ProviderError):
        prov.route_many(q)
