from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from backfill.core.routing import (
    CellPair,
    GridSnapper,
    ProviderError,
    RouteResolution,
    RoutingProvider,
    is_valid_distance,
    pair_key,
)
from backfill.providers.breaker import CircuitBreaker
from backfill.providers.cache import RouteCache


class RouterObserver(Protocol):
    def provider_call(self, pairs: int, ok: bool, seconds: float) -> None: ...
    def lookup(self, source: str, count: int) -> None: ...


class _NullObserver:
    def provider_call(self, pairs: int, ok: bool, seconds: float) -> None:
        return None

    def lookup(self, source: str, count: int) -> None:
        return None


class ResilientRouter:
    """Everything between the engine and the paid routing provider.

    Lookup order per unique cell pair: pinned (answers already used on this trip's previous
    invoice) -> shared cache -> provider, in batches of ``batch_size``. When the breaker is
    open or a batch fails, those pairs come back as None and the engine marks the trip
    DEGRADED; nothing is retried inline, the recompute worker owns retries.
    """

    def __init__(
        self,
        provider: RoutingProvider,
        snapper: GridSnapper,
        breaker: CircuitBreaker,
        cache: RouteCache | None = None,
        batch_size: int = 50,
        observer: RouterObserver | None = None,
    ) -> None:
        self._provider = provider
        self._snapper = snapper
        self._breaker = breaker
        self._cache = cache
        self._batch = batch_size
        self._obs: RouterObserver = observer or _NullObserver()

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def cache(self) -> RouteCache | None:
        return self._cache

    def resolve(
        self, pairs: Sequence[CellPair], pinned: Mapping[str, float] | None = None
    ) -> RouteResolution:
        out = RouteResolution()
        missing: list[CellPair] = []
        for p in sorted(set(pairs)):
            key = pair_key(p)
            if pinned is not None and key in pinned:
                out.distances[p], out.sources[p] = pinned[key], "pinned"
                continue
            cached = self._cache.get(p) if self._cache is not None else None
            if cached is not None:
                out.distances[p], out.sources[p] = cached, "cache"
                continue
            missing.append(p)

        self._obs.lookup("pinned", out.count("pinned"))
        self._obs.lookup("cache", out.count("cache"))
        self._obs.lookup("miss", len(missing))

        for start in range(0, len(missing), self._batch):
            chunk = missing[start : start + self._batch]
            if not self._breaker.allow_request():
                self._mark_failed(out, chunk, "breaker_open")
                continue
            out.provider_calls += 1
            began = time.perf_counter()
            try:
                answers = self._provider.route_many([self._snapper.query(p) for p in chunk])
                if len(answers) != len(chunk) or not all(is_valid_distance(a) for a in answers):
                    raise ProviderError("malformed provider response")
            except ProviderError:
                self._breaker.record_failure()
                self._obs.provider_call(len(chunk), False, time.perf_counter() - began)
                self._mark_failed(out, chunk, "failed")
                continue
            self._breaker.record_success()
            self._obs.provider_call(len(chunk), True, time.perf_counter() - began)
            for p, d in zip(chunk, answers, strict=True):
                out.distances[p], out.sources[p] = float(d), "provider"
                if self._cache is not None:
                    self._cache.put(p, float(d))
        return out

    @staticmethod
    def _mark_failed(out: RouteResolution, chunk: list[CellPair], source: str) -> None:
        for p in chunk:
            out.distances[p], out.sources[p] = None, source
