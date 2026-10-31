from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

from backfill.core.geo import METRES_PER_DEGREE
from backfill.core.routing import ProviderError, RouteQuery


class ManhattanRoadNetwork:
    """A city of straight north-south and east-west streets.

    The driving distance between two points is their L1 distance in local metres. The
    synthetic trip generator drives cabs on the same grid, so for every gap we know the true
    driving distance exactly, which is what lets the benchmark measure accuracy.
    """

    def distance_m(self, q: RouteQuery) -> float:
        mean_lat = math.radians((q.from_lat + q.to_lat) / 2.0)
        dy = abs(q.to_lat - q.from_lat) * METRES_PER_DEGREE
        dx = abs(q.to_lon - q.from_lon) * METRES_PER_DEGREE * math.cos(mean_lat)
        return dx + dy


class MockRoutingProvider:
    """In-process provider with the failure modes of a real one: latency, outages, call cost."""

    def __init__(
        self,
        network: ManhattanRoadNetwork | None = None,
        latency_s: float = 0.0,
        fail_after_calls: int | None = None,
    ) -> None:
        self._network = network or ManhattanRoadNetwork()
        self._latency = latency_s
        self._fail_after = fail_after_calls
        self._down = False
        self._lock = threading.Lock()
        self.calls = 0
        self.pairs = 0

    def set_down(self, down: bool) -> None:
        with self._lock:
            self._down = down
            if not down:
                self._fail_after = None

    @property
    def is_down(self) -> bool:
        return self._down

    def route_many(self, queries: Sequence[RouteQuery]) -> list[float]:
        with self._lock:
            self.calls += 1
            if self._down or (self._fail_after is not None and self.calls > self._fail_after):
                raise ProviderError("routing provider unavailable")
            self.pairs += len(queries)
        if self._latency:
            time.sleep(self._latency)
        return [self._network.distance_m(q) for q in queries]
