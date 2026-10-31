from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from backfill.core.routing import CellPair


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def size_hint(self) -> int:
        return self.hits + self.misses


class RouteCache:
    """Thread-safe LRU of centre-to-centre driving distances, with a TTL.

    Keys are directed cell pairs (one-way streets make A->B differ from B->A). Values can only
    go stale when the road network changes, so staleness is bounded by ``ttl_s``. Failures are
    never cached.
    """

    def __init__(
        self,
        max_entries: int = 200_000,
        ttl_s: float = 24 * 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._clock = clock
        self._data: OrderedDict[CellPair, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: CellPair) -> float | None:
        now = self._clock()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            value, stored_at = entry
            if now - stored_at > self._ttl:
                del self._data[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def put(self, key: CellPair, value: float) -> None:
        now = self._clock()
        with self._lock:
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
