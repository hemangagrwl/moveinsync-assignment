from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backfill.core.config import RuleConfig
from backfill.core.engine import DistanceEngine
from backfill.core.geo import METRES_PER_DEGREE
from backfill.core.models import GeoPoint, PingBatch, Rider, TripEvidence, TripType
from backfill.core.routing import GridSnapper
from backfill.core.rules import RuleSet
from backfill.providers.breaker import CircuitBreaker
from backfill.providers.cache import RouteCache
from backfill.providers.mock import MockRoutingProvider
from backfill.providers.resilient import ResilientRouter

T0 = 1_767_250_000_000  # fixed epoch ms for readable tests
LAT0, LON0 = 12.9700, 77.6000


def east(p: GeoPoint, metres: float) -> GeoPoint:
    return GeoPoint(p.lat, p.lon + metres / (METRES_PER_DEGREE * math.cos(math.radians(p.lat))))


def north(p: GeoPoint, metres: float) -> GeoPoint:
    return GeoPoint(p.lat + metres / METRES_PER_DEGREE, p.lon)


@dataclass
class Drive:
    """Scripted cab: drive straight legs at a constant speed, pinging every ``every_s`` seconds."""

    start: GeoPoint
    speed_ms: float = 10.0
    every_s: float = 4.0
    t_s: float = 0.0
    pos: GeoPoint = field(init=False)
    ts: list[float] = field(default_factory=list)
    lat: list[float] = field(default_factory=list)
    lon: list[float] = field(default_factory=list)
    driven_m: float = 0.0

    def __post_init__(self) -> None:
        self.pos = self.start
        self._ping()

    def _ping(self) -> None:
        self.ts.append(self.t_s)
        self.lat.append(self.pos.lat)
        self.lon.append(self.pos.lon)

    def leg(self, metres: float, axis: str = "east") -> Drive:
        step = self.speed_ms * self.every_s
        done = 0.0
        sign = 1.0 if metres >= 0 else -1.0
        total = abs(metres)
        while done < total:
            d = min(step, total - done)
            self.pos = east(self.pos, sign * d) if axis == "east" else north(self.pos, sign * d)
            self.t_s += d / self.speed_ms
            done += d
            self._ping()
        self.driven_m += total
        return self

    def wait(self, seconds: float) -> Drive:
        self.t_s += seconds
        self._ping()
        return self

    def at(self) -> tuple[GeoPoint, int]:
        return self.pos, T0 + round(self.t_s * 1000)

    def pings(self, drop_between_s: list[tuple[float, float]] | None = None) -> PingBatch:
        ts = np.array(self.ts)
        keep = np.ones(len(ts), dtype=bool)
        for a, b in drop_between_s or []:
            keep &= ~((ts > a) & (ts < b))
        return PingBatch.from_arrays(
            T0 + np.round(ts[keep] * 1000).astype(np.int64),
            np.array(self.lat)[keep],
            np.array(self.lon)[keep],
            np.full(int(keep.sum()), 5.0),
        )

    @property
    def end_ms(self) -> int:
        return T0 + round(self.t_s * 1000)


def evidence(
    pings: PingBatch,
    *,
    end_ms: int,
    office: GeoPoint | None = None,
    riders: tuple[Rider, ...] = (),
    trip_type: TripType = TripType.LOGOUT,
    start_location: GeoPoint | None = None,
    end_location: GeoPoint | None = None,
) -> TripEvidence:
    return TripEvidence(
        trip_id="T-test",
        trip_type=trip_type,
        office=office or GeoPoint(LAT0, LON0),
        start_time_ms=T0,
        end_time_ms=end_ms,
        pings=pings,
        riders=riders,
        start_location=start_location,
        end_location=end_location,
    )


def make_engine(
    config: RuleConfig | None = None,
    provider: MockRoutingProvider | None = None,
    cache: RouteCache | None = None,
    breaker: CircuitBreaker | None = None,
) -> tuple[DistanceEngine, MockRoutingProvider]:
    cfg = config or RuleConfig()
    prov = provider or MockRoutingProvider()
    router = ResilientRouter(prov, GridSnapper(cfg.cell_size_deg), breaker or CircuitBreaker(), cache)
    return DistanceEngine(RuleSet.default(cfg), router), prov
