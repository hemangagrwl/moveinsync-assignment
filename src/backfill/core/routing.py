from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

CellPair = tuple[int, int, int, int]  # (from_row, from_col, to_row, to_col)


class ProviderError(Exception):
    """The routing provider failed, timed out, or refused the request."""


@dataclass(frozen=True, slots=True)
class RouteQuery:
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float


class RoutingProvider(Protocol):
    """Anything that returns driving distances in metres, one per query, in order."""

    def route_many(self, queries: Sequence[RouteQuery]) -> list[float]: ...


class GridSnapper:
    """Snaps coordinates to a fixed lat/lon grid.

    The provider is always asked about cell *centres*, never raw points. That makes a cached
    answer independent of which trip happened to ask first, which is what keeps billing
    deterministic while still sharing the cache across trips.
    """

    def __init__(self, cell_size_deg: float) -> None:
        self.cell = cell_size_deg

    def cells(
        self, lat: NDArray[np.float64], lon: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return (
            np.floor(lat / self.cell).astype(np.int64),
            np.floor(lon / self.cell).astype(np.int64),
        )

    def centre(self, row: int, col: int) -> tuple[float, float]:
        return ((row + 0.5) * self.cell, (col + 0.5) * self.cell)

    def query(self, pair: CellPair) -> RouteQuery:
        a = self.centre(pair[0], pair[1])
        b = self.centre(pair[2], pair[3])
        return RouteQuery(a[0], a[1], b[0], b[1])


def pair_key(pair: CellPair) -> str:
    return f"{pair[0]}:{pair[1]}>{pair[2]}:{pair[3]}"


def parse_pair_key(key: str) -> CellPair:
    a, b = key.split(">")
    r1, c1 = a.split(":")
    r2, c2 = b.split(":")
    return (int(r1), int(c1), int(r2), int(c2))


def is_valid_distance(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


@dataclass(slots=True)
class RouteResolution:
    """Driving distance between cell centres, or None where the provider could not answer."""

    distances: dict[CellPair, float | None] = field(default_factory=dict)
    sources: dict[CellPair, str] = field(default_factory=dict)  # pinned | cache | provider | failed
    provider_calls: int = 0

    def count(self, source: str) -> int:
        return sum(1 for s in self.sources.values() if s == source)


class RouteResolver(Protocol):
    def resolve(
        self, pairs: Sequence[CellPair], pinned: Mapping[str, float] | None = None
    ) -> RouteResolution: ...
