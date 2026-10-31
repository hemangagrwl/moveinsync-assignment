from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

E7 = 10_000_000


class TripType(StrEnum):
    LOGIN = "LOGIN"  # pick employees up, end at the office
    LOGOUT = "LOGOUT"  # start at the office, drop employees home


class RiderType(StrEnum):
    PLANNED = "PLANNED"
    ADHOC = "ADHOC"


class RiderStatus(StrEnum):
    TRAVELLED = "TRAVELLED"
    NO_SHOW = "NO_SHOW"


@dataclass(frozen=True, slots=True)
class GeoPoint:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class Rider:
    """One employee on the trip. ``actual_*`` is the pickup (LOGIN) or drop (LOGOUT) event."""

    employee_ref: str
    rider_type: RiderType = RiderType.PLANNED
    status: RiderStatus = RiderStatus.TRAVELLED
    planned_location: GeoPoint | None = None
    planned_time_ms: int | None = None
    actual_location: GeoPoint | None = None
    actual_time_ms: int | None = None
    profile_home: GeoPoint | None = None
    sign_in_time_ms: int | None = None
    no_show_location: GeoPoint | None = None
    no_show_time_ms: int | None = None


def quantize_e7(values: ArrayLike) -> NDArray[np.float64]:
    """Snap coordinates to 1e-7 degrees (~1 cm).

    The database stores coordinates as integers in 1e-7 degrees, so quantising on the way in
    makes an in-memory run and a run that round-trips through the database bit-identical.
    """
    return np.round(np.asarray(values, dtype=np.float64) * E7) / E7


@dataclass(frozen=True, slots=True)
class PingBatch:
    """Raw device pings in columnar form. ``accuracy_m`` is NaN when the device did not report it."""

    ts_ms: NDArray[np.int64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    accuracy_m: NDArray[np.float64]

    @classmethod
    def from_arrays(
        cls,
        ts_ms: ArrayLike,
        lat: ArrayLike,
        lon: ArrayLike,
        accuracy_m: ArrayLike | None = None,
    ) -> PingBatch:
        ts = np.asarray(ts_ms, dtype=np.int64).reshape(-1)
        n = len(ts)
        acc = (
            np.full(n, np.nan) if accuracy_m is None else np.asarray(accuracy_m, dtype=np.float64).reshape(-1)
        )
        lat_a = quantize_e7(lat).reshape(-1)
        lon_a = quantize_e7(lon).reshape(-1)
        if not (len(lat_a) == len(lon_a) == len(acc) == n):
            raise ValueError("ping columns must have equal length")
        return cls(ts, lat_a, lon_a, acc)

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[int, float, float, float | None]]) -> PingBatch:
        materialised = list(rows)
        if not materialised:
            return cls.empty()
        ts, lat, lon, acc = zip(*materialised, strict=True)
        return cls.from_arrays(ts, lat, lon, [np.nan if a is None else a for a in acc])

    @classmethod
    def empty(cls) -> PingBatch:
        return cls.from_arrays(np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0))

    def __len__(self) -> int:
        return len(self.ts_ms)

    def concat(self, other: PingBatch) -> PingBatch:
        return PingBatch(
            np.concatenate([self.ts_ms, other.ts_ms]),
            np.concatenate([self.lat, other.lat]),
            np.concatenate([self.lon, other.lon]),
            np.concatenate([self.accuracy_m, other.accuracy_m]),
        )

    def take(self, index: NDArray[np.intp]) -> PingBatch:
        return PingBatch(self.ts_ms[index], self.lat[index], self.lon[index], self.accuracy_m[index])


@dataclass(frozen=True, slots=True)
class TripEvidence:
    """Everything known about one trip at computation time."""

    trip_id: str
    trip_type: TripType
    office: GeoPoint
    start_time_ms: int
    end_time_ms: int
    pings: PingBatch
    riders: tuple[Rider, ...] = ()
    start_location: GeoPoint | None = None
    end_location: GeoPoint | None = None
