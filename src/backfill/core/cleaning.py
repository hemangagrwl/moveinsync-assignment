from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from backfill.core.config import RuleConfig
from backfill.core.geo import (
    METRES_PER_DEGREE,
    haversine_m,
    haversine_np,
    speed_kmh,
    speed_kmh_np,
    valid_coordinates_np,
)
from backfill.core.models import PingBatch


class RejectReason(StrEnum):
    INVALID_COORDINATE = "INVALID_COORDINATE"
    OUT_OF_WINDOW = "OUT_OF_WINDOW"
    LOW_ACCURACY = "LOW_ACCURACY"
    DUPLICATE = "DUPLICATE"
    TELEPORT = "TELEPORT"
    STATIONARY_JITTER = "STATIONARY_JITTER"


@dataclass(frozen=True, slots=True)
class CleanTrack:
    """Pings that survived cleaning, sorted by time with strictly increasing timestamps."""

    ts_ms: NDArray[np.int64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    rejected: dict[RejectReason, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ts_ms)


class PingCleaner:
    """Turns a raw, unordered, duplicated ping stream into a trustworthy track.

    Steps, in a fixed order: invalid coordinates -> outside the trip window -> low accuracy
    -> sort and de-duplicate -> teleport filter -> stationary jitter filter. Each rejection is
    counted by reason so the result can say how many pings were thrown away and why.
    """

    def __init__(self, config: RuleConfig) -> None:
        self._cfg = config

    def clean(self, pings: PingBatch, window_ms: tuple[int, int] | None = None) -> CleanTrack:
        rejected = dict.fromkeys(RejectReason, 0)
        batch = pings

        keep = valid_coordinates_np(batch.lat, batch.lon)
        rejected[RejectReason.INVALID_COORDINATE] = int((~keep).sum())
        batch = batch.take(np.flatnonzero(keep))

        if window_ms is not None:
            start, end = window_ms
            keep = (batch.ts_ms >= start) & (batch.ts_ms <= end)
            rejected[RejectReason.OUT_OF_WINDOW] = int((~keep).sum())
            batch = batch.take(np.flatnonzero(keep))

        # A missing accuracy (NaN) is accepted: many devices never report it.
        keep = ~(batch.accuracy_m > self._cfg.max_accuracy_m)
        rejected[RejectReason.LOW_ACCURACY] = int((~keep).sum())
        batch = batch.take(np.flatnonzero(keep))

        batch, duplicates = self._sort_and_dedupe(batch)
        rejected[RejectReason.DUPLICATE] = duplicates

        teleport_keep = self._teleport_mask(batch.ts_ms, batch.lat, batch.lon)
        rejected[RejectReason.TELEPORT] = int((~teleport_keep).sum())
        idx = np.flatnonzero(teleport_keep)
        ts, lat, lon = batch.ts_ms[idx], batch.lat[idx], batch.lon[idx]

        jitter_keep = self._jitter_mask(lat, lon)
        rejected[RejectReason.STATIONARY_JITTER] = int((~jitter_keep).sum())
        idx = np.flatnonzero(jitter_keep)

        return CleanTrack(ts[idx], lat[idx], lon[idx], rejected)

    @staticmethod
    def _sort_and_dedupe(batch: PingBatch) -> tuple[PingBatch, int]:
        """Order by time; for pings sharing a timestamp keep the most accurate one.

        Ties are broken by coordinates so the survivor never depends on arrival order.
        """
        if len(batch) == 0:
            return batch, 0
        acc = np.where(np.isnan(batch.accuracy_m), np.inf, batch.accuracy_m)
        order = np.lexsort((batch.lon, batch.lat, acc, batch.ts_ms))
        ordered = batch.take(order)
        first = np.ones(len(ordered), dtype=bool)
        first[1:] = ordered.ts_ms[1:] != ordered.ts_ms[:-1]
        return ordered.take(np.flatnonzero(first)), int((~first).sum())

    def _teleport_mask(
        self, ts: NDArray[np.int64], lat: NDArray[np.float64], lon: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Drop pings whose implied speed from the last accepted ping exceeds the limit.

        Fast path: if no consecutive pair is too fast, nothing is dropped and no Python loop
        runs. Otherwise a forward pass compares each ping with the last *accepted* one.

        If ``teleport_reset_count`` rejected pings in a row agree with each other, the track
        really did move (or the earlier fix was the bad one), so the run is accepted. Without
        this, one bad first fix would discard the whole trip.
        """
        n = len(ts)
        keep = np.ones(n, dtype=bool)
        if n < 2:
            return keep
        cfg = self._cfg
        dt_s = np.diff(ts) / 1000.0
        step_speed = speed_kmh_np(
            haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:]), dt_s, cfg.min_speed_dt_s
        )
        if not (step_speed > cfg.max_speed_kmh).any():
            return keep

        last = 0
        pending: list[int] = []
        for i in range(1, n):
            d = haversine_m(lat[last], lon[last], lat[i], lon[i])
            v = speed_kmh(d, (ts[i] - ts[last]) / 1000.0, cfg.min_speed_dt_s)
            if v <= cfg.max_speed_kmh:
                last = i
                pending.clear()
                continue
            keep[i] = False
            if pending and step_speed[i - 1] > cfg.max_speed_kmh:
                pending.clear()  # the rejected run is not self-consistent; start a new one
            pending.append(i)
            if len(pending) >= cfg.teleport_reset_count:
                keep[pending] = True
                last = pending[-1]
                pending.clear()
        return keep

    def _jitter_mask(self, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Drop pings that moved less than ``min_move_m`` from the last kept ping.

        While a cab waits at a pickup the fix wanders by a few metres every second, and summing
        that wander bills distance nobody drove. This is the standard distance filter GPS
        trackers apply on the device. The first and last pings are always kept. The decision
        uses a local flat projection, which is exact enough for an 8 m threshold.
        """
        n = len(lat)
        keep = np.ones(n, dtype=bool)
        r = self._cfg.min_move_m
        if r <= 0 or n < 3:
            return keep
        ky = METRES_PER_DEGREE
        kx = METRES_PER_DEGREE * math.cos(math.radians(float(lat[0])))
        ys = ((lat - lat[0]) * ky).tolist()
        xs = ((lon - lon[0]) * kx).tolist()
        r2 = r * r
        lx, ly = xs[0], ys[0]
        for i in range(1, n - 1):
            dx = xs[i] - lx
            dy = ys[i] - ly
            if dx * dx + dy * dy < r2:
                keep[i] = False
            else:
                lx, ly = xs[i], ys[i]
        return keep
