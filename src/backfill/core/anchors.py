from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from backfill.core.config import RuleConfig
from backfill.core.geo import haversine_m, haversine_np, is_valid_coordinate, speed_kmh_np
from backfill.core.models import GeoPoint, Rider, RiderStatus, RiderType, TripEvidence, TripType


class AnchorKind(StrEnum):
    TRIP_START = "TRIP_START"
    TRIP_END = "TRIP_END"
    PICKUP = "PICKUP"
    DROP = "DROP"
    NO_SHOW = "NO_SHOW"
    OFFICE = "OFFICE"


class AnchorSource(StrEnum):
    ACTUAL = "ACTUAL"
    PROFILE_HOME = "PROFILE_HOME"
    PLANNED = "PLANNED"
    NO_SHOW_REPORTED = "NO_SHOW_REPORTED"
    OFFICE_ACTUAL = "OFFICE_ACTUAL"
    OFFICE_PLANNED = "OFFICE_PLANNED"


class AnchorStatus(StrEnum):
    USED = "USED"
    RETIMED = "RETIMED"  # fallback anchor moved into the GPS gap it belongs to
    REDUNDANT = "REDUNDANT"  # GPS already covers this moment, the anchor adds nothing


# Sources whose timestamp is a real event, not a plan or a profile value.
_TRUSTED_TIME = {AnchorSource.ACTUAL, AnchorSource.NO_SHOW_REPORTED, AnchorSource.OFFICE_ACTUAL}


@dataclass(frozen=True, slots=True)
class Anchor:
    kind: AnchorKind
    source: AnchorSource
    lat: float
    lon: float
    ts_ms: int
    ref: str  # opaque employee reference or "office"; never a name or phone number

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.ref}:{self.source}"

    @property
    def time_is_trusted(self) -> bool:
        return self.source in _TRUSTED_TIME


@dataclass(frozen=True, slots=True)
class AnchorDecision:
    anchor: Anchor
    status: AnchorStatus
    original_ts_ms: int


@dataclass(frozen=True, slots=True)
class AnchorBuildResult:
    anchors: tuple[Anchor, ...]
    riders_without_anchor: tuple[str, ...]


class AnchorBuilder:
    """Builds the trusted points (employee events, office) that patch GPS gaps."""

    def __init__(self, config: RuleConfig) -> None:
        self._cfg = config

    def build(self, ev: TripEvidence) -> AnchorBuildResult:
        anchors: list[Anchor] = []
        missing: list[str] = []

        start_side = (
            self._office_anchor(ev)
            if ev.trip_type is TripType.LOGOUT
            else self._edge(ev.start_location, ev.start_time_ms, AnchorKind.TRIP_START)
        )
        end_side = (
            self._office_anchor(ev)
            if ev.trip_type is TripType.LOGIN
            else self._edge(ev.end_location, ev.end_time_ms, AnchorKind.TRIP_END)
        )
        for a in (start_side, end_side):
            if a is not None:
                anchors.append(a)

        for rider in sorted(ev.riders, key=lambda r: r.employee_ref):
            if rider.status is RiderStatus.TRAVELLED:
                anchor = self._rider_anchor(ev, rider)
            elif self._cfg.include_no_show_anchors:
                anchor = self._no_show_anchor(ev, rider)
            else:
                continue
            if anchor is None:
                missing.append(rider.employee_ref)
            else:
                anchors.append(anchor)

        return AnchorBuildResult(tuple(anchors), tuple(missing))

    # -- individual anchor kinds ---------------------------------------------------------

    def _rider_anchor(self, ev: TripEvidence, r: Rider) -> Anchor | None:
        kind = AnchorKind.PICKUP if ev.trip_type is TripType.LOGIN else AnchorKind.DROP
        expected = r.planned_location or r.profile_home
        if self._is_valid_event(ev, r.actual_location, r.actual_time_ms, near=expected):
            return self._anchor(kind, AnchorSource.ACTUAL, r.actual_location, r.actual_time_ms, r)
        if r.rider_type is RiderType.ADHOC:
            if self._is_valid_event(ev, r.profile_home, r.sign_in_time_ms):
                return self._anchor(kind, AnchorSource.PROFILE_HOME, r.profile_home, r.sign_in_time_ms, r)
        elif self._is_valid_event(ev, r.planned_location, r.planned_time_ms):
            return self._anchor(kind, AnchorSource.PLANNED, r.planned_location, r.planned_time_ms, r)
        return None

    def _no_show_anchor(self, ev: TripEvidence, r: Rider) -> Anchor | None:
        ts = r.no_show_time_ms if r.no_show_time_ms is not None else r.planned_time_ms
        if self._is_valid_event(ev, r.no_show_location, ts, near=r.planned_location):
            return self._anchor(AnchorKind.NO_SHOW, AnchorSource.NO_SHOW_REPORTED, r.no_show_location, ts, r)
        if self._is_valid_event(ev, r.planned_location, ts):
            return self._anchor(AnchorKind.NO_SHOW, AnchorSource.PLANNED, r.planned_location, ts, r)
        return None

    def _office_anchor(self, ev: TripEvidence) -> Anchor:
        """The office end of the trip: the actual point if within the radius, else the plan."""
        if ev.trip_type is TripType.LOGIN:
            actual, ts = ev.end_location, ev.end_time_ms
        else:
            actual, ts = ev.start_location, ev.start_time_ms
        if (
            actual is not None
            and is_valid_coordinate(actual.lat, actual.lon)
            and haversine_m(actual.lat, actual.lon, ev.office.lat, ev.office.lon) <= self._cfg.office_radius_m
        ):
            return Anchor(AnchorKind.OFFICE, AnchorSource.OFFICE_ACTUAL, actual.lat, actual.lon, ts, "office")
        return Anchor(
            AnchorKind.OFFICE, AnchorSource.OFFICE_PLANNED, ev.office.lat, ev.office.lon, ts, "office"
        )

    @staticmethod
    def _edge(point: GeoPoint | None, ts: int, kind: AnchorKind) -> Anchor | None:
        if point is None or not is_valid_coordinate(point.lat, point.lon):
            return None
        return Anchor(kind, AnchorSource.ACTUAL, point.lat, point.lon, ts, "trip")

    # -- helpers ------------------------------------------------------------------------

    def _is_valid_event(
        self, ev: TripEvidence, point: GeoPoint | None, ts: int | None, near: GeoPoint | None = None
    ) -> bool:
        if point is None or ts is None or not is_valid_coordinate(point.lat, point.lon):
            return False
        grace_ms = int(self._cfg.anchor_time_grace_s * 1000)
        if not (ev.start_time_ms - grace_ms <= ts <= ev.end_time_ms + grace_ms):
            return False
        if near is not None and is_valid_coordinate(near.lat, near.lon):
            deviation = haversine_m(point.lat, point.lon, near.lat, near.lon)
            if deviation > self._cfg.anchor_max_deviation_m:
                return False
        return True

    @staticmethod
    def _anchor(
        kind: AnchorKind, source: AnchorSource, point: GeoPoint | None, ts: int | None, r: Rider
    ) -> Anchor:
        assert point is not None and ts is not None
        return Anchor(kind, source, point.lat, point.lon, ts, r.employee_ref)


class AnchorPlacer:
    """Decides where each employee anchor goes in the cleaned GPS track.

    Edge anchors (office, trip start, trip end) are always used. For employee anchors:

    0. An anchor with a trusted time (a real pickup/drop event) inside a GPS gap is used.
    1. Otherwise, if the GPS track passes within ``anchor_visit_radius_m`` of the anchor around that time,
       the anchor is REDUNDANT: the track already proves the cab went there, and a planned
       time that is a few minutes off would only add a zig-zag.
    2. An anchor with a trusted time (a real pickup/drop event) is used at that time, unless
       GPS already covers that moment on both sides.
    3. A fallback anchor (planned or profile time) is RETIMED into a nearby GPS gap. Gaps the
       cab could not have driven through the anchor at a legal speed are skipped; of the rest,
       the one with the lowest cost wins, where cost is the detour the anchor adds plus how far
       the planned time falls outside the gap, priced at ``retime_metres_per_s``. Its location
       is trusted, its time is not, so its new time is interpolated by distance between the
       two ends of the gap: the cab keeps one speed across the gap.
    """

    def __init__(self, config: RuleConfig) -> None:
        self._cfg = config

    def place(
        self,
        anchors: tuple[Anchor, ...],
        ts: NDArray[np.int64],
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
    ) -> list[AnchorDecision]:
        """Edge anchors first, then employee anchors in time order. Each placed anchor is added
        to the reference track, so two drops in one gap split it instead of colliding."""
        ref = (ts, lat, lon)
        decided: dict[int, AnchorDecision] = {}
        for k, a in enumerate(anchors):
            if a.kind in _EDGE_KINDS or len(ts) == 0:
                decided[k] = AnchorDecision(a, AnchorStatus.USED, a.ts_ms)
                ref = _insert(ref, decided[k].anchor)
        rest = sorted(
            (k for k in range(len(anchors)) if k not in decided),
            key=lambda k: (anchors[k].ts_ms, anchors[k].label),
        )
        for k in rest:
            d = self._place_one(anchors[k], (ts, lat, lon), ref)
            decided[k] = d
            if d.status is not AnchorStatus.REDUNDANT:
                ref = _insert(ref, d.anchor)
        return [decided[k] for k in range(len(anchors))]

    def _place_one(self, a: Anchor, gps: _Series, ref: _Series) -> AnchorDecision:
        if a.time_is_trusted and self._inside_gap(a.ts_ms, ref):
            return AnchorDecision(a, AnchorStatus.USED, a.ts_ms)
        if self._is_visited(a, gps):
            return AnchorDecision(a, AnchorStatus.REDUNDANT, a.ts_ms)
        if not a.time_is_trusted:
            retimed = self._retime_into_gap(a, ref)
            if retimed is not None:
                return AnchorDecision(retimed, AnchorStatus.RETIMED, a.ts_ms)
        if self._is_covered(a.ts_ms, gps[0]):
            return AnchorDecision(a, AnchorStatus.REDUNDANT, a.ts_ms)
        return AnchorDecision(a, AnchorStatus.USED, a.ts_ms)

    def _inside_gap(self, t: int, ref: _Series) -> bool:
        ts = ref[0]
        i = int(np.searchsorted(ts, t, side="right"))
        return 0 < i < len(ts) and ts[i] - ts[i - 1] >= self._cfg.short_gap_s * 1000

    def _is_visited(self, a: Anchor, gps: _Series) -> bool:
        ts, lat, lon = gps
        window_ms = self._cfg.anchor_visit_window_s * 1000
        lo = int(np.searchsorted(ts, a.ts_ms - window_ms, side="left"))
        hi = int(np.searchsorted(ts, a.ts_ms + window_ms, side="right"))
        if hi <= lo:
            return False
        near = haversine_np(lat[lo:hi], lon[lo:hi], np.full(hi - lo, a.lat), np.full(hi - lo, a.lon))
        return bool((near <= self._cfg.anchor_visit_radius_m).any())

    def _is_covered(self, t: int, ts: NDArray[np.int64]) -> bool:
        cover_ms = self._cfg.anchor_coverage_s * 1000
        i = int(np.searchsorted(ts, t, side="right"))
        has_before = i > 0 and t - ts[i - 1] <= cover_ms
        has_after = i < len(ts) and ts[i] - t <= cover_ms
        return bool(has_before and has_after)

    def _retime_into_gap(self, a: Anchor, ref: _Series) -> Anchor | None:
        """Every interval between consecutive reference points near the planned time is a
        candidate, evaluated at once. In densely tracked stretches the anchor cannot be reached
        at a legal speed, so in practice only real gaps survive the plausibility filter."""
        ts, lat, lon = ref
        cfg = self._cfg
        if len(ts) < 2:
            return None
        search_ms = cfg.fallback_anchor_search_s * 1000
        lo = max(int(np.searchsorted(ts, a.ts_ms - search_ms, side="left")) - 1, 0)
        hi = min(int(np.searchsorted(ts, a.ts_ms + search_ms, side="right")) + 1, len(ts))
        idx = np.arange(lo, hi - 1)
        if len(idx) == 0:
            return None
        t0, t1 = ts[idx], ts[idx + 1]
        alat, alon = np.full(len(idx), a.lat), np.full(len(idx), a.lon)
        d_in = haversine_np(lat[idx], lon[idx], alat, alon)
        d_out = haversine_np(alat, alon, lat[idx + 1], lon[idx + 1])
        chord = haversine_np(lat[idx], lon[idx], lat[idx + 1], lon[idx + 1])
        moved = np.maximum(d_in + d_out - cfg.speed_tolerance_m, 0.0)
        # The cab must be able to reach the anchor and carry on within the interval.
        plausible = (t1 - t0 >= 2) & (
            speed_kmh_np(moved, (t1 - t0) / 1000.0, cfg.min_speed_dt_s) <= cfg.max_speed_kmh
        )
        outside_s = np.maximum(0, np.maximum(t0 - a.ts_ms, a.ts_ms - t1)) / 1000.0
        # A planned time that is off by a minute costs as much as ~500 m of detour.
        cost = np.where(plausible, d_in + d_out - chord + outside_s * cfg.retime_metres_per_s, np.inf)
        k = int(np.argmin(cost))
        if not np.isfinite(cost[k]):
            return None
        total = d_in[k] + d_out[k]
        share = float(d_in[k] / total) if total > 0 else 0.5
        start, end = int(t0[k]), int(t1[k])
        return replace(a, ts_ms=start + 1 + round((end - start - 2) * share))


def _insert(ref: _Series, a: Anchor) -> _Series:
    ts, lat, lon = ref
    i = int(np.searchsorted(ts, a.ts_ms, side="right"))
    return (np.insert(ts, i, a.ts_ms), np.insert(lat, i, a.lat), np.insert(lon, i, a.lon))


_Series = tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]
_EDGE_KINDS = frozenset({AnchorKind.OFFICE, AnchorKind.TRIP_START, AnchorKind.TRIP_END})
