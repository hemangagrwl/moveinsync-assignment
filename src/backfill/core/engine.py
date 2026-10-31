from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backfill.core.anchors import AnchorBuilder, AnchorDecision, AnchorPlacer, AnchorStatus
from backfill.core.cleaning import CleanTrack, PingCleaner
from backfill.core.geo import haversine_m, haversine_np, speed_kmh_np
from backfill.core.models import TripEvidence
from backfill.core.routing import CellPair, GridSnapper, RouteResolver, pair_key
from backfill.core.rules import (
    REJECTED_CODES,
    RULE_CODES,
    RULE_INDEX,
    RuleCode,
    RuleSet,
    SegmentBatch,
)

_UNCLAIMED = -1
_AERIAL = RULE_INDEX[RuleCode.SHORT_GAP_AERIAL]


class TripStatus(StrEnum):
    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"  # at least one gap was estimated because the provider was down


class ReviewReason(StrEnum):
    REJECTED_SEGMENT = "REJECTED_SEGMENT"
    NO_TRACK = "NO_TRACK"
    RIDER_WITHOUT_ANCHOR = "RIDER_WITHOUT_ANCHOR"


def metres_to_invoice_km(metres: int) -> Decimal:
    """The single place where metres become billable kilometres: 2 decimals, half-up."""
    return (Decimal(metres) / Decimal(1000)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class LineItem:
    """One explainable piece of the trip.

    Consecutive SHORT_GAP_AERIAL segments are merged into one item (a dispute is never about
    the 4-second hop at 10:02:16); every other rule gets one item per segment.
    """

    rule: RuleCode
    from_point: int
    to_point: int
    start_ms: int
    end_ms: int
    segments: int
    distance_m: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.value,
            "from_point": self.from_point,
            "to_point": self.to_point,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segments": self.segments,
            "distance_m": self.distance_m,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class AnchorUse:
    label: str
    status: AnchorStatus
    ts_ms: int
    original_ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status.value,
            "ts_ms": self.ts_ms,
            "original_ts_ms": self.original_ts_ms,
        }


@dataclass(frozen=True, slots=True)
class Track:
    """Cleaned pings merged with the anchors that were used, in time order."""

    ts_ms: NDArray[np.int64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    is_anchor: NDArray[np.bool_]
    labels: dict[int, str]

    def __len__(self) -> int:
        return len(self.ts_ms)


@dataclass(frozen=True)
class TripDistance:
    trip_id: str
    rule_version: str
    input_fingerprint: str
    total_m: int
    status: TripStatus
    line_items: tuple[LineItem, ...]
    anchors: tuple[AnchorUse, ...]
    rejected_pings: dict[str, int]
    review_reasons: tuple[ReviewReason, ...]
    provider_calls: int
    routed_pairs: int
    cache_hits: int
    pinned_hits: int
    route_evidence: dict[str, float]
    rule_totals_m: dict[str, int]
    track: Track | None = field(default=None, repr=False, compare=False)

    @property
    def total_km(self) -> Decimal:
        return metres_to_invoice_km(self.total_m)

    @property
    def review_required(self) -> bool:
        return bool(self.review_reasons)

    def track_dict(self) -> dict[str, Any] | None:
        if self.track is None:
            return None
        return {
            "ts_ms": self.track.ts_ms.tolist(),
            "lat": self.track.lat.tolist(),
            "lon": self.track.lon.tolist(),
            "labels": {str(k): v for k, v in self.track.labels.items()},
        }

    def to_dict(self, *, include_track: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trip_id": self.trip_id,
            "rule_version": self.rule_version,
            "input_fingerprint": self.input_fingerprint,
            "total_m": self.total_m,
            "total_km": str(self.total_km),
            "status": self.status.value,
            "review_required": self.review_required,
            "review_reasons": [r.value for r in self.review_reasons],
            "rule_totals_m": self.rule_totals_m,
            "line_items": [li.to_dict() for li in self.line_items],
            "anchors": [a.to_dict() for a in self.anchors],
            "rejected_pings": self.rejected_pings,
            "provider": {
                "calls": self.provider_calls,
                "routed_pairs": self.routed_pairs,
                "cache_hits": self.cache_hits,
                "pinned_hits": self.pinned_hits,
            },
            "route_evidence": self.route_evidence,
        }
        if include_track:
            out["track"] = self.track_dict()
        return out


class DistanceEngine:
    """Rebuilds the billable distance of one trip.

    Pipeline: clean pings -> build anchors -> place anchors -> merge -> classify every segment
    with the ordered rule set -> route the remaining real gaps in one batch -> round each
    segment to whole metres -> sum. Same evidence + same rule version + same route answers
    always gives the same total, because every step is ordered and every tie is broken.
    """

    def __init__(self, ruleset: RuleSet, router: RouteResolver) -> None:
        self._rules = ruleset
        self._cfg = ruleset.config
        self._cleaner = PingCleaner(self._cfg)
        self._anchor_builder = AnchorBuilder(self._cfg)
        self._placer = AnchorPlacer(self._cfg)
        self._snapper = GridSnapper(self._cfg.cell_size_deg)
        self._router = router

    @property
    def ruleset(self) -> RuleSet:
        return self._rules

    def compute(
        self,
        ev: TripEvidence,
        *,
        pinned_routes: Mapping[str, float] | None = None,
        include_track: bool = False,
    ) -> TripDistance:
        cleaned = self._cleaner.clean(ev.pings, window_ms=(ev.start_time_ms, ev.end_time_ms))
        built = self._anchor_builder.build(ev)
        decisions = self._placer.place(built.anchors, cleaned.ts_ms, cleaned.lat, cleaned.lon)
        track = self._merge(cleaned, decisions)

        review: list[ReviewReason] = []
        if built.riders_without_anchor:
            review.append(ReviewReason.RIDER_WITHOUT_ANCHOR)
        anchor_uses = tuple(
            AnchorUse(d.anchor.label, d.status, d.anchor.ts_ms, d.original_ts_ms) for d in decisions
        )
        rejected = {k.value: v for k, v in cleaned.rejected.items()}

        if len(track) < 2:
            review.append(ReviewReason.NO_TRACK)
            return TripDistance(
                ev.trip_id,
                self._rules.version,
                self._fingerprint(track),
                0,
                TripStatus.COMPLETE,
                (),
                anchor_uses,
                rejected,
                tuple(review),
                0,
                0,
                0,
                0,
                {},
                {c.value: 0 for c in RULE_CODES},
                track if include_track else None,
            )

        seg = self._segments(track)
        codes = np.full(len(seg), _UNCLAIMED, dtype=np.int8)
        dist = np.zeros(len(seg), dtype=np.float64)
        for rule in self._rules.local_rules:
            mask = rule.claims(seg) & (codes == _UNCLAIMED)
            codes[mask] = RULE_INDEX[rule.code]
            dist[mask] = rule.distance_m(seg, mask)

        notes: dict[int, str] = {}
        routed = self._route_remaining(track, seg, codes, dist, notes, pinned_routes)
        resolution, routed_pairs = routed

        # Each segment is rounded to whole metres once; totals are exact integer sums after that.
        seg_m = np.floor(dist + 0.5).astype(np.int64)
        total_m = int(seg_m.sum())

        line_items = self._line_items(track, codes, seg_m, notes)
        rule_totals = {c.value: 0 for c in RULE_CODES}
        for li in line_items:
            rule_totals[li.rule.value] += li.distance_m

        degraded = bool((codes == RULE_INDEX[RuleCode.DEGRADED_AERIAL]).any())
        if any(RULE_CODES[c] in REJECTED_CODES for c in np.unique(codes)):
            review.append(ReviewReason.REJECTED_SEGMENT)

        evidence = {pair_key(p): d for p, d in sorted(resolution.distances.items()) if d is not None}
        return TripDistance(
            trip_id=ev.trip_id,
            rule_version=self._rules.version,
            input_fingerprint=self._fingerprint(track),
            total_m=total_m,
            status=TripStatus.DEGRADED if degraded else TripStatus.COMPLETE,
            line_items=line_items,
            anchors=anchor_uses,
            rejected_pings=rejected,
            review_reasons=tuple(review),
            provider_calls=resolution.provider_calls,
            routed_pairs=routed_pairs,
            cache_hits=resolution.count("cache"),
            pinned_hits=resolution.count("pinned"),
            route_evidence=evidence,
            rule_totals_m=rule_totals,
            track=track if include_track else None,
        )

    # -- pipeline steps -----------------------------------------------------------------

    @staticmethod
    def _merge(cleaned: CleanTrack, decisions: list[AnchorDecision]) -> Track:
        used = [d.anchor for d in decisions if d.status is not AnchorStatus.REDUNDANT]
        n = len(cleaned)
        ts = np.concatenate([cleaned.ts_ms, np.array([a.ts_ms for a in used], dtype=np.int64)])
        lat = np.concatenate([cleaned.lat, np.array([a.lat for a in used], dtype=np.float64)])
        lon = np.concatenate([cleaned.lon, np.array([a.lon for a in used], dtype=np.float64)])
        is_anchor = np.concatenate([np.zeros(n, dtype=bool), np.ones(len(used), dtype=bool)])
        # Time first; at equal time anchors before pings; then coordinates. Fully deterministic.
        order = np.lexsort((lon, lat, ~is_anchor, ts))
        position = np.empty(len(order), dtype=np.int64)
        position[order] = np.arange(len(order))
        labels = {int(position[n + j]): a.label for j, a in enumerate(used)}
        return Track(ts[order], lat[order], lon[order], is_anchor[order], labels)

    def _segments(self, t: Track) -> SegmentBatch:
        dt_s = np.diff(t.ts_ms) / 1000.0
        aerial = haversine_np(t.lat[:-1], t.lon[:-1], t.lat[1:], t.lon[1:])
        # Plausibility uses distance minus the positional uncertainty of the two fixes, so an
        # anchor 40 m from a ping one second away is not mistaken for a 144 km/h jump.
        moved = np.maximum(aerial - self._cfg.speed_tolerance_m, 0.0)
        return SegmentBatch(dt_s, aerial, speed_kmh_np(moved, dt_s, self._cfg.min_speed_dt_s))

    def _route_remaining(
        self,
        t: Track,
        seg: SegmentBatch,
        codes: NDArray[np.int8],
        dist: NDArray[np.float64],
        notes: dict[int, str],
        pinned: Mapping[str, float] | None,
    ) -> tuple[Any, int]:
        rest = np.flatnonzero(codes == _UNCLAIMED)
        if len(rest) == 0:
            return self._router.resolve([], pinned), 0
        rows, cols = self._snapper.cells(t.lat, t.lon)
        pairs: list[CellPair] = [
            (int(rows[i]), int(cols[i]), int(rows[i + 1]), int(cols[i + 1])) for i in rest
        ]
        resolution = self._router.resolve(pairs, pinned)
        policy = self._rules.gap_policy

        for i, pair in zip(rest, pairs, strict=True):
            aerial = float(seg.aerial_m[i])
            centre_routed = resolution.distances.get(pair)
            routed: float | None = None
            if centre_routed is not None:
                routed = self._scale_to_points(pair, centre_routed, aerial)
            verdict = policy.judge(aerial, float(seg.dt_s[i]), routed)
            codes[i] = RULE_INDEX[verdict.code]
            dist[i] = verdict.distance_m
            if verdict.code is RuleCode.DEGRADED_AERIAL:
                notes[int(i)] = (
                    f"provider unavailable; aerial {aerial:.0f} m x {self._cfg.aerial_fallback_factor}"
                )
            elif routed is not None:
                notes[int(i)] = f"routed {routed:.0f} m, aerial {aerial:.0f} m"
        return resolution, len(set(pairs))

    def _scale_to_points(self, pair: CellPair, centre_routed: float, aerial: float) -> float:
        """Carry the detour ratio of the centre-to-centre route over to the real endpoints."""
        a = self._snapper.centre(pair[0], pair[1])
        b = self._snapper.centre(pair[2], pair[3])
        centre_aerial = haversine_m(a[0], a[1], b[0], b[1])
        if centre_aerial < 1.0 or centre_routed <= 0.0:
            return aerial
        return centre_routed * (aerial / centre_aerial)

    @staticmethod
    def _line_items(
        t: Track, codes: NDArray[np.int8], seg_m: NDArray[np.int64], notes: dict[int, str]
    ) -> tuple[LineItem, ...]:
        k = len(codes)
        groupable = codes == _AERIAL
        starts_mask = np.ones(k, dtype=bool)
        starts_mask[1:] = ~(groupable[1:] & groupable[:-1])
        starts = np.flatnonzero(starts_mask)
        ends = np.append(starts[1:], k)
        cum = np.concatenate([[0], np.cumsum(seg_m)])
        items: list[LineItem] = []
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            parts = []
            if s in notes:
                parts.append(notes[s])
            for p in (s, e):
                if p in t.labels:
                    parts.append(("from " if p == s else "to ") + t.labels[p])
            items.append(
                LineItem(
                    rule=RULE_CODES[codes[s]],
                    from_point=s,
                    to_point=e,
                    start_ms=int(t.ts_ms[s]),
                    end_ms=int(t.ts_ms[e]),
                    segments=e - s,
                    distance_m=int(cum[e] - cum[s]),
                    note="; ".join(parts),
                )
            )
        return tuple(items)

    def _fingerprint(self, t: Track) -> str:
        h = hashlib.sha256()
        h.update(self._rules.version.encode())
        for arr in (t.ts_ms, t.lat, t.lon, t.is_anchor):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
