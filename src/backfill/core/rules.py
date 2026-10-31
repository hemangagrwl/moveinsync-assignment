from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from backfill.core.config import RuleConfig
from backfill.core.geo import speed_kmh


class RuleCode(StrEnum):
    SHORT_GAP_AERIAL = "SHORT_GAP_AERIAL"
    PARKED = "PARKED"
    ROUTED = "ROUTED"
    DEGRADED_AERIAL = "DEGRADED_AERIAL"
    REJECTED_SPEED = "REJECTED_SPEED"
    REJECTED_MAX_DISTANCE = "REJECTED_MAX_DISTANCE"


RULE_CODES: tuple[RuleCode, ...] = tuple(RuleCode)
RULE_INDEX: dict[RuleCode, int] = {c: i for i, c in enumerate(RULE_CODES)}
REJECTED_CODES = frozenset({RuleCode.REJECTED_SPEED, RuleCode.REJECTED_MAX_DISTANCE})


@dataclass(frozen=True, slots=True)
class SegmentBatch:
    """All segments of one track. Segment i joins point i to point i+1."""

    dt_s: NDArray[np.float64]
    aerial_m: NDArray[np.float64]
    speed_kmh: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.dt_s)


class LocalRule(ABC):
    """A rule decided from the two endpoints alone, with no external call.

    Rules are evaluated in a fixed order. Each one claims the still-unclaimed segments it
    applies to, all at once over numpy arrays, and says how far each claimed segment is.
    """

    code: ClassVar[RuleCode]

    def __init__(self, config: RuleConfig) -> None:
        self._cfg = config

    @abstractmethod
    def claims(self, seg: SegmentBatch) -> NDArray[np.bool_]: ...

    def distance_m(self, seg: SegmentBatch, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        return seg.aerial_m[mask]


class ShortGapAerialRule(LocalRule):
    """Points less than ``short_gap_s`` apart at a plausible speed: straight line is accurate."""

    code = RuleCode.SHORT_GAP_AERIAL

    def claims(self, seg: SegmentBatch) -> NDArray[np.bool_]:
        return (seg.dt_s < self._cfg.short_gap_s) & (seg.speed_kmh <= self._cfg.max_speed_kmh)


class ParkedRule(LocalRule):
    """A long gap that implies walking pace or less: the cab was parked, use the straight line."""

    code = RuleCode.PARKED

    def claims(self, seg: SegmentBatch) -> NDArray[np.bool_]:
        return (seg.dt_s >= self._cfg.short_gap_s) & (seg.speed_kmh < self._cfg.parked_speed_kmh)


@dataclass(frozen=True, slots=True)
class RouteVerdict:
    code: RuleCode
    distance_m: float


class RoutedGapPolicy:
    """Judges a segment that needed the routing provider.

    ``routed_m`` is None when the provider could not answer (failure or open circuit).
    """

    def __init__(self, config: RuleConfig) -> None:
        self._cfg = config

    def judge(self, aerial_m: float, dt_s: float, routed_m: float | None) -> RouteVerdict:
        cfg = self._cfg
        fallback = aerial_m * cfg.aerial_fallback_factor
        if routed_m is None:
            return RouteVerdict(RuleCode.DEGRADED_AERIAL, fallback)
        if routed_m > cfg.max_routed_m:
            return RouteVerdict(RuleCode.REJECTED_MAX_DISTANCE, 0.0)
        tol = cfg.speed_tolerance_m
        routed_too_fast = speed_kmh(max(routed_m - tol, 0.0), dt_s, cfg.min_speed_dt_s) > cfg.max_speed_kmh
        fallback_too_fast = speed_kmh(max(fallback - tol, 0.0), dt_s, cfg.min_speed_dt_s) > cfg.max_speed_kmh
        if routed_too_fast and fallback_too_fast:
            return RouteVerdict(RuleCode.REJECTED_SPEED, 0.0)
        return RouteVerdict(RuleCode.ROUTED, routed_m)


@dataclass(frozen=True, slots=True)
class RuleSet:
    """The ordered, versioned set of rules. Order is part of the contract."""

    config: RuleConfig
    local_rules: tuple[LocalRule, ...]
    gap_policy: RoutedGapPolicy

    @classmethod
    def default(cls, config: RuleConfig | None = None) -> RuleSet:
        cfg = config or RuleConfig()
        return cls(cfg, (ShortGapAerialRule(cfg), ParkedRule(cfg)), RoutedGapPolicy(cfg))

    @property
    def version(self) -> str:
        return self.config.rule_version
