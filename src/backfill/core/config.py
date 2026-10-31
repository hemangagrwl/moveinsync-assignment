from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Every threshold that can change a billed metre.

    The config is part of the rule version: changing any value changes ``fingerprint()``,
    so a stored result always records exactly which thresholds produced it.
    """

    version: str = "2026.10-v1"

    # Cleaning
    max_speed_kmh: float = 120.0
    max_accuracy_m: float = 50.0
    min_speed_dt_s: float = 1.0
    teleport_reset_count: int = 10
    min_move_m: float = 8.0

    # Segment rules
    short_gap_s: float = 70.0
    parked_speed_kmh: float = 5.0
    max_routed_m: float = 100_000.0
    aerial_fallback_factor: float = 1.4
    speed_tolerance_m: float = 50.0

    # Anchors
    office_radius_m: float = 400.0
    anchor_max_deviation_m: float = 2_000.0
    anchor_time_grace_s: float = 900.0
    anchor_coverage_s: float = 60.0
    fallback_anchor_search_s: float = 600.0
    anchor_visit_radius_m: float = 150.0
    anchor_visit_window_s: float = 1800.0
    retime_metres_per_s: float = 8.0
    include_no_show_anchors: bool = False

    # Routing cache cells (~55 m at Indian latitudes)
    cell_size_deg: float = 0.0005

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def rule_version(self) -> str:
        return f"{self.version}#{self.fingerprint()}"
