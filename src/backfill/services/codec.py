"""Converts stored trip JSON into the engine's immutable evidence objects."""

from __future__ import annotations

from typing import Any

from backfill.core.models import (
    GeoPoint,
    PingBatch,
    Rider,
    RiderStatus,
    RiderType,
    TripEvidence,
    TripType,
)


def _point(raw: dict[str, Any] | None) -> GeoPoint | None:
    if not raw:
        return None
    return GeoPoint(float(raw["lat"]), float(raw["lon"]))


def _rider(raw: dict[str, Any]) -> Rider:
    return Rider(
        employee_ref=raw["employee_ref"],
        rider_type=RiderType(raw.get("rider_type", "PLANNED")),
        status=RiderStatus(raw.get("status", "TRAVELLED")),
        planned_location=_point(raw.get("planned_location")),
        planned_time_ms=raw.get("planned_time_ms"),
        actual_location=_point(raw.get("actual_location")),
        actual_time_ms=raw.get("actual_time_ms"),
        profile_home=_point(raw.get("profile_home")),
        sign_in_time_ms=raw.get("sign_in_time_ms"),
        no_show_location=_point(raw.get("no_show_location")),
        no_show_time_ms=raw.get("no_show_time_ms"),
    )


def to_evidence(trip_ref: str, trip_type: str, evidence: dict[str, Any], pings: PingBatch) -> TripEvidence:
    office = _point(evidence["office"])
    assert office is not None
    return TripEvidence(
        trip_id=trip_ref,
        trip_type=TripType(trip_type),
        office=office,
        start_time_ms=int(evidence["start_time_ms"]),
        end_time_ms=int(evidence["end_time_ms"]),
        pings=pings,
        riders=tuple(_rider(r) for r in evidence.get("riders", [])),
        start_location=_point(evidence.get("start_location")),
        end_location=_point(evidence.get("end_location")),
    )


def rider_refs(evidence: dict[str, Any]) -> set[str]:
    return {r["employee_ref"] for r in evidence.get("riders", [])}
