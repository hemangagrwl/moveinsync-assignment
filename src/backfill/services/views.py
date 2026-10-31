"""Plain read models returned by services, so the API layer never touches ORM rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backfill.core.engine import metres_to_invoice_km
from backfill.persistence.orm import ComputationRow, LedgerRow, TripRow


@dataclass(frozen=True)
class TripView:
    trip_ref: str
    vendor_id: str
    trip_type: str
    invoice_status: str
    created_at_ms: int
    rider_refs: frozenset[str]

    @classmethod
    def of(cls, row: TripRow) -> TripView:
        refs = frozenset(r["employee_ref"] for r in row.evidence.get("riders", []))
        return cls(row.trip_ref, row.vendor_id, row.trip_type, row.invoice_status, row.created_at_ms, refs)


@dataclass(frozen=True)
class ComputationView:
    computation_id: int
    trip_ref: str
    status: str
    total_m: int
    rule_version: str
    input_fingerprint: str
    review_required: bool
    reason: str
    created_at_ms: int
    result: dict[str, Any]

    @property
    def total_km(self) -> str:
        return str(metres_to_invoice_km(self.total_m))

    @classmethod
    def of(cls, row: ComputationRow, trip_ref: str) -> ComputationView:
        return cls(
            row.id,
            trip_ref,
            row.status,
            row.total_m,
            row.rule_version,
            row.input_fingerprint,
            row.review_required,
            row.reason,
            row.created_at_ms,
            row.result,
        )


@dataclass(frozen=True)
class LedgerEntryView:
    entry_id: int
    entry_type: str
    delta_m: int
    balance_m: int
    rule_version: str
    reason: str
    computation_id: int
    created_at_ms: int

    @property
    def delta_km(self) -> str:
        km = metres_to_invoice_km(abs(self.delta_m))
        return f"{'+' if self.delta_m >= 0 else '-'}{km}"

    @property
    def balance_km(self) -> str:
        return str(metres_to_invoice_km(self.balance_m))

    @classmethod
    def of(cls, row: LedgerRow) -> LedgerEntryView:
        return cls(
            row.id,
            row.entry_type,
            row.delta_m,
            row.balance_m,
            row.rule_version,
            row.reason,
            row.computation_id,
            row.created_at_ms,
        )


@dataclass(frozen=True)
class IngestResult:
    received: int
    accepted: int
    duplicates: int
    recompute_scheduled: bool
