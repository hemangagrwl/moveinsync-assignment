from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from backfill.errors import Conflict, NotFound
from backfill.observability import metrics as m
from backfill.observability.logging import get_logger
from backfill.persistence.db import SessionFactory
from backfill.persistence.orm import ComputationRow, LedgerRow, TripRow
from backfill.persistence.repositories import (
    ComputationRepository,
    LedgerRepository,
    PingRepository,
    TripRepository,
)
from backfill.services.views import LedgerEntryView

log = get_logger(__name__)


class BillingService:
    """Owns the append-only invoice ledger.

    Closing an invoice writes an ORIGINAL entry. After that the invoice is never edited: a
    recompute on newer evidence that changes the distance appends a signed ADJUSTMENT whose
    ``delta_m`` brings the balance to the new total.
    """

    def __init__(self, sessions: SessionFactory, clock_ms: Callable[[], int]) -> None:
        self._sessions = sessions
        self._now = clock_ms

    def close_invoice(self, tenant_id: str, trip_ref: str) -> LedgerEntryView:
        with self._sessions.begin() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref, for_update=True)
            if trip is None:
                raise NotFound("Trip not found.")
            if trip.invoice_status == "CLOSED":
                raise Conflict("This invoice is already closed.", code="INVOICE_ALREADY_CLOSED")
            latest = ComputationRepository(s).latest(trip.id)
            if latest is None or latest.ping_watermark < PingRepository(s).max_id(trip.id):
                raise Conflict("Distance is out of date; compute it before closing.", code="DISTANCE_STALE")
            if latest.status != "COMPLETE":
                raise Conflict(
                    "Distance is an estimate while the routing provider is down; retry after recovery.",
                    code="DISTANCE_DEGRADED",
                )
            entry = LedgerRepository(s).add(
                LedgerRow(
                    trip_id=trip.id,
                    tenant_id=trip.tenant_id,
                    entry_type="ORIGINAL",
                    computation_id=latest.id,
                    delta_m=latest.total_m,
                    balance_m=latest.total_m,
                    rule_version=latest.rule_version,
                    reason="INVOICE_CLOSED",
                    created_at_ms=self._now(),
                )
            )
            trip.invoice_status = "CLOSED"
            trip.billed_watermark = latest.ping_watermark
            m.INVOICES_CLOSED.inc()
            log.info("invoice_closed", trip_id=trip.id, total_m=latest.total_m)
            return LedgerEntryView.of(entry)

    def apply_computation(self, s: Session, trip: TripRow, row: ComputationRow) -> LedgerRow | None:
        """Called inside the computation's transaction, with the trip row locked."""
        if trip.invoice_status != "CLOSED" or row.status != "COMPLETE":
            return None
        if row.ping_watermark < trip.billed_watermark:
            return None  # a slower recompute on older evidence finished late: ignore it
        ledger = LedgerRepository(s)
        if ledger.applied(row.id):
            return None
        trip.billed_watermark = row.ping_watermark
        balance = ledger.balance(trip.id)
        delta = row.total_m - balance
        if delta == 0:
            return None
        entry = ledger.add(
            LedgerRow(
                trip_id=trip.id,
                tenant_id=trip.tenant_id,
                entry_type="ADJUSTMENT",
                computation_id=row.id,
                delta_m=delta,
                balance_m=row.total_m,
                rule_version=row.rule_version,
                reason=row.reason,
                created_at_ms=self._now(),
            )
        )
        m.LEDGER_ADJUSTMENTS.inc()
        m.LEDGER_ADJUSTMENT_METRES.inc(abs(delta))
        log.info("ledger_adjustment", trip_id=trip.id, delta_m=delta, reason=row.reason)
        return entry

    def ledger(self, tenant_id: str, trip_ref: str) -> list[LedgerEntryView]:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            return [LedgerEntryView.of(e) for e in LedgerRepository(s).entries(trip.id)]
