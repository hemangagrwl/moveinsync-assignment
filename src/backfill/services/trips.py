from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from backfill.core.engine import DistanceEngine, TripDistance
from backfill.core.models import E7
from backfill.errors import NotFound
from backfill.observability import metrics as m
from backfill.observability.logging import get_logger
from backfill.persistence.db import SessionFactory
from backfill.persistence.orm import ComputationRow, TripRow
from backfill.persistence.repositories import (
    ComputationRepository,
    JobRepository,
    PingRepository,
    TripRepository,
)
from backfill.services import codec
from backfill.services.billing import BillingService
from backfill.services.views import ComputationView, IngestResult, TripView

log = get_logger(__name__)


class TripService:
    """Ingest evidence and turn it into stored, billable distances.

    The engine call (which may wait on the routing provider) runs *outside* any database
    transaction; the result is then written in a short transaction with the trip row locked.
    Writing is idempotent: the same evidence under the same rule version maps to the same
    stored computation, so retries and concurrent recomputes never double-bill.
    """

    def __init__(
        self,
        sessions: SessionFactory,
        engine: DistanceEngine,
        billing: BillingService,
        clock_ms: Callable[[], int],
        late_ping_debounce_s: float = 2.0,
        degraded_retry_s: float = 15.0,
    ) -> None:
        self._sessions = sessions
        self._engine = engine
        self._billing = billing
        self._now = clock_ms
        self._debounce_ms = int(late_ping_debounce_s * 1000)
        self._degraded_retry_ms = int(degraded_retry_s * 1000)

    # -- commands -------------------------------------------------------------------------

    def create_trip(
        self, tenant_id: str, trip_ref: str, vendor_id: str, trip_type: str, evidence: dict[str, Any]
    ) -> TripView:
        with self._sessions.begin() as s:
            row = TripRepository(s).add(
                TripRow(
                    tenant_id=tenant_id,
                    trip_ref=trip_ref,
                    vendor_id=vendor_id,
                    trip_type=trip_type,
                    evidence=evidence,
                    invoice_status="OPEN",
                    billed_watermark=0,
                    created_at_ms=self._now(),
                )
            )
            log.info("trip_created", trip_id=row.id, rider_count=len(evidence.get("riders", [])))
            return TripView.of(row)

    def ingest_pings(self, tenant_id: str, trip_ref: str, pings: Sequence[dict[str, Any]]) -> IngestResult:
        now = self._now()
        rows = [
            {
                "ts_ms": int(p["ts_ms"]),
                "lat_e7": round(float(p["lat"]) * E7),
                "lon_e7": round(float(p["lon"]) * E7),
                "accuracy_dm": None
                if p.get("accuracy_m") is None or not math.isfinite(p["accuracy_m"])
                else round(float(p["accuracy_m"]) * 10),
                "received_at_ms": now,
            }
            for p in pings
        ]
        with self._sessions.begin() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref, for_update=True)
            if trip is None:
                raise NotFound("Trip not found.")
            accepted = PingRepository(s).insert_many(trip.id, rows)
            scheduled = False
            if accepted and ComputationRepository(s).latest(trip.id) is not None:
                JobRepository(s).enqueue(trip, "LATE_PINGS", now + self._debounce_ms, now)
                scheduled = True
        m.PINGS_INGESTED.inc(accepted)
        m.PINGS_DUPLICATE_ON_INGEST.inc(len(rows) - accepted)
        log.info("pings_ingested", trip_id=trip.id, received=len(rows), accepted=accepted, late=scheduled)
        return IngestResult(len(rows), accepted, len(rows) - accepted, scheduled)

    def compute(self, tenant_id: str, trip_ref: str, reason: str = "ON_DEMAND") -> ComputationView:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            trip_id = trip.id
        return self.compute_by_id(trip_id, reason)

    def compute_by_id(self, trip_id: int, reason: str) -> ComputationView:
        with self._sessions() as s:
            trip = TripRepository(s).get_by_id(trip_id)
            if trip is None:
                raise NotFound("Trip not found.")
            batch, watermark = PingRepository(s).load(trip.id)
            last_ok = ComputationRepository(s).latest(trip.id, complete_only=True)
            pinned = last_ok.result.get("route_evidence") if last_ok else None
            evidence = codec.to_evidence(trip.trip_ref, trip.trip_type, trip.evidence, batch)

        started = time.perf_counter()
        result = self._engine.compute(evidence, pinned_routes=pinned, include_track=True)
        m.COMPUTE_SECONDS.observe(time.perf_counter() - started)

        with self._sessions.begin() as s:
            trip = TripRepository(s).get_by_id(trip_id, for_update=True)
            assert trip is not None
            comps = ComputationRepository(s)
            row = comps.find(trip.id, result.input_fingerprint, result.status.value)
            created = row is None
            if row is None:
                row = comps.add(self._row(trip, result, watermark, reason))
            self._billing.apply_computation(s, trip, row)
            if result.status.value == "DEGRADED":
                now = self._now()
                JobRepository(s).enqueue(trip, "PROVIDER_DEGRADED", now + self._degraded_retry_ms, now)
            view = ComputationView.of(row, trip.trip_ref)

        if created:
            self._record_metrics(result, reason)
        log.info(
            "distance_computed",
            trip_id=trip_id,
            total_m=result.total_m,
            status=result.status.value,
            new=created,
            provider_calls=result.provider_calls,
            reason=reason,
        )
        return view

    # -- queries --------------------------------------------------------------------------

    def get_trip(self, tenant_id: str, trip_ref: str) -> TripView:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            return TripView.of(trip)

    def latest(self, tenant_id: str, trip_ref: str) -> ComputationView:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            row = ComputationRepository(s).latest(trip.id)
            if row is None:
                raise NotFound("No distance has been computed for this trip yet.", code="NOT_COMPUTED")
            return ComputationView.of(row, trip.trip_ref)

    def latest_with_track(self, tenant_id: str, trip_ref: str) -> tuple[ComputationView, dict[str, Any]]:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            row = ComputationRepository(s).latest(trip.id, with_track=True)
            if row is None:
                raise NotFound("No distance has been computed for this trip yet.", code="NOT_COMPUTED")
            return ComputationView.of(row, trip.trip_ref), row.track or {}

    def history(self, tenant_id: str, trip_ref: str) -> list[ComputationView]:
        with self._sessions() as s:
            trip = TripRepository(s).get(tenant_id, trip_ref)
            if trip is None:
                raise NotFound("Trip not found.")
            return [ComputationView.of(r, trip.trip_ref) for r in ComputationRepository(s).history(trip.id)]

    # -- helpers --------------------------------------------------------------------------

    def _row(self, trip: TripRow, result: TripDistance, watermark: int, reason: str) -> ComputationRow:
        return ComputationRow(
            trip_id=trip.id,
            tenant_id=trip.tenant_id,
            rule_version=result.rule_version,
            input_fingerprint=result.input_fingerprint,
            status=result.status.value,
            total_m=result.total_m,
            review_required=result.review_required,
            ping_watermark=watermark,
            provider_calls=result.provider_calls,
            reason=reason,
            result=result.to_dict(),
            track=result.track_dict(),
            created_at_ms=self._now(),
        )

    @staticmethod
    def _record_metrics(result: TripDistance, reason: str) -> None:
        m.COMPUTATIONS.labels(result.status.value, reason).inc()
        if result.review_required:
            m.REVIEW_REQUIRED.inc()
        for k, v in result.rejected_pings.items():
            if v:
                m.PINGS_REJECTED.labels(k).inc(v)
        for li in result.line_items:
            m.SEGMENTS.labels(li.rule.value).inc(li.segments)
        for rule, metres in result.rule_totals_m.items():
            if metres:
                m.SEGMENT_METRES.labels(rule).inc(metres)
