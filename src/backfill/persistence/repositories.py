"""Repositories. Every read that starts from client input takes a tenant id, so a query can
never return another tenant's rows; lookups by internal id are only used by the worker."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backfill.core.models import E7, PingBatch
from backfill.errors import Conflict
from backfill.persistence.orm import ComputationRow, LedgerRow, PingRow, RecomputeJobRow, TripRow

PING_INSERT_CHUNK = 2_000


class TripRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, row: TripRow) -> TripRow:
        self._s.add(row)
        try:
            with self._s.begin_nested():
                self._s.flush()
        except IntegrityError as exc:
            raise Conflict("A trip with this reference already exists.", code="TRIP_EXISTS") from exc
        return row

    def get(self, tenant_id: str, trip_ref: str, *, for_update: bool = False) -> TripRow | None:
        stmt = select(TripRow).where(TripRow.tenant_id == tenant_id, TripRow.trip_ref == trip_ref)
        if for_update:
            stmt = stmt.with_for_update()
        return self._s.scalars(stmt).one_or_none()

    def get_by_id(self, trip_id: int, *, for_update: bool = False) -> TripRow | None:
        stmt = select(TripRow).where(TripRow.id == trip_id)
        if for_update:
            stmt = stmt.with_for_update()
        return self._s.scalars(stmt).one_or_none()


class PingRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def insert_many(self, trip_id: int, rows: Sequence[dict[str, Any]]) -> int:
        """Insert, silently skipping pings already stored. Returns how many were new."""
        if not rows:
            return 0
        before = self.count(trip_id)
        dialect = self._s.get_bind().dialect.name
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        for start in range(0, len(rows), PING_INSERT_CHUNK):
            chunk = [dict(r, trip_id=trip_id) for r in rows[start : start + PING_INSERT_CHUNK]]
            stmt = insert(PingRow).values(chunk).on_conflict_do_nothing()
            self._s.execute(stmt)
        return self.count(trip_id) - before

    def count(self, trip_id: int) -> int:
        return int(self._s.scalar(select(func.count()).where(PingRow.trip_id == trip_id)) or 0)

    def max_id(self, trip_id: int) -> int:
        return int(self._s.scalar(select(func.max(PingRow.id)).where(PingRow.trip_id == trip_id)) or 0)

    def load(self, trip_id: int) -> tuple[PingBatch, int]:
        """All pings of a trip plus the highest ping id seen (the evidence watermark)."""
        rows = self._s.execute(
            select(PingRow.id, PingRow.ts_ms, PingRow.lat_e7, PingRow.lon_e7, PingRow.accuracy_dm).where(
                PingRow.trip_id == trip_id
            )
        ).all()
        if not rows:
            return PingBatch.empty(), 0
        arr = np.array([(r[1], r[2], r[3], -1 if r[4] is None else r[4]) for r in rows], dtype=np.int64)
        acc = np.where(arr[:, 3] < 0, np.nan, arr[:, 3] / 10.0)
        batch = PingBatch.from_arrays(arr[:, 0], arr[:, 1] / E7, arr[:, 2] / E7, acc)
        return batch, max(r[0] for r in rows)


class ComputationRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, row: ComputationRow) -> ComputationRow:
        self._s.add(row)
        self._s.flush()
        return row

    def find(self, trip_id: int, fingerprint: str, status: str) -> ComputationRow | None:
        return self._s.scalars(
            select(ComputationRow).where(
                ComputationRow.trip_id == trip_id,
                ComputationRow.input_fingerprint == fingerprint,
                ComputationRow.status == status,
            )
        ).one_or_none()

    def latest(self, trip_id: int, *, complete_only: bool = False) -> ComputationRow | None:
        stmt = select(ComputationRow).where(ComputationRow.trip_id == trip_id)
        if complete_only:
            stmt = stmt.where(ComputationRow.status == "COMPLETE")
        stmt = stmt.order_by(ComputationRow.ping_watermark.desc(), ComputationRow.id.desc()).limit(1)
        return self._s.scalars(stmt).one_or_none()

    def history(self, trip_id: int) -> list[ComputationRow]:
        return list(
            self._s.scalars(
                select(ComputationRow).where(ComputationRow.trip_id == trip_id).order_by(ComputationRow.id)
            )
        )


class LedgerRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def balance(self, trip_id: int) -> int:
        last = self._s.scalars(
            select(LedgerRow).where(LedgerRow.trip_id == trip_id).order_by(LedgerRow.id.desc()).limit(1)
        ).one_or_none()
        return last.balance_m if last else 0

    def applied(self, computation_id: int) -> bool:
        stmt = select(LedgerRow.id).where(LedgerRow.computation_id == computation_id)
        return self._s.scalar(stmt) is not None

    def add(self, row: LedgerRow) -> LedgerRow:
        self._s.add(row)
        self._s.flush()
        return row

    def entries(self, trip_id: int) -> list[LedgerRow]:
        return list(
            self._s.scalars(select(LedgerRow).where(LedgerRow.trip_id == trip_id).order_by(LedgerRow.id))
        )


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def enqueue(self, trip: TripRow, reason: str, available_at_ms: int, now_ms: int) -> RecomputeJobRow:
        """One pending job per trip. A new late-ping batch pushes the due time back (debounce)."""
        pending = self._s.scalars(
            select(RecomputeJobRow).where(
                RecomputeJobRow.trip_id == trip.id, RecomputeJobRow.status == "PENDING"
            )
        ).first()
        if pending is not None:
            if reason == "LATE_PINGS":
                pending.available_at_ms = max(pending.available_at_ms, available_at_ms)
                pending.reason = reason
            return pending
        job = RecomputeJobRow(
            trip_id=trip.id,
            tenant_id=trip.tenant_id,
            reason=reason,
            status="PENDING",
            attempts=0,
            available_at_ms=available_at_ms,
            created_at_ms=now_ms,
        )
        self._s.add(job)
        self._s.flush()
        return job

    def claim_next(self, worker_id: str, now_ms: int, lease_ms: int) -> RecomputeJobRow | None:
        """Claim the oldest due job with a compare-and-set, so two workers never run the same
        job. A RUNNING job whose lease expired (its worker crashed) is claimable again."""
        due = or_(
            (RecomputeJobRow.status == "PENDING") & (RecomputeJobRow.available_at_ms <= now_ms),
            (RecomputeJobRow.status == "RUNNING") & (RecomputeJobRow.claimed_at_ms < now_ms - lease_ms),
        )
        candidates = self._s.scalars(
            select(RecomputeJobRow.id).where(due).order_by(RecomputeJobRow.available_at_ms).limit(5)
        ).all()
        for job_id in candidates:
            result = self._s.execute(
                update(RecomputeJobRow)
                .where(RecomputeJobRow.id == job_id, due)
                .values(
                    status="RUNNING",
                    claimed_by=worker_id,
                    claimed_at_ms=now_ms,
                    attempts=RecomputeJobRow.attempts + 1,
                )
            )
            if result.rowcount == 1:  # type: ignore[attr-defined]
                return self._s.get(RecomputeJobRow, job_id)
        return None

    def complete(self, job_id: int) -> None:
        self._s.execute(update(RecomputeJobRow).where(RecomputeJobRow.id == job_id).values(status="DONE"))

    def fail(self, job_id: int, error_code: str, retry_at_ms: int, max_attempts: int) -> str:
        job = self._s.get(RecomputeJobRow, job_id)
        if job is None:
            return "MISSING"
        job.last_error = error_code[:64]
        if job.attempts >= max_attempts:
            job.status = "DEAD"
        else:
            job.status = "PENDING"
            job.available_at_ms = retry_at_ms
        return job.status

    def release_degraded(self, now_ms: int) -> int:
        result = self._s.execute(
            update(RecomputeJobRow)
            .where(RecomputeJobRow.status == "PENDING", RecomputeJobRow.reason == "PROVIDER_DEGRADED")
            .values(available_at_ms=now_ms)
        )
        return int(result.rowcount)  # type: ignore[attr-defined]

    def queue_stats(self, now_ms: int, tenant_id: str | None = None) -> tuple[int, float]:
        """(pending jobs, seconds the oldest due job has been waiting)."""
        where = [RecomputeJobRow.status == "PENDING"]
        if tenant_id is not None:
            where.append(RecomputeJobRow.tenant_id == tenant_id)
        depth = int(self._s.scalar(select(func.count()).where(*where)) or 0)
        oldest = self._s.scalar(
            select(func.min(RecomputeJobRow.available_at_ms)).where(
                *where, RecomputeJobRow.available_at_ms <= now_ms
            )
        )
        lag = max(0.0, (now_ms - oldest) / 1000.0) if oldest is not None else 0.0
        return depth, lag

    def jobs_for_trip(self, trip_id: int) -> list[RecomputeJobRow]:
        return list(
            self._s.scalars(
                select(RecomputeJobRow).where(RecomputeJobRow.trip_id == trip_id).order_by(RecomputeJobRow.id)
            )
        )
