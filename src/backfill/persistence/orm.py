from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column

# SQLite only auto-increments an INTEGER PRIMARY KEY; Postgres gets a real BIGINT.
PK = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


class TripRow(Base):
    __tablename__ = "trips"
    __table_args__ = (UniqueConstraint("tenant_id", "trip_ref", name="uq_trip_ref"),)

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    trip_ref: Mapped[str] = mapped_column(String(64))
    vendor_id: Mapped[str] = mapped_column(String(64))
    trip_type: Mapped[str] = mapped_column(String(16))
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON)  # contains home locations: never logged
    invoice_status: Mapped[str] = mapped_column(String(16), default="OPEN")
    billed_watermark: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at_ms: Mapped[int] = mapped_column(BigInteger)


class PingRow(Base):
    """Raw pings exactly as received (coordinates in 1e-7 degrees). Never updated or deleted."""

    __tablename__ = "pings"
    __table_args__ = (UniqueConstraint("trip_id", "ts_ms", "lat_e7", "lon_e7", name="uq_ping"),)

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), index=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger)
    lat_e7: Mapped[int] = mapped_column(BigInteger)
    lon_e7: Mapped[int] = mapped_column(BigInteger)
    accuracy_dm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    received_at_ms: Mapped[int] = mapped_column(BigInteger)


class ComputationRow(Base):
    """Append-only: every distinct result ever computed for a trip."""

    __tablename__ = "distance_computations"
    __table_args__ = (
        UniqueConstraint("trip_id", "input_fingerprint", "status", name="uq_computation_input"),
        Index("ix_computation_latest", "trip_id", "ping_watermark", "id"),
    )

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"))
    tenant_id: Mapped[str] = mapped_column(String(64))
    rule_version: Mapped[str] = mapped_column(String(64))
    input_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    total_m: Mapped[int] = mapped_column(BigInteger)
    review_required: Mapped[bool] = mapped_column(Boolean)
    ping_watermark: Mapped[int] = mapped_column(BigInteger)
    provider_calls: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(32))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    # The merged track (coordinates) is only needed by the map view, so it is not loaded by default.
    track: Mapped[dict[str, Any] | None] = deferred(mapped_column(JSON, nullable=True))
    created_at_ms: Mapped[int] = mapped_column(BigInteger)


class LedgerRow(Base):
    """Append-only invoice ledger. A closed invoice is never edited; late evidence adds a signed delta."""

    __tablename__ = "invoice_ledger"

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    entry_type: Mapped[str] = mapped_column(String(16))  # ORIGINAL | ADJUSTMENT
    computation_id: Mapped[int] = mapped_column(ForeignKey("distance_computations.id"), unique=True)
    delta_m: Mapped[int] = mapped_column(BigInteger)
    balance_m: Mapped[int] = mapped_column(BigInteger)
    rule_version: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(32))
    created_at_ms: Mapped[int] = mapped_column(BigInteger)


class RecomputeJobRow(Base):
    __tablename__ = "recompute_jobs"
    __table_args__ = (Index("ix_job_due", "status", "available_at_ms"),)

    id: Mapped[int] = mapped_column(PK, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(32))  # LATE_PINGS | PROVIDER_DEGRADED
    status: Mapped[str] = mapped_column(String(16))  # PENDING | RUNNING | DONE | DEAD
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at_ms: Mapped[int] = mapped_column(BigInteger)
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger)
