from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backfill.core.models import RiderStatus, RiderType, TripType

REF = r"^[A-Za-z0-9_.:-]{1,64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeoPointIn(_Strict):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RiderIn(_Strict):
    employee_ref: str = Field(pattern=REF, description="Opaque employee id. Never a name or phone.")
    rider_type: RiderType = RiderType.PLANNED
    status: RiderStatus = RiderStatus.TRAVELLED
    planned_location: GeoPointIn | None = None
    planned_time_ms: int | None = None
    actual_location: GeoPointIn | None = Field(
        None, description="Reported pickup/drop point. May be bad data; the engine validates it."
    )
    actual_time_ms: int | None = None
    profile_home: GeoPointIn | None = None
    sign_in_time_ms: int | None = None
    no_show_location: GeoPointIn | None = None
    no_show_time_ms: int | None = None


class TripCreate(_Strict):
    trip_ref: str = Field(pattern=REF)
    vendor_id: str = Field(pattern=REF)
    trip_type: TripType
    office: GeoPointIn
    start_time_ms: int = Field(gt=0)
    end_time_ms: int = Field(gt=0)
    start_location: GeoPointIn | None = None
    end_location: GeoPointIn | None = None
    riders: list[RiderIn] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def _window(self) -> TripCreate:
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("end_time_ms must be after start_time_ms")
        if self.end_time_ms - self.start_time_ms > 24 * 3600 * 1000:
            raise ValueError("a trip cannot last more than 24 hours")
        if len({r.employee_ref for r in self.riders}) != len(self.riders):
            raise ValueError("employee_ref must be unique within a trip")
        return self

    def evidence(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"trip_ref", "vendor_id", "trip_type"})


class PingIn(_Strict):
    ts_ms: int = Field(gt=0)
    # Out-of-range coordinates are device faults, not client errors: they are stored and then
    # rejected by cleaning (and counted), so the bounds here only guard the storage format.
    lat: float = Field(ge=-1000, le=1000)
    lon: float = Field(ge=-1000, le=1000)
    accuracy_m: float | None = Field(None, ge=0, le=100_000)


class PingBatchIn(_Strict):
    pings: list[PingIn] = Field(min_length=1, max_length=20_000)


class LoginIn(_Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TripOut(BaseModel):
    trip_ref: str
    vendor_id: str
    trip_type: str
    invoice_status: str
    created_at_ms: int


class IngestOut(BaseModel):
    received: int
    accepted: int
    duplicates: int
    recompute_scheduled: bool


class DistanceOut(BaseModel):
    trip_ref: str
    computation_id: int
    status: str
    total_m: int
    total_km: str
    rule_version: str
    input_fingerprint: str
    review_required: bool
    reason: str
    created_at_ms: int
    review_reasons: list[str] = []
    rule_totals_m: dict[str, int] = {}
    line_items: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    rejected_pings: dict[str, int] = {}
    provider: dict[str, int] = {}


class EmployeeDistanceOut(BaseModel):
    """What a rider may see: the trip total and how it was built, without other riders' data."""

    trip_ref: str
    status: str
    total_km: str
    rule_version: str
    rule_totals_m: dict[str, int]


class LedgerEntryOut(BaseModel):
    entry_id: int
    entry_type: str
    delta_m: int
    delta_km: str
    balance_m: int
    balance_km: str
    rule_version: str
    reason: str
    computation_id: int
    created_at_ms: int


class OpsStatusOut(BaseModel):
    breaker_state: str
    route_cache_entries: int
    route_cache_hits: int
    route_cache_misses: int
    recompute_queue_depth: int
    recompute_lag_s: float
    provider_mode: str


class ChaosIn(_Strict):
    down: bool
