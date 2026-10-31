from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from backfill.api import mapview
from backfill.api.deps import Admin, AnyUser, ContainerDep, Staff
from backfill.api.schemas import (
    ChaosIn,
    DistanceOut,
    EmployeeDistanceOut,
    IngestOut,
    LedgerEntryOut,
    LoginIn,
    OpsStatusOut,
    PingBatchIn,
    TokenOut,
    TripCreate,
    TripOut,
)
from backfill.api.security import Principal, Role
from backfill.container import Container, wall_clock_ms
from backfill.errors import Conflict, NotFound
from backfill.persistence.repositories import JobRepository
from backfill.providers.mock import MockRoutingProvider
from backfill.services.views import ComputationView, LedgerEntryView, TripView

auth = APIRouter(prefix="/v1/auth", tags=["auth"])
trips = APIRouter(prefix="/v1/trips", tags=["trips"])
ops = APIRouter(prefix="/v1/ops", tags=["ops"])
health = APIRouter(tags=["health"])


# -- auth ----------------------------------------------------------------------------------------


@auth.post("/token", response_model=TokenOut)
def login(body: LoginIn, c: ContainerDep) -> TokenOut:
    principal = c.users.authenticate(body.username, body.password)
    return TokenOut(access_token=c.tokens.issue(principal), expires_in=c.tokens.ttl_s)


# -- trips ---------------------------------------------------------------------------------------


def _trip_out(t: TripView) -> TripOut:
    return TripOut(
        trip_ref=t.trip_ref,
        vendor_id=t.vendor_id,
        trip_type=t.trip_type,
        invoice_status=t.invoice_status,
        created_at_ms=t.created_at_ms,
    )


def _distance_out(v: ComputationView) -> DistanceOut:
    r = v.result
    return DistanceOut(
        trip_ref=v.trip_ref,
        computation_id=v.computation_id,
        status=v.status,
        total_m=v.total_m,
        total_km=v.total_km,
        rule_version=v.rule_version,
        input_fingerprint=v.input_fingerprint,
        review_required=v.review_required,
        reason=v.reason,
        created_at_ms=v.created_at_ms,
        review_reasons=r.get("review_reasons", []),
        rule_totals_m=r.get("rule_totals_m", {}),
        line_items=r.get("line_items", []),
        anchors=r.get("anchors", []),
        rejected_pings=r.get("rejected_pings", {}),
        provider=r.get("provider", {}),
    )


def _ledger_out(e: LedgerEntryView) -> LedgerEntryOut:
    return LedgerEntryOut(
        entry_id=e.entry_id,
        entry_type=e.entry_type,
        delta_m=e.delta_m,
        delta_km=e.delta_km,
        balance_m=e.balance_m,
        balance_km=e.balance_km,
        rule_version=e.rule_version,
        reason=e.reason,
        computation_id=e.computation_id,
        created_at_ms=e.created_at_ms,
    )


def _visible_trip(c: Container, p: Principal, trip_ref: str) -> TripView:
    """Tenant isolation comes from the tenant id in the token. An employee additionally only
    sees trips they rode in; anything else is a 404, so existence is not leaked."""
    trip = c.trips.get_trip(p.tenant_id, trip_ref)
    if p.role is Role.EMPLOYEE and p.employee_ref not in trip.rider_refs:
        raise NotFound("Trip not found.")
    return trip


@trips.post("", response_model=TripOut, status_code=201)
def create_trip(body: TripCreate, p: Staff, c: ContainerDep) -> TripOut:
    view = c.trips.create_trip(
        p.tenant_id, body.trip_ref, body.vendor_id, body.trip_type.value, body.evidence()
    )
    return _trip_out(view)


@trips.get("/{trip_ref}", response_model=TripOut)
def get_trip(trip_ref: str, p: AnyUser, c: ContainerDep) -> TripOut:
    return _trip_out(_visible_trip(c, p, trip_ref))


@trips.post("/{trip_ref}/pings", response_model=IngestOut, status_code=202)
def ingest_pings(trip_ref: str, body: PingBatchIn, p: Staff, c: ContainerDep) -> IngestOut:
    r = c.trips.ingest_pings(p.tenant_id, trip_ref, [ping.model_dump() for ping in body.pings])
    return IngestOut(
        received=r.received,
        accepted=r.accepted,
        duplicates=r.duplicates,
        recompute_scheduled=r.recompute_scheduled,
    )


@trips.post("/{trip_ref}/distance:compute", response_model=DistanceOut)
def compute_distance(trip_ref: str, p: Staff, c: ContainerDep) -> DistanceOut:
    return _distance_out(c.trips.compute(p.tenant_id, trip_ref))


@trips.get("/{trip_ref}/distance", response_model=DistanceOut | EmployeeDistanceOut)
def get_distance(trip_ref: str, p: AnyUser, c: ContainerDep) -> DistanceOut | EmployeeDistanceOut:
    _visible_trip(c, p, trip_ref)
    v = c.trips.latest(p.tenant_id, trip_ref)
    if p.role is Role.EMPLOYEE:
        return EmployeeDistanceOut(
            trip_ref=v.trip_ref,
            status=v.status,
            total_km=v.total_km,
            rule_version=v.rule_version,
            rule_totals_m=v.result.get("rule_totals_m", {}),
        )
    return _distance_out(v)


@trips.get("/{trip_ref}/distance/history", response_model=list[DistanceOut])
def distance_history(trip_ref: str, p: Staff, c: ContainerDep) -> list[DistanceOut]:
    return [_distance_out(v) for v in c.trips.history(p.tenant_id, trip_ref)]


@trips.get("/{trip_ref}/map", response_class=HTMLResponse)
def trip_map(trip_ref: str, p: Staff, c: ContainerDep) -> HTMLResponse:
    v, track = c.trips.latest_with_track(p.tenant_id, trip_ref)
    page = mapview.render(v.trip_ref, {**v.result, "track": track})
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@trips.post("/{trip_ref}/invoice:close", response_model=LedgerEntryOut, status_code=201)
def close_invoice(trip_ref: str, p: Admin, c: ContainerDep) -> LedgerEntryOut:
    return _ledger_out(c.billing.close_invoice(p.tenant_id, trip_ref))


@trips.get("/{trip_ref}/ledger", response_model=list[LedgerEntryOut])
def ledger(trip_ref: str, p: Staff, c: ContainerDep) -> list[LedgerEntryOut]:
    return [_ledger_out(e) for e in c.billing.ledger(p.tenant_id, trip_ref)]


# -- ops -----------------------------------------------------------------------------------------


@ops.get("/status", response_model=OpsStatusOut)
def ops_status(p: Staff, c: ContainerDep) -> OpsStatusOut:
    with c.sessions() as s:
        depth, lag = JobRepository(s).queue_stats(wall_clock_ms(), tenant_id=p.tenant_id)
    return OpsStatusOut(
        breaker_state=c.breaker.state.value,
        route_cache_entries=len(c.cache),
        route_cache_hits=c.cache.stats.hits,
        route_cache_misses=c.cache.stats.misses,
        recompute_queue_depth=depth,
        recompute_lag_s=round(lag, 3),
        provider_mode=c.settings.routing_mode,
    )


@ops.post("/chaos/provider", status_code=204)
def chaos_provider(body: ChaosIn, p: Admin, c: ContainerDep) -> Response:
    if not c.settings.chaos_enabled or not isinstance(c.provider, MockRoutingProvider):
        raise Conflict(
            "Chaos switch is only available with CHAOS_ENABLED and the in-process provider. "
            "With the HTTP provider, stop the router container instead.",
            code="CHAOS_DISABLED",
        )
    c.provider.set_down(body.down)
    return Response(status_code=204)


# -- health --------------------------------------------------------------------------------------


@health.get("/health")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@health.get("/ready")
def readiness(c: ContainerDep) -> dict[str, str]:
    with c.sessions() as s:
        s.execute(text("SELECT 1"))
    return {"status": "ready", "breaker": c.breaker.state.value}
