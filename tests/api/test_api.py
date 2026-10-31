from __future__ import annotations

import time

import structlog

from backfill.observability.logging import PII_KEYS, scrub_pii
from backfill.sim.generator import TripGenerator
from tests.api.conftest import Harness, ping_payload, trip_payload

# -- authentication ------------------------------------------------------------------------------


def test_wrong_password_and_unknown_user_look_the_same(harness: Harness) -> None:
    a = harness.client.post("/v1/auth/token", json={"username": "acme-admin", "password": "nope"})
    b = harness.client.post("/v1/auth/token", json={"username": "ghost", "password": "nope"})
    assert a.status_code == b.status_code == 401
    assert a.json()["error"]["code"] == b.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_requests_without_or_with_bad_tokens_are_rejected(harness: Harness) -> None:
    assert harness.client.get("/v1/trips/X").status_code == 401
    r = harness.client.get("/v1/trips/X", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "INVALID_TOKEN"


# -- role-based access ---------------------------------------------------------------------------


def test_roles(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(1)
    body = trip_payload(trip, "T-ROLES")
    assert (
        harness.client.post("/v1/trips", json=body, headers=harness.headers("acme-rider")).status_code == 403
    )
    harness.create_trip("acme-ops", trip, "T-ROLES")
    harness.send_pings("acme-ops", "T-ROLES", ping_payload(trip))
    harness.compute("acme-ops", "T-ROLES")
    r = harness.client.post("/v1/trips/T-ROLES/invoice:close", headers=harness.headers("acme-ops"))
    assert r.status_code == 403
    r = harness.client.post("/v1/trips/T-ROLES/invoice:close", headers=harness.headers("acme-admin"))
    assert r.status_code == 201


def test_employee_sees_only_own_trips_and_only_a_summary(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(2)
    body = trip_payload(trip, "T-EMP")
    body["riders"][0]["employee_ref"] = "E00042"  # the acme-rider account
    harness.client.post("/v1/trips", json=body, headers=harness.headers("acme-ops"))
    harness.create_trip("acme-ops", trip, "T-OTHER")
    for ref in ("T-EMP", "T-OTHER"):
        harness.send_pings("acme-ops", ref, ping_payload(trip))
        harness.compute("acme-ops", ref)

    own = harness.client.get("/v1/trips/T-EMP/distance", headers=harness.headers("acme-rider"))
    assert own.status_code == 200
    assert set(own.json()) == {"trip_ref", "status", "total_km", "rule_version", "rule_totals_m"}
    other = harness.client.get("/v1/trips/T-OTHER/distance", headers=harness.headers("acme-rider"))
    assert other.status_code == 404
    assert harness.client.get("/v1/trips/T-EMP/map", headers=harness.headers("acme-rider")).status_code == 403


# -- tenant isolation ----------------------------------------------------------------------------


def test_one_tenant_can_never_read_another_tenants_trip(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(3)
    harness.create_trip("acme-ops", trip, "T-SHARED-REF")
    harness.send_pings("acme-ops", "T-SHARED-REF", ping_payload(trip))
    harness.compute("acme-ops", "T-SHARED-REF")

    for path in ("", "/distance", "/ledger", "/map", "/distance/history"):
        r = harness.client.get(f"/v1/trips/T-SHARED-REF{path}", headers=harness.headers("globex-admin"))
        assert r.status_code == 404, path
    r = harness.client.post(
        "/v1/trips/T-SHARED-REF/pings",
        json={"pings": ping_payload(trip)[:5]},
        headers=harness.headers("globex-ops"),
    )
    assert r.status_code == 404
    # The same reference is free in the other tenant.
    harness.create_trip("globex-ops", trip, "T-SHARED-REF")


# -- ingestion, idempotency, validation ------------------------------------------------------------


def test_duplicate_ping_delivery_and_repeat_compute_are_idempotent(
    harness: Harness, generator: TripGenerator
) -> None:
    trip = generator.generate(4)
    harness.create_trip("acme-ops", trip, "T-IDEM")
    pings = ping_payload(trip)
    first = harness.send_pings("acme-ops", "T-IDEM", pings)
    second = harness.send_pings("acme-ops", "T-IDEM", pings[::-1])
    assert first["accepted"] + first["duplicates"] == len(pings)
    assert second["accepted"] == 0
    a = harness.compute("acme-ops", "T-IDEM")
    b = harness.compute("acme-ops", "T-IDEM")
    assert a["computation_id"] == b["computation_id"]
    assert a["total_m"] == b["total_m"]
    history = harness.client.get(
        "/v1/trips/T-IDEM/distance/history", headers=harness.headers("acme-ops")
    ).json()
    assert len(history) == 1


def test_validation_errors_never_echo_submitted_values(harness: Harness, generator: TripGenerator) -> None:
    body = trip_payload(generator.generate(5), "T-BAD")
    body["office"] = {"lat": 123.4567891, "lon": 77.6}
    r = harness.client.post("/v1/trips", json=body, headers=harness.headers("acme-ops"))
    assert r.status_code == 422
    assert "123.456" not in r.text
    assert r.json()["error"]["details"][0]["field"].endswith("office.lat")


def test_duplicate_trip_ref_is_a_conflict(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(6)
    harness.create_trip("acme-ops", trip, "T-DUP")
    r = harness.client.post(
        "/v1/trips", json=trip_payload(trip, "T-DUP"), headers=harness.headers("acme-ops")
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "TRIP_EXISTS"


# -- billing ledger ------------------------------------------------------------------------------


def test_late_pings_after_billing_produce_a_signed_delta(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(7)
    pings = ping_payload(trip)
    pings.sort(key=lambda p: p["ts_ms"])
    early, late = pings[: len(pings) // 2], pings[len(pings) // 2 :]
    harness.create_trip("acme-ops", trip, "T-LATE")
    harness.send_pings("acme-ops", "T-LATE", early)
    before = harness.compute("acme-ops", "T-LATE")
    closed = harness.client.post("/v1/trips/T-LATE/invoice:close", headers=harness.headers("acme-admin"))
    assert closed.status_code == 201
    assert closed.json()["entry_type"] == "ORIGINAL"

    ingest = harness.send_pings("acme-ops", "T-LATE", late)
    assert ingest["recompute_scheduled"] is True
    assert harness.container.worker.drain() == 0  # debounced: not due yet
    harness.clock.advance(2)
    assert harness.container.worker.drain() == 1

    after = harness.client.get("/v1/trips/T-LATE/distance", headers=harness.headers("acme-ops")).json()
    ledger = harness.client.get("/v1/trips/T-LATE/ledger", headers=harness.headers("acme-ops")).json()
    assert [e["entry_type"] for e in ledger] == ["ORIGINAL", "ADJUSTMENT"]
    delta = after["total_m"] - before["total_m"]
    assert ledger[1]["delta_m"] == delta
    assert ledger[1]["balance_m"] == after["total_m"]
    assert ledger[1]["delta_km"].startswith("+" if delta >= 0 else "-")
    assert ledger[1]["reason"] == "LATE_PINGS"

    # Recomputing the same evidence again must not touch the closed invoice.
    harness.compute("acme-ops", "T-LATE")
    assert len(harness.client.get("/v1/trips/T-LATE/ledger", headers=harness.headers("acme-ops")).json()) == 2


def test_cannot_close_an_invoice_on_stale_distance(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(8)
    pings = ping_payload(trip)
    harness.create_trip("acme-ops", trip, "T-STALE")
    harness.send_pings("acme-ops", "T-STALE", pings[:100])
    harness.compute("acme-ops", "T-STALE")
    harness.send_pings("acme-ops", "T-STALE", pings[100:])
    r = harness.client.post("/v1/trips/T-STALE/invoice:close", headers=harness.headers("acme-admin"))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "DISTANCE_STALE"


# -- failure injection through the API --------------------------------------------------------------


def test_provider_outage_degrades_then_recovers_automatically(
    harness: Harness, generator: TripGenerator
) -> None:
    trips = [generator.generate(10 + i) for i in range(4)]
    for i, t in enumerate(trips):
        harness.create_trip("acme-ops", t, f"T-OUT-{i}")
        harness.send_pings("acme-ops", f"T-OUT-{i}", ping_payload(t))

    r = harness.client.post(
        "/v1/ops/chaos/provider", json={"down": True}, headers=harness.headers("acme-admin")
    )
    assert r.status_code == 204
    results = [harness.compute("acme-ops", f"T-OUT-{i}") for i in range(4)]
    assert all(x["status"] == "DEGRADED" for x in results)
    assert any(li["rule"] == "DEGRADED_AERIAL" for li in results[0]["line_items"])
    status = harness.client.get("/v1/ops/status", headers=harness.headers("acme-ops")).json()
    assert status["breaker_state"] == "OPEN"
    assert status["recompute_queue_depth"] == 4
    close = harness.client.post("/v1/trips/T-OUT-0/invoice:close", headers=harness.headers("acme-admin"))
    assert close.json()["error"]["code"] == "DISTANCE_DEGRADED"

    harness.client.post("/v1/ops/chaos/provider", json={"down": False}, headers=harness.headers("acme-admin"))
    time.sleep(0.25)  # breaker reset timeout
    harness.clock.advance(31)  # degraded jobs come due
    assert harness.container.worker.drain() == 4

    for i in range(4):
        d = harness.client.get(f"/v1/trips/T-OUT-{i}/distance", headers=harness.headers("acme-ops")).json()
        assert d["status"] == "COMPLETE"
        assert d["reason"] == "PROVIDER_DEGRADED"
    assert (
        harness.client.get("/v1/ops/status", headers=harness.headers("acme-ops")).json()["breaker_state"]
        == "CLOSED"
    )


# -- observability and privacy ---------------------------------------------------------------------


def test_logs_never_carry_coordinates(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(20)
    coords = {round(v, 4) for v in list(trip.evidence.pings.lat[:20]) + list(trip.evidence.pings.lon[:20])}
    with structlog.testing.capture_logs() as events:
        harness.create_trip("acme-ops", trip, "T-LOG")
        harness.send_pings("acme-ops", "T-LOG", ping_payload(trip))
        harness.compute("acme-ops", "T-LOG")
    assert events
    for e in events:
        assert not (set(e) & PII_KEYS), e
        for v in e.values():
            if isinstance(v, float):
                assert round(v, 4) not in coords


def test_scrubber_redacts_known_pii_keys() -> None:
    out = scrub_pii(None, "info", {"event": "x", "lat": 12.9, "email": "a@b.c", "trip_id": 4})
    assert out == {"event": "x", "lat": "[redacted]", "email": "[redacted]", "trip_id": 4}


def test_metrics_and_map(harness: Harness, generator: TripGenerator) -> None:
    trip = generator.generate(21)
    harness.create_trip("acme-ops", trip, "T-MAP")
    harness.send_pings("acme-ops", "T-MAP", ping_payload(trip))
    harness.compute("acme-ops", "T-MAP")
    page = harness.client.get("/v1/trips/T-MAP/map", headers=harness.headers("acme-ops"))
    assert page.status_code == 200
    assert "leaflet" in page.text and "SHORT_GAP_AERIAL" in page.text
    metrics = harness.client.get("/metrics").text
    for name in (
        "backfill_breaker_state",
        "backfill_recompute_queue_depth",
        "backfill_pings_rejected_total",
        "backfill_provider_calls_total",
        "backfill_http_request_seconds",
    ):
        assert name in metrics


def test_errors_carry_request_id(harness: Harness) -> None:
    r = harness.client.get(
        "/v1/trips/NOPE", headers={**harness.headers("acme-ops"), "X-Request-ID": "req-123"}
    )
    assert r.status_code == 404
    assert r.json()["error"]["request_id"] == "req-123"
    assert r.headers["X-Request-ID"] == "req-123"
