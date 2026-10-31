from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backfill.api.app import create_app
from backfill.container import Container
from backfill.providers.mock import MockRoutingProvider
from backfill.settings import Settings
from backfill.sim.generator import GenConfig, SyntheticTrip, TripGenerator

ROOT = Path(__file__).resolve().parents[2]
PASSWORDS = {
    "acme-admin": "acme-admin-pw",
    "acme-ops": "acme-ops-pw",
    "acme-rider": "acme-rider-pw",
    "globex-admin": "globex-admin-pw",
    "globex-ops": "globex-ops-pw",
}


class FakeClock:
    def __init__(self, start_ms: int = 1_800_000_000_000) -> None:
        self.ms = start_ms

    def __call__(self) -> int:
        return self.ms

    def advance(self, seconds: float) -> None:
        self.ms += int(seconds * 1000)


@dataclass
class Harness:
    client: TestClient
    container: Container
    provider: MockRoutingProvider
    clock: FakeClock
    tokens: dict[str, str]

    def headers(self, user: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens[user]}"}

    def create_trip(self, user: str, trip: SyntheticTrip, trip_ref: str, **overrides: Any) -> dict[str, Any]:
        body = trip_payload(trip, trip_ref, **overrides)
        r = self.client.post("/v1/trips", json=body, headers=self.headers(user))
        assert r.status_code == 201, r.text
        return body

    def send_pings(self, user: str, trip_ref: str, pings: list[dict[str, Any]]) -> dict[str, Any]:
        r = self.client.post(f"/v1/trips/{trip_ref}/pings", json={"pings": pings}, headers=self.headers(user))
        assert r.status_code == 202, r.text
        return r.json()

    def compute(self, user: str, trip_ref: str) -> dict[str, Any]:
        r = self.client.post(f"/v1/trips/{trip_ref}/distance:compute", headers=self.headers(user))
        assert r.status_code == 200, r.text
        return r.json()


def trip_payload(trip: SyntheticTrip, trip_ref: str, **overrides: Any) -> dict[str, Any]:
    ev = trip.evidence

    def pt(p: Any) -> dict[str, float] | None:
        return None if p is None else {"lat": p.lat, "lon": p.lon}

    body: dict[str, Any] = {
        "trip_ref": trip_ref,
        "vendor_id": "V-7",
        "trip_type": ev.trip_type.value,
        "office": pt(ev.office),
        "start_time_ms": ev.start_time_ms,
        "end_time_ms": ev.end_time_ms,
        "start_location": pt(ev.start_location),
        "end_location": pt(ev.end_location),
        "riders": [
            {
                "employee_ref": r.employee_ref,
                "rider_type": r.rider_type.value,
                "status": r.status.value,
                "planned_location": pt(r.planned_location),
                "planned_time_ms": r.planned_time_ms,
                "actual_location": pt(r.actual_location),
                "actual_time_ms": r.actual_time_ms,
                "profile_home": pt(r.profile_home),
                "sign_in_time_ms": r.sign_in_time_ms,
            }
            for r in ev.riders
        ],
    }
    body.update(overrides)
    return body


def ping_payload(trip: SyntheticTrip) -> list[dict[str, Any]]:
    p = trip.evidence.pings
    return [
        {
            "ts_ms": int(p.ts_ms[i]),
            "lat": float(p.lat[i]),
            "lon": float(p.lon[i]),
            "accuracy_m": None if p.accuracy_m[i] != p.accuracy_m[i] else float(p.accuracy_m[i]),
        }
        for i in range(len(p))
    ]


@pytest.fixture
def generator() -> TripGenerator:
    return TripGenerator(config=GenConfig(target_pings=300))


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        users_file=str(ROOT / "deploy" / "users.json"),
        worker_enabled=False,
        chaos_enabled=True,
        breaker_failure_threshold=2,
        breaker_reset_timeout_s=0.2,
        late_ping_debounce_s=1,
        degraded_retry_s=30,
        log_json=True,
        log_level="INFO",
    )
    provider = MockRoutingProvider()
    clock = FakeClock()
    container = Container.build(settings, provider=provider, clock_ms=clock)
    app = create_app(container=container)
    with TestClient(app) as client:
        tokens = {}
        for user, pw in PASSWORDS.items():
            r = client.post("/v1/auth/token", json={"username": user, "password": pw})
            assert r.status_code == 200, r.text
            tokens[user] = r.json()["access_token"]
        yield Harness(client, container, provider, clock, tokens)
    container.db.dispose()
