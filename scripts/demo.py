"""End-to-end demo against a running service (docker compose up, or uvicorn locally).

    python scripts/demo.py --base-url http://localhost:8080

Walks the main flows: ingest a messy trip, compute an explained distance, close the invoice,
deliver late pings and watch a signed adjustment appear, then prove tenant isolation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backfill.sim.generator import GenConfig, TripGenerator

USERS = {
    "ops": ("acme-ops", "acme-ops-pw"),
    "admin": ("acme-admin", "acme-admin-pw"),
    "other": ("globex-admin", "globex-admin-pw"),
}


def token(c: httpx.Client, who: str) -> dict[str, str]:
    u, p = USERS[who]
    r = c.post("/v1/auth/token", json={"username": u, "password": p})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def trip_body(trip: Any, ref: str) -> dict[str, Any]:
    ev = trip.evidence

    def pt(p: Any) -> dict[str, float] | None:
        return None if p is None else {"lat": p.lat, "lon": p.lon}

    return {
        "trip_ref": ref,
        "vendor_id": "VENDOR-7",
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


def pings(trip: Any) -> list[dict[str, Any]]:
    p = trip.evidence.pings
    return [
        {
            "ts_ms": int(p.ts_ms[i]),
            "lat": float(p.lat[i]),
            "lon": float(p.lon[i]),
            "accuracy_m": None if np.isnan(p.accuracy_m[i]) else float(p.accuracy_m[i]),
        }
        for i in range(len(p))
    ]


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--trip", type=int, default=1, help="synthetic trip index")
    a = ap.parse_args()
    ref = f"DEMO-{int(time.time())}"
    trip = TripGenerator(config=GenConfig(target_pings=1500)).generate(a.trip)

    with httpx.Client(base_url=a.base_url, timeout=30) as c:
        ops, admin, other = token(c, "ops"), token(c, "admin"), token(c, "other")

        step("1. Create trip and upload pings (shuffled, duplicated, some held back)")
        c.post("/v1/trips", json=trip_body(trip, ref), headers=ops).raise_for_status()
        all_pings = sorted(pings(trip), key=lambda p: p["ts_ms"])
        held = all_pings[len(all_pings) * 6 // 10 : len(all_pings) * 7 // 10]  # buffered on the device
        sent = [p for p in all_pings if p not in held]
        for chunk in (sent[::2], sent[1::2], sent[:200]):  # out of order, with a duplicate burst
            r = c.post(f"/v1/trips/{ref}/pings", json={"pings": chunk}, headers=ops).json()
            print(
                f"  batch: received {r['received']:>5}  accepted {r['accepted']:>5}"
                f"  duplicates {r['duplicates']}"
            )

        step("2. Compute the billable distance")
        d = c.post(f"/v1/trips/{ref}/distance:compute", headers=ops).json()
        print(f"  total {d['total_km']} km   status {d['status']}   rule {d['rule_version']}")
        print(
            f"  true distance (ground truth) {trip.true_distance_m / 1000:.2f} km,"
            f" naive sum {trip.naive_distance_m / 1000:.2f} km"
        )
        for rule, m in d["rule_totals_m"].items():
            if m:
                print(f"    {rule:<22} {m / 1000:8.2f} km")
        print("  rejected pings:", {k: v for k, v in d["rejected_pings"].items() if v})
        print("  provider:", d["provider"])
        for li in d["line_items"]:
            if li["rule"] != "SHORT_GAP_AERIAL":
                print(f"    {li['rule']:<16} {li['distance_m'] / 1000:6.2f} km  {li['note']}")
        print(f"  map: {a.base_url}/v1/trips/{ref}/map  (needs the ops bearer token)")

        step("3. Close the invoice")
        e = c.post(f"/v1/trips/{ref}/invoice:close", headers=admin).json()
        print(f"  {e['entry_type']}  {e['delta_km']} km  balance {e['balance_km']} km")

        step("4. Late pings arrive after billing")
        r = c.post(f"/v1/trips/{ref}/pings", json={"pings": held}, headers=ops).json()
        print(f"  accepted {r['accepted']} late pings, recompute scheduled: {r['recompute_scheduled']}")
        for _ in range(60):
            ledger = c.get(f"/v1/trips/{ref}/ledger", headers=ops).json()
            if len(ledger) > 1:
                break
            time.sleep(0.5)
        for e in ledger:
            print(
                f"  {e['entry_type']:<10} {e['delta_km']:>8} km"
                f"  balance {e['balance_km']} km  ({e['reason']})"
            )

        step("5. Another tenant cannot see this trip")
        r = c.get(f"/v1/trips/{ref}/distance", headers=other)
        print(f"  globex-admin GET -> HTTP {r.status_code} {r.json()['error']['code']}")

        step("6. Ops status")
        print(" ", c.get("/v1/ops/status", headers=ops).json())


if __name__ == "__main__":
    main()
