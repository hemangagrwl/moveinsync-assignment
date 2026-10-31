"""Live failure drill against the docker compose stack.

    python scripts/outage_drill.py

Stops the routing provider container, computes trips (they come back DEGRADED), starts the
provider again and waits until every trip has been recomputed to COMPLETE by the background
worker, with no further API calls from us.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import pings, token, trip_body

from backfill.sim.generator import GenConfig, TripGenerator


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--trips", type=int, default=3)
    ap.add_argument("--timeout-s", type=float, default=90)
    a = ap.parse_args()
    gen = TripGenerator(config=GenConfig(target_pings=800))

    with httpx.Client(base_url=a.base_url, timeout=60) as c:
        h = token(c, "ops")
        refs = []
        tag = int(time.time())
        for i in range(50):
            if len(refs) == a.trips:
                break
            trip = gen.generate(500 + i)
            ps = pings(trip)
            if not ps:
                continue
            ref = f"DRILL-{tag}-{i}"
            c.post("/v1/trips", json=trip_body(trip, ref), headers=h).raise_for_status()
            c.post(f"/v1/trips/{ref}/pings", json={"pings": ps}, headers=h).raise_for_status()
            refs.append(ref)

        print("stopping the routing provider ...")
        compose("stop", "router")
        degraded = 0
        for ref in refs:
            d = c.post(f"/v1/trips/{ref}/distance:compute", headers=h).json()
            print(f"  {ref}: {d['status']:<9} {d['total_km']} km")
            degraded += d["status"] == "DEGRADED"
        print("  ops:", c.get("/v1/ops/status", headers=h).json())
        if degraded == 0:
            sys.exit("expected DEGRADED results while the provider is down")

        print("starting the routing provider again ...")
        compose("start", "router")
        deadline = time.time() + a.timeout_s
        while time.time() < deadline:
            states = [c.get(f"/v1/trips/{r}/distance", headers=h).json() for r in refs]
            if all(s["status"] == "COMPLETE" for s in states):
                for r, s in zip(refs, states, strict=True):
                    print(f"  {r}: {s['status']:<9} {s['total_km']} km  (reason {s['reason']})")
                print("recovered without intervention")
                return
            time.sleep(2)
        sys.exit("trips did not recover in time")


if __name__ == "__main__":
    main()
