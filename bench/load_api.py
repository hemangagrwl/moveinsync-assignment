"""HTTP load test: throughput and p50/p95/p99 per endpoint against a running service.

    python bench/load_api.py --base-url http://127.0.0.1:8080 --trips 30 --concurrency 32 --seconds 30

Seeds trips first (not measured), then runs a mixed workload:
  70% GET  /v1/trips/{ref}/distance            (read path)
  20% POST /v1/trips/{ref}/distance:compute    (engine + idempotent write)
  10% POST /v1/trips/{ref}/pings               (late pings -> debounced recompute job)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo import pings, token, trip_body  # noqa: E402

from backfill.sim.generator import GenConfig, TripGenerator  # noqa: E402


def _worker_process(
    args: tuple[str, dict[str, str], list[str], dict[str, list[dict]], float, int, int],
) -> dict:
    """One load-generator process: ``threads`` blocking clients, each with its own connection.
    Several processes are used because a single Python process cannot generate enough load."""
    base_url, headers, refs, late, stop, threads, seed, p_read, p_compute = args
    lat: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    lock = threading.Lock()

    def run(tid: int) -> None:
        rng = random.Random(seed * 1000 + tid)
        with httpx.Client(base_url=base_url, timeout=60) as c:
            while time.time() < stop:
                ref = rng.choice(refs)
                roll = rng.random()
                t0 = time.perf_counter()
                if roll < p_read:
                    name, r = "GET distance", c.get(f"/v1/trips/{ref}/distance", headers=headers)
                elif roll < p_read + p_compute:
                    name, r = "POST compute", c.post(f"/v1/trips/{ref}/distance:compute", headers=headers)
                else:
                    batch = [
                        dict(p, ts_ms=p["ts_ms"] + rng.randint(1, 10**6)) for p in rng.sample(late[ref], 5)
                    ]
                    name, r = (
                        "POST pings",
                        c.post(f"/v1/trips/{ref}/pings", json={"pings": batch}, headers=headers),
                    )
                elapsed = time.perf_counter() - t0
                with lock:
                    lat[name].append(elapsed)
                    if r.status_code >= 400:
                        errors[name] += 1

    pool = [threading.Thread(target=run, args=(t,)) for t in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return {"lat": dict(lat), "errors": dict(errors)}


def run(a: argparse.Namespace) -> dict:
    gen = TripGenerator(config=GenConfig(target_pings=a.pings))
    tag = int(time.time())
    with httpx.Client(base_url=a.base_url, timeout=60) as sc:
        headers = token(sc, "ops")
        refs, late = [], {}
        for i in range(a.trips):
            trip = gen.generate(1000 + i)
            ps = pings(trip)
            if not ps:
                continue
            ref = f"LOAD-{tag}-{i}"
            sc.post("/v1/trips", json=trip_body(trip, ref), headers=headers).raise_for_status()
            sc.post(f"/v1/trips/{ref}/pings", json={"pings": ps}, headers=headers).raise_for_status()
            sc.post(f"/v1/trips/{ref}/distance:compute", headers=headers).raise_for_status()
            refs.append(ref)
            late[ref] = random.Random(i).sample(ps, min(50, len(ps)))

    threads = max(1, a.concurrency // a.processes)
    stop = time.time() + a.seconds
    started = time.perf_counter()
    p_read, p_compute = {"mixed": (0.7, 0.2), "read": (1.0, 0.0), "compute": (0.0, 1.0)}[a.mix]
    jobs = [
        (a.base_url, headers, refs, late, stop, threads, p, p_read, p_compute) for p in range(a.processes)
    ]
    with ProcessPoolExecutor(a.processes) as pool:
        parts = list(pool.map(_worker_process, jobs))
    elapsed = time.perf_counter() - started

    lat: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for part in parts:
        for k, v in part["lat"].items():
            lat[k].extend(v)
        for k, v in part["errors"].items():
            errors[k] += v
    total = sum(len(v) for v in lat.values())
    return {
        "mix": a.mix,
        "concurrency": threads * a.processes,
        "seconds": round(elapsed, 1),
        "requests": total,
        "throughput_rps": round(total / elapsed, 1),
        "endpoints": {
            name: {
                "count": len(v),
                "errors": errors[name],
                "p50_ms": round(float(np.percentile(v, 50)) * 1000, 1),
                "p95_ms": round(float(np.percentile(v, 95)) * 1000, 1),
                "p99_ms": round(float(np.percentile(v, 99)) * 1000, 1),
                "mean_ms": round(statistics.fmean(v) * 1000, 1),
            }
            for name, v in sorted(lat.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--trips", type=int, default=40)
    ap.add_argument("--pings", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--mix", choices=["mixed", "read", "compute"], default="mixed")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "load_api.json"))
    a = ap.parse_args()
    report = run(a)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
