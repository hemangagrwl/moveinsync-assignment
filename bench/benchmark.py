"""Engine benchmark: N synthetic trips, accuracy against ground truth, cost and failure.

    python bench/benchmark.py --trips 10000 --pings 1500 --workers 8

Phases
  1. main      every trip computed with a healthy provider; timings, memory, accuracy,
               provider calls with and without the cache.
  2. chaos     the first --chaos-trips trips are recomputed on a fresh engine whose provider
               dies halfway; DEGRADED trips are recomputed after recovery and every total
               must equal phase 1 exactly.

Trip generation is excluded from engine timings. Each worker process has its own route
cache (a shared cache such as Redis would only raise the hit rate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backfill.core.config import RuleConfig  # noqa: E402
from backfill.core.engine import DistanceEngine, TripStatus  # noqa: E402
from backfill.core.routing import GridSnapper  # noqa: E402
from backfill.core.rules import RuleSet  # noqa: E402
from backfill.providers.breaker import CircuitBreaker  # noqa: E402
from backfill.providers.cache import RouteCache  # noqa: E402
from backfill.providers.mock import MockRoutingProvider  # noqa: E402
from backfill.providers.resilient import ResilientRouter  # noqa: E402
from backfill.sim.generator import GenConfig, TripGenerator  # noqa: E402

BATCH = 50


@dataclass
class TripStat:
    index: int
    pings: int
    compute_ms: float
    total_m: int
    true_m: float
    naive_m: float
    routed_pairs: int
    provider_calls: int
    cache_hits: int
    status: str


def _engine(cache: bool, provider: MockRoutingProvider, breaker: CircuitBreaker | None = None) -> DistanceEngine:
    cfg = RuleConfig()
    router = ResilientRouter(
        provider,
        GridSnapper(cfg.cell_size_deg),
        breaker or CircuitBreaker(),
        RouteCache() if cache else None,
        batch_size=BATCH,
    )
    return DistanceEngine(RuleSet.default(cfg), router)


def _peak_rss_mb() -> float:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return peak / 1e6


def run_chunk(args: tuple[list[int], int]) -> tuple[list[TripStat], float, float, int, int]:
    indices, pings = args
    gen = TripGenerator(config=GenConfig(target_pings=pings))
    provider = MockRoutingProvider()
    engine = _engine(True, provider)
    stats: list[TripStat] = []
    gen_s = 0.0
    for i in indices:
        t0 = time.perf_counter()
        trip = gen.generate(i)
        t1 = time.perf_counter()
        r = engine.compute(trip.evidence)
        t2 = time.perf_counter()
        gen_s += t1 - t0
        stats.append(
            TripStat(
                i,
                len(trip.evidence.pings),
                (t2 - t1) * 1000,
                r.total_m,
                trip.true_distance_m,
                trip.naive_distance_m,
                r.routed_pairs,
                r.provider_calls,
                r.cache_hits,
                r.status.value,
            )
        )
    peak = _peak_rss_mb()
    # Recompute pass (late pings, nightly reconciliation, recovery): same gaps asked again.
    before = provider.calls
    for i in indices:
        engine.compute(gen.generate(i).evidence)
    recompute_calls = provider.calls - before
    return stats, gen_s, peak, recompute_calls, len(indices)


def run_chaos(args: tuple[list[int], int]) -> list[tuple[int, int, str, int]]:
    """Provider dies after half the trips; DEGRADED ones are recomputed after recovery."""
    indices, pings = args
    gen = TripGenerator(config=GenConfig(target_pings=pings))
    provider = MockRoutingProvider()
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_s=1, clock=lambda: now[0])
    engine = _engine(True, provider, breaker)
    trips = {i: gen.generate(i).evidence for i in indices}
    first: dict[int, tuple[int, str]] = {}
    for n, i in enumerate(indices):
        if n == len(indices) // 2:
            provider.set_down(True)
        r = engine.compute(trips[i])
        first[i] = (r.total_m, r.status.value)
    provider.set_down(False)
    now[0] += 2
    out = []
    for i in indices:
        total, status = first[i]
        if status == TripStatus.DEGRADED.value:
            r = engine.compute(trips[i])
            out.append((i, r.total_m, r.status.value, 1))
        else:
            out.append((i, total, status, 0))
    return out


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", type=int, default=10_000)
    ap.add_argument("--pings", type=int, default=1_500)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--chaos-trips", type=int, default=2_000)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "benchmark.json"))
    a = ap.parse_args()

    chunks = [list(range(w, a.trips, a.workers)) for w in range(a.workers)]
    print(f"phase 1: {a.trips} trips x ~{a.pings} pings on {a.workers} processes ...", flush=True)
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(run_chunk, [(c, a.pings) for c in chunks]))
    wall = time.perf_counter() - wall0

    stats = [s for r in results for s in r[0]]
    stats.sort(key=lambda s: s.index)
    gen_s = sum(r[1] for r in results)
    peak_worker_mb = max(r[2] for r in results)
    compute_ms = [s.compute_ms for s in stats]
    pings_total = sum(s.pings for s in stats)
    err = [(s.total_m - s.true_m) / s.true_m * 100 for s in stats]
    naive = [(s.naive_m - s.true_m) / s.true_m * 100 for s in stats]
    pairs = sum(s.routed_pairs for s in stats)
    batched_no_cache = sum(math.ceil(s.routed_pairs / BATCH) for s in stats)
    calls = sum(s.provider_calls for s in stats)
    recompute_calls = sum(r[3] for r in results)
    engine_cpu_s = sum(compute_ms) / 1000

    chaos: dict[str, object] = {}
    if a.chaos_trips:
        n = min(a.chaos_trips, a.trips)
        print(f"phase 2: provider fails halfway through {n} trips ...", flush=True)
        cchunks = [list(range(w, n, a.workers)) for w in range(a.workers)]
        # Each worker loses the provider halfway through its own share of the trips.
        with ProcessPoolExecutor(a.workers) as pool:
            cres = [x for r in pool.map(run_chaos, [(c, a.pings) for c in cchunks]) for x in r]
        expected = {s.index: s.total_m for s in stats}
        mismatches = [i for i, total, _, _ in cres if total != expected[i]]
        chaos = {
            "trips": n,
            "degraded_then_recomputed": sum(x[3] for x in cres),
            "still_degraded": sum(1 for x in cres if x[2] != "COMPLETE"),
            "totals_matching_healthy_run": n - len(mismatches),
            "mismatches": len(mismatches),
        }

    report = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "workload": {
            "trips": a.trips,
            "workers": a.workers,
            "pings_total": pings_total,
            "pings_per_trip_mean": round(pings_total / len(stats), 1),
        },
        "time": {
            "wall_s_including_generation": round(wall, 2),
            "generation_cpu_s": round(gen_s, 2),
            "engine_cpu_s": round(engine_cpu_s, 2),
            "engine_trips_per_s_per_core": round(len(stats) / engine_cpu_s, 1),
            "engine_pings_per_s_per_core": round(pings_total / engine_cpu_s),
            "engine_ms_per_trip_p50": round(pct(compute_ms, 50), 3),
            "engine_ms_per_trip_p99": round(pct(compute_ms, 99), 3),
            "engine_ms_per_trip_max": round(max(compute_ms), 3),
        },
        "memory": {"peak_worker_rss_mb": round(peak_worker_mb, 1), "main_rss_mb": round(_peak_rss_mb(), 1)},
        "accuracy_vs_ground_truth_pct": {
            "engine": {
                "median": round(statistics.median(err), 3),
                "mean_abs": round(statistics.fmean(abs(e) for e in err), 3),
                "p5": round(pct(err, 5), 3),
                "p95": round(pct(err, 95), 3),
                "max_abs": round(max(abs(e) for e in err), 3),
            },
            "naive_sum": {
                "median": round(statistics.median(naive), 3),
                "mean_abs": round(statistics.fmean(abs(e) for e in naive), 3),
                "p5": round(pct(naive, 5), 3),
                "p95": round(pct(naive, 95), 3),
                "max_abs": round(max(abs(e) for e in naive), 3),
            },
        },
        "provider_calls_per_trip": {
            "one_call_per_gap_no_cache": round(pairs / len(stats), 3),
            "batched_no_cache": round(batched_no_cache / len(stats), 3),
            "batched_with_cache": round(calls / len(stats), 3),
            "cache_hit_rate_first_pass": round(sum(s.cache_hits for s in stats) / max(1, pairs), 4),
            "recompute_pass_with_cache": round(recompute_calls / len(stats), 3),
            "recompute_pass_without_cache": round(batched_no_cache / len(stats), 3),
        },
        "status": {
            "complete": sum(1 for s in stats if s.status == "COMPLETE"),
            "degraded": sum(1 for s in stats if s.status == "DEGRADED"),
        },
        "chaos_provider_fails_halfway": chaos,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["TripStat", "asdict"]
