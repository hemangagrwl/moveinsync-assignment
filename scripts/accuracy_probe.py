"""Quick accuracy check against synthetic ground truth: python scripts/accuracy_probe.py [trips]"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from backfill.core.config import RuleConfig
from backfill.core.engine import DistanceEngine
from backfill.core.routing import GridSnapper
from backfill.core.rules import RuleSet
from backfill.providers.breaker import CircuitBreaker
from backfill.providers.mock import MockRoutingProvider
from backfill.providers.resilient import ResilientRouter
from backfill.sim.generator import TripGenerator


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    overrides = dict(arg.split("=") for arg in sys.argv[2:])
    trips = [TripGenerator().generate(i) for i in range(n)]
    cfg = RuleConfig()
    cfg = dataclasses.replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in overrides.items()})
    router = ResilientRouter(MockRoutingProvider(), GridSnapper(cfg.cell_size_deg), CircuitBreaker())
    engine = DistanceEngine(RuleSet.default(cfg), router)
    results = [engine.compute(t.evidence) for t in trips]
    pairs = zip(results, trips, strict=True)
    e = np.array([(r.total_m - t.true_distance_m) / t.true_distance_m * 100 for r, t in pairs])
    nv = np.array([(t.naive_distance_m - t.true_distance_m) / t.true_distance_m * 100 for t in trips])
    for name, arr in (("engine", e), ("naive", nv)):
        print(
            f"{name:6s} mean {arr.mean():6.2f}%  median {np.median(arr):6.2f}%  "
            f"p5 {np.percentile(arr, 5):6.2f}%  p95 {np.percentile(arr, 95):6.2f}%  "
            f"MAE {np.abs(arr).mean():6.2f}%  max|err| {np.abs(arr).max():6.1f}%"
        )
    review = sum(r.review_required for r in results)
    print(f"review_required {review}/{n}   worst trips {np.argsort(-np.abs(e))[:5].tolist()}")


if __name__ == "__main__":
    main()
