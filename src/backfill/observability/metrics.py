"""Counters an on-call engineer needs at 3 AM.

Is the provider down?           backfill_breaker_state, backfill_provider_calls_total{outcome}
Are we billing on estimates?    backfill_computations_total{status="DEGRADED"}
Is recompute keeping up?        backfill_recompute_queue_depth, backfill_recompute_lag_seconds
Is the device data bad?         backfill_pings_rejected_total{reason}
Are invoices moving?            backfill_ledger_adjustments_total, ..._metres_total
Is the API slow?                backfill_http_request_seconds
"""

from __future__ import annotations

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from backfill.providers.breaker import BreakerState

PINGS_INGESTED = Counter("backfill_pings_ingested_total", "Pings accepted into storage")
PINGS_DUPLICATE_ON_INGEST = Counter("backfill_pings_duplicate_ingest_total", "Pings already stored")
PINGS_REJECTED = Counter("backfill_pings_rejected_total", "Pings dropped by cleaning", ["reason"])
SEGMENTS = Counter("backfill_segments_total", "Segments classified, by rule", ["rule"])
SEGMENT_METRES = Counter("backfill_segment_metres_total", "Metres attributed, by rule", ["rule"])
COMPUTATIONS = Counter("backfill_computations_total", "Distance computations", ["status", "reason"])
COMPUTE_SECONDS = Histogram(
    "backfill_compute_seconds",
    "Engine time per trip",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
REVIEW_REQUIRED = Counter("backfill_review_required_total", "Trips flagged for manual review")

PROVIDER_CALLS = Counter("backfill_provider_calls_total", "Routing provider batches", ["outcome"])
PROVIDER_PAIRS = Counter("backfill_provider_pairs_total", "Cell pairs sent to the provider")
PROVIDER_SECONDS = Histogram("backfill_provider_call_seconds", "Provider batch latency")
PROVIDER_RETRIES = Counter("backfill_provider_retries_total", "HTTP retries against the provider")
ROUTE_LOOKUPS = Counter("backfill_route_lookups_total", "Route lookups by source", ["source"])
# With several worker processes each has its own breaker; "max" shows the worst one.
BREAKER_STATE = Gauge("backfill_breaker_state", "0=closed 1=half-open 2=open", multiprocess_mode="max")

RECOMPUTE_QUEUE_DEPTH = Gauge(
    "backfill_recompute_queue_depth", "Pending recompute jobs", multiprocess_mode="max"
)
RECOMPUTE_LAG = Gauge(
    "backfill_recompute_lag_seconds", "Age of the oldest due recompute job", multiprocess_mode="max"
)
RECOMPUTE_JOBS = Counter("backfill_recompute_jobs_total", "Recompute job outcomes", ["outcome"])

LEDGER_ADJUSTMENTS = Counter("backfill_ledger_adjustments_total", "Signed deltas on closed invoices")
LEDGER_ADJUSTMENT_METRES = Counter(
    "backfill_ledger_adjustment_metres_total", "Absolute metres moved by adjustments"
)
INVOICES_CLOSED = Counter("backfill_invoices_closed_total", "Invoices closed")

HTTP_REQUESTS = Counter("backfill_http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_SECONDS = Histogram("backfill_http_request_seconds", "HTTP latency", ["route", "method"])

_BREAKER_VALUE = {BreakerState.CLOSED: 0, BreakerState.HALF_OPEN: 1, BreakerState.OPEN: 2}


def record_breaker(state: BreakerState) -> None:
    BREAKER_STATE.set(_BREAKER_VALUE[state])


class PrometheusRouterObserver:
    def provider_call(self, pairs: int, ok: bool, seconds: float) -> None:
        PROVIDER_CALLS.labels("ok" if ok else "error").inc()
        PROVIDER_PAIRS.inc(pairs)
        PROVIDER_SECONDS.observe(seconds)

    def lookup(self, source: str, count: int) -> None:
        if count:
            ROUTE_LOOKUPS.labels(source).inc(count)


def render_latest() -> tuple[bytes, str]:
    """Metrics for /metrics. Under several uvicorn workers (PROMETHEUS_MULTIPROC_DIR set) each
    process writes its samples to that directory and every scrape aggregates all of them;
    otherwise a scrape would only see whichever process happened to answer."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
