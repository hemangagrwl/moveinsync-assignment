"""Concurrency guarantees, exercised with real threads against a real database file."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select

from backfill.container import Container
from backfill.persistence.orm import ComputationRow, LedgerRow, RecomputeJobRow
from backfill.services.worker import RecomputeWorker
from backfill.settings import Settings
from backfill.sim.generator import GenConfig, TripGenerator
from tests.api.conftest import ROOT, FakeClock, ping_payload, trip_payload


def build(tmp_path: Path) -> tuple[Container, FakeClock]:
    clock = FakeClock()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'c.db'}",
        users_file=str(ROOT / "deploy" / "users.json"),
        worker_enabled=False,
        late_ping_debounce_s=0,
        log_level="WARNING",
    )
    return Container.build(settings, clock_ms=clock), clock


def run_threads(n: int, fn: Callable[[int], None]) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def wrapped(i: int) -> None:
        try:
            barrier.wait()
            fn(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def seed(c: Container, gen: TripGenerator, index: int, ref: str) -> None:
    trip = gen.generate(index)
    body = trip_payload(trip, ref)
    c.trips.create_trip(
        "acme",
        ref,
        body["vendor_id"],
        body["trip_type"],
        {k: v for k, v in body.items() if k not in ("trip_ref", "vendor_id", "trip_type")},
    )
    c.trips.ingest_pings("acme", ref, ping_payload(trip))


def test_concurrent_computes_of_one_trip_store_one_result(tmp_path: Path) -> None:
    c, _ = build(tmp_path)
    seed(c, TripGenerator(config=GenConfig(target_pings=300)), 1, "T-RACE")
    ids: list[int] = []
    lock = threading.Lock()

    def compute(_: int) -> None:
        v = c.trips.compute("acme", "T-RACE")
        with lock:
            ids.append(v.computation_id)

    assert run_threads(8, compute) == []
    assert len(set(ids)) == 1
    with c.sessions() as s:
        assert s.scalar(select(func.count()).select_from(ComputationRow)) == 1


def test_competing_workers_process_each_job_exactly_once(tmp_path: Path) -> None:
    c, clock = build(tmp_path)
    gen = TripGenerator(config=GenConfig(target_pings=200))
    refs = [f"T-W{i}" for i in range(12)]
    for i, ref in enumerate(refs):
        seed(c, gen, 30 + i, ref)
        c.trips.compute("acme", ref)
        late = ping_payload(gen.generate(30 + i))[:3]
        for p in late:
            p["ts_ms"] += 1  # new timestamps -> genuinely new pings -> one LATE_PINGS job per trip
        c.trips.ingest_pings("acme", ref, late)
    clock.advance(1)

    workers = [
        RecomputeWorker(c.sessions, c.trips, c.breaker, clock, worker_id=f"w{i}", lease_s=600)
        for i in range(4)
    ]
    processed = [0] * len(workers)

    def drain(i: int) -> None:
        processed[i] = workers[i].drain()

    assert run_threads(len(workers), drain) == []
    assert sum(processed) == len(refs)
    with c.sessions() as s:
        statuses = s.scalars(select(RecomputeJobRow.status)).all()
        assert sorted(statuses) == ["DONE"] * len(refs)
        claimed = s.scalars(select(RecomputeJobRow.attempts)).all()
        assert set(claimed) == {1}


def test_racing_recomputes_on_a_closed_invoice_keep_the_ledger_consistent(tmp_path: Path) -> None:
    c, _ = build(tmp_path)
    gen = TripGenerator(config=GenConfig(target_pings=300))
    trip = gen.generate(5)
    pings = sorted(ping_payload(trip), key=lambda p: p["ts_ms"])
    body = trip_payload(trip, "T-LEDGER")
    c.trips.create_trip(
        "acme",
        "T-LEDGER",
        "V",
        body["trip_type"],
        {k: v for k, v in body.items() if k not in ("trip_ref", "vendor_id", "trip_type")},
    )
    c.trips.ingest_pings("acme", "T-LEDGER", pings[:100])
    c.trips.compute("acme", "T-LEDGER")
    c.billing.close_invoice("acme", "T-LEDGER")

    chunks = [pings[100 + i * 40 : 100 + (i + 1) * 40] for i in range(5)]

    def deliver_and_recompute(i: int) -> None:
        c.trips.ingest_pings("acme", "T-LEDGER", chunks[i])
        c.trips.compute("acme", "T-LEDGER", reason="LATE_PINGS")

    assert run_threads(len(chunks), deliver_and_recompute) == []
    final = c.trips.compute("acme", "T-LEDGER", reason="LATE_PINGS")

    with c.sessions() as s:
        entries = s.scalars(select(LedgerRow).order_by(LedgerRow.id)).all()
    assert entries[0].entry_type == "ORIGINAL"
    assert sum(e.delta_m for e in entries) == entries[-1].balance_m == final.total_m
    assert len({e.computation_id for e in entries}) == len(entries)  # no computation billed twice
