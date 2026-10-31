from __future__ import annotations

import threading
from collections.abc import Callable

from backfill.observability import metrics as m
from backfill.observability.logging import get_logger
from backfill.persistence.db import SessionFactory
from backfill.persistence.repositories import JobRepository
from backfill.providers.breaker import BreakerState, CircuitBreaker
from backfill.services.trips import TripService

log = get_logger(__name__)


class RecomputeWorker:
    """Drains the recompute queue: late pings and trips billed while the provider was down.

    * Jobs are claimed with a compare-and-set, so several workers (or API replicas) can run.
    * A job whose worker died is reclaimed once its lease expires.
    * While the breaker is OPEN the worker waits instead of producing more DEGRADED results.
    * Failures back off exponentially; after ``max_attempts`` a job is parked as DEAD.
    """

    def __init__(
        self,
        sessions: SessionFactory,
        trips: TripService,
        breaker: CircuitBreaker,
        clock_ms: Callable[[], int],
        worker_id: str = "worker-1",
        poll_s: float = 0.5,
        lease_s: float = 60.0,
        max_attempts: int = 8,
    ) -> None:
        self._sessions = sessions
        self._trips = trips
        self._breaker = breaker
        self._now = clock_ms
        self._id = worker_id
        self._poll = poll_s
        self._lease_ms = int(lease_s * 1000)
        self._max_attempts = max_attempts
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        """Process at most one job. Returns True if a job was processed."""
        now = self._now()
        # Peek with a read session first, so an idle worker never takes the write lock.
        with self._sessions() as s:
            jobs = JobRepository(s)
            depth, lag = jobs.queue_stats(now)
            m.RECOMPUTE_QUEUE_DEPTH.set(depth)
            m.RECOMPUTE_LAG.set(lag)
            if self._breaker.state is BreakerState.OPEN or not jobs.has_due(now, self._lease_ms):
                return False
        with self._sessions.begin() as s:
            job = JobRepository(s).claim_next(self._id, now, self._lease_ms)
            if job is None:
                return False
            job_id, trip_id, reason, attempts = job.id, job.trip_id, job.reason, job.attempts

        try:
            self._trips.compute_by_id(trip_id, reason)
        except Exception as exc:
            backoff_ms = min(300_000, 1000 * 2**attempts)
            with self._sessions.begin() as s:
                outcome = JobRepository(s).fail(
                    job_id, type(exc).__name__, self._now() + backoff_ms, self._max_attempts
                )
            m.RECOMPUTE_JOBS.labels("dead" if outcome == "DEAD" else "retry").inc()
            log.warning("recompute_failed", job_id=job_id, error=type(exc).__name__, outcome=outcome)
            return True

        with self._sessions.begin() as s:
            JobRepository(s).complete(job_id)
        m.RECOMPUTE_JOBS.labels("done").inc()
        log.info("recompute_done", job_id=job_id, trip_id=trip_id, reason=reason)
        return True

    def drain(self, max_jobs: int = 1000) -> int:
        done = 0
        while done < max_jobs and self.run_once():
            done += 1
        return done

    def release_degraded_now(self) -> int:
        """Called when the breaker closes: everything billed on estimates is due immediately."""
        with self._sessions.begin() as s:
            released = JobRepository(s).release_degraded(self._now())
        if released:
            log.info("degraded_jobs_released", count=released)
        return released

    # -- thread lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="recompute-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.run_once()
            except Exception as exc:  # the loop itself must never die
                log.error("worker_loop_error", error=type(exc).__name__)
                busy = False
            if not busy:
                self._stop.wait(self._poll)
