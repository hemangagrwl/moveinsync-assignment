from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum

StateListener = Callable[["BreakerState", "BreakerState"], None]


class BreakerState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Classic three-state breaker.

    CLOSED -> OPEN after ``failure_threshold`` consecutive failures.
    OPEN -> HALF_OPEN once ``reset_timeout_s`` has passed; exactly one trial call is let through.
    HALF_OPEN -> CLOSED on a successful trial, back to OPEN on a failed one.

    Listeners are called outside the lock, after the transition, so they may call back in.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False
        self._listeners: list[StateListener] = []

    def add_listener(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    @property
    def state(self) -> BreakerState:
        with self._lock:
            change = self._maybe_half_open()
            state = self._state
        self._notify(change)
        return state

    def allow_request(self) -> bool:
        with self._lock:
            change = self._maybe_half_open()
            if self._state is BreakerState.CLOSED:
                allowed = True
            elif self._state is BreakerState.HALF_OPEN and not self._trial_in_flight:
                self._trial_in_flight = True
                allowed = True
            else:
                allowed = False
        self._notify(change)
        return allowed

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._trial_in_flight = False
            change = self._transition(BreakerState.CLOSED)
        self._notify(change)

    def record_failure(self) -> None:
        with self._lock:
            self._trial_in_flight = False
            change = None
            if self._state is BreakerState.HALF_OPEN:
                change = self._open()
            elif self._state is BreakerState.CLOSED:
                self._failures += 1
                if self._failures >= self._threshold:
                    change = self._open()
        self._notify(change)

    # -- internals (call with lock held) --------------------------------------------------

    def _open(self) -> tuple[BreakerState, BreakerState] | None:
        self._opened_at = self._clock()
        self._failures = 0
        return self._transition(BreakerState.OPEN)

    def _maybe_half_open(self) -> tuple[BreakerState, BreakerState] | None:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self._reset_timeout:
            self._trial_in_flight = False
            return self._transition(BreakerState.HALF_OPEN)
        return None

    def _transition(self, new: BreakerState) -> tuple[BreakerState, BreakerState] | None:
        old = self._state
        if old is new:
            return None
        self._state = new
        return (old, new)

    def _notify(self, change: tuple[BreakerState, BreakerState] | None) -> None:
        if change is None:
            return
        for listener in self._listeners:
            listener(*change)
