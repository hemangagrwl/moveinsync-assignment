"""Stand-in for a paid routing API, run as its own container.

It behaves like the real thing where it matters for this service: it is slow, rate-limited,
counts every call, and can be switched off (``POST /v1/chaos``) or killed outright
(``docker compose stop router``) to demonstrate the circuit breaker live.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backfill.core.routing import RouteQuery
from backfill.providers.mock import ManhattanRoadNetwork


class BatchIn(BaseModel):
    pairs: list[list[float]] = Field(max_length=500)


class ChaosIn(BaseModel):
    down: bool | None = None
    latency_ms: int | None = Field(None, ge=0, le=30_000)


class _State:
    def __init__(self) -> None:
        self.down = False
        self.latency_ms = int(os.environ.get("ROUTER_LATENCY_MS", "40"))
        self.rate_per_s = float(os.environ.get("ROUTER_RATE_PER_S", "200"))
        self.tokens = self.rate_per_s
        self.last = time.monotonic()
        self.calls = 0
        self.pairs = 0
        self.lock = threading.Lock()

    def take_token(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate_per_s, self.tokens + (now - self.last) * self.rate_per_s)
            self.last = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True


state = _State()
network = ManhattanRoadNetwork()
app = FastAPI(title="Mock Routing Provider")


@app.post("/v1/route/batch")
async def route_batch(body: BatchIn) -> dict[str, list[float]]:
    if state.down:
        raise HTTPException(503, "provider down (chaos)")
    if not state.take_token():
        raise HTTPException(429, "rate limited")
    await asyncio.sleep(state.latency_ms / 1000)
    with state.lock:
        state.calls += 1
        state.pairs += len(body.pairs)
    return {"distances_m": [network.distance_m(RouteQuery(*p)) for p in body.pairs]}


@app.post("/v1/chaos")
def chaos(body: ChaosIn) -> dict[str, object]:
    if body.down is not None:
        state.down = body.down
    if body.latency_ms is not None:
        state.latency_ms = body.latency_ms
    return {"down": state.down, "latency_ms": state.latency_ms}


@app.get("/v1/stats")
def stats() -> dict[str, object]:
    return {"calls": state.calls, "pairs": state.pairs, "down": state.down, "latency_ms": state.latency_ms}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
