from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import httpx

from backfill.core.routing import ProviderError, RouteQuery

_RETRYABLE = {429, 500, 502, 503, 504}


class HttpRoutingProvider:
    """Calls a routing service over HTTP with a hard timeout and a small retry budget.

    Retries cover blips only (one retry by default). A longer outage is the breaker's job.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 2.0,
        retries: int = 1,
        backoff_s: float = 0.1,
        on_retry: Callable[[], None] | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._retries = retries
        self._backoff = backoff_s
        self._on_retry = on_retry or (lambda: None)

    def route_many(self, queries: Sequence[RouteQuery]) -> list[float]:
        payload = {"pairs": [[q.from_lat, q.from_lon, q.to_lat, q.to_lon] for q in queries]}
        last_error = "unknown"
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.post("/v1/route/batch", json=payload)
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
            else:
                if resp.status_code == 200:
                    try:
                        distances = resp.json()["distances_m"]
                        return [float(d) for d in distances]
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ProviderError("malformed provider response") from exc
                if resp.status_code not in _RETRYABLE:
                    raise ProviderError(f"provider rejected request: HTTP {resp.status_code}")
                last_error = f"HTTP {resp.status_code}"
            if attempt < self._retries:
                self._on_retry()
                time.sleep(self._backoff * (2**attempt))
        raise ProviderError(f"provider unavailable: {last_error}")

    def close(self) -> None:
        self._client.close()
