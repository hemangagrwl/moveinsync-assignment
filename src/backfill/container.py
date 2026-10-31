"""Composition root: the only place that knows which concrete classes are wired together."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import Engine

from backfill.api.security import TokenService, UserStore
from backfill.core.config import RuleConfig
from backfill.core.engine import DistanceEngine
from backfill.core.routing import GridSnapper, RoutingProvider
from backfill.core.rules import RuleSet
from backfill.observability import metrics as m
from backfill.persistence.db import SessionFactory, init_schema, make_engine, make_session_factory
from backfill.providers.breaker import BreakerState, CircuitBreaker
from backfill.providers.cache import RouteCache
from backfill.providers.http import HttpRoutingProvider
from backfill.providers.mock import MockRoutingProvider
from backfill.providers.resilient import ResilientRouter
from backfill.services.billing import BillingService
from backfill.services.trips import TripService
from backfill.services.worker import RecomputeWorker
from backfill.settings import Settings


def wall_clock_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Container:
    settings: Settings
    db: Engine
    sessions: SessionFactory
    provider: RoutingProvider
    breaker: CircuitBreaker
    cache: RouteCache
    router: ResilientRouter
    engine: DistanceEngine
    trips: TripService
    billing: BillingService
    worker: RecomputeWorker
    users: UserStore
    tokens: TokenService

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        provider: RoutingProvider | None = None,
        clock_ms: Callable[[], int] = wall_clock_ms,
        rule_config: RuleConfig | None = None,
    ) -> Container:
        db = make_engine(settings.database_url)
        init_schema(db)
        sessions = make_session_factory(db)

        if provider is None:
            if settings.routing_mode == "http":
                provider = HttpRoutingProvider(
                    settings.routing_url,
                    timeout_s=settings.routing_timeout_s,
                    on_retry=m.PROVIDER_RETRIES.inc,
                )
            else:
                provider = MockRoutingProvider()

        breaker = CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_reset_timeout_s)
        cache = RouteCache(settings.route_cache_entries, settings.route_cache_ttl_s)
        cfg = rule_config or RuleConfig()
        router = ResilientRouter(
            provider,
            GridSnapper(cfg.cell_size_deg),
            breaker,
            cache,
            batch_size=settings.routing_batch_size,
            observer=m.PrometheusRouterObserver(),
        )
        engine = DistanceEngine(RuleSet.default(cfg), router)
        billing = BillingService(sessions, clock_ms)
        trips = TripService(
            sessions,
            engine,
            billing,
            clock_ms,
            late_ping_debounce_s=settings.late_ping_debounce_s,
            degraded_retry_s=settings.degraded_retry_s,
        )
        worker = RecomputeWorker(
            sessions,
            trips,
            breaker,
            clock_ms,
            worker_id=settings.worker_id,
            poll_s=settings.worker_poll_s,
            lease_s=settings.worker_lease_s,
            max_attempts=settings.worker_max_attempts,
        )

        def on_breaker_change(old: BreakerState, new: BreakerState) -> None:
            m.record_breaker(new)
            if new is BreakerState.CLOSED:
                worker.release_degraded_now()

        breaker.add_listener(on_breaker_change)
        m.record_breaker(breaker.state)

        return cls(
            settings=settings,
            db=db,
            sessions=sessions,
            provider=provider,
            breaker=breaker,
            cache=cache,
            router=router,
            engine=engine,
            trips=trips,
            billing=billing,
            worker=worker,
            users=UserStore.from_file(settings.users_file),
            tokens=TokenService(settings.jwt_secret, settings.jwt_ttl_s),
        )
