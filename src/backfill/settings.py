from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./backfill.db"
    jwt_secret: str = "dev-only-secret-change-me-0123456789"
    jwt_ttl_s: int = 3600
    users_file: str = "deploy/users.json"

    routing_mode: str = "mock"  # "mock" (in-process) or "http"
    routing_url: str = "http://router:8081"
    routing_timeout_s: float = 2.0
    routing_batch_size: int = 50
    breaker_failure_threshold: int = 5
    breaker_reset_timeout_s: float = 10.0
    route_cache_entries: int = 200_000
    route_cache_ttl_s: float = 24 * 3600

    worker_enabled: bool = True
    worker_poll_s: float = 0.5
    worker_lease_s: float = 60.0
    worker_max_attempts: int = 8
    late_ping_debounce_s: float = 2.0
    degraded_retry_s: float = 15.0

    chaos_enabled: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    worker_id: str = field(default_factory=lambda: f"worker-{os.getpid()}")

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            database_url=_env("DATABASE_URL", d.database_url),
            jwt_secret=_env("JWT_SECRET", d.jwt_secret),
            jwt_ttl_s=int(_env("JWT_TTL_S", str(d.jwt_ttl_s))),
            users_file=_env("USERS_FILE", d.users_file),
            routing_mode=_env("ROUTING_MODE", d.routing_mode),
            routing_url=_env("ROUTING_URL", d.routing_url),
            routing_timeout_s=float(_env("ROUTING_TIMEOUT_S", str(d.routing_timeout_s))),
            routing_batch_size=int(_env("ROUTING_BATCH_SIZE", str(d.routing_batch_size))),
            breaker_failure_threshold=int(
                _env("BREAKER_FAILURE_THRESHOLD", str(d.breaker_failure_threshold))
            ),
            breaker_reset_timeout_s=float(_env("BREAKER_RESET_TIMEOUT_S", str(d.breaker_reset_timeout_s))),
            route_cache_entries=int(_env("ROUTE_CACHE_ENTRIES", str(d.route_cache_entries))),
            route_cache_ttl_s=float(_env("ROUTE_CACHE_TTL_S", str(d.route_cache_ttl_s))),
            worker_enabled=_env_bool("WORKER_ENABLED", d.worker_enabled),
            worker_poll_s=float(_env("WORKER_POLL_S", str(d.worker_poll_s))),
            worker_lease_s=float(_env("WORKER_LEASE_S", str(d.worker_lease_s))),
            worker_max_attempts=int(_env("WORKER_MAX_ATTEMPTS", str(d.worker_max_attempts))),
            late_ping_debounce_s=float(_env("LATE_PING_DEBOUNCE_S", str(d.late_ping_debounce_s))),
            degraded_retry_s=float(_env("DEGRADED_RETRY_S", str(d.degraded_retry_s))),
            chaos_enabled=_env_bool("CHAOS_ENABLED", d.chaos_enabled),
            log_level=_env("LOG_LEVEL", d.log_level),
            log_json=_env_bool("LOG_JSON", d.log_json),
        )
