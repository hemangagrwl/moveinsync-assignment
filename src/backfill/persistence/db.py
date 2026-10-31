from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backfill.persistence.orm import Base

SessionFactory = sessionmaker[Session]


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        _harden_sqlite(engine)
        return engine
    return create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)


def _harden_sqlite(engine: Engine) -> None:
    """WAL for concurrent readers, and BEGIN IMMEDIATE so two writers serialise instead of
    one failing late with a stale snapshot. Postgres gets the same effect from row locks."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _: Any) -> None:
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Any) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def make_session_factory(engine: Engine) -> SessionFactory:
    return sessionmaker(engine, expire_on_commit=False)


def init_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
