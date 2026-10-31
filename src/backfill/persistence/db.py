from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.session import SessionTransaction

from backfill.persistence.orm import Base

_WRITE = "backfill_write"


class SessionFactory:
    """``factory()`` opens a read session; ``factory.begin()`` opens a write transaction.

    The split matters on SQLite only: writes start with BEGIN IMMEDIATE so two writers
    serialise up front instead of one failing late on a stale snapshot, while reads use a
    plain BEGIN and run concurrently under WAL. On Postgres both use the same engine and row
    locks (SELECT ... FOR UPDATE) do the serialising.
    """

    def __init__(self, engine: Engine) -> None:
        self._read = sessionmaker(engine, expire_on_commit=False)
        self._write = sessionmaker(engine.execution_options(**{_WRITE: True}), expire_on_commit=False)

    def __call__(self) -> Session:
        return self._read()

    def begin(self) -> _WriteContext:
        return _WriteContext(self._write())


class _WriteContext:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._tx: SessionTransaction | None = None

    def __enter__(self) -> Session:
        self._tx = self._session.begin()
        self._tx.__enter__()
        return self._session

    def __exit__(self, *exc: Any) -> None:
        try:
            assert self._tx is not None
            self._tx.__exit__(*exc)
        finally:
            self._session.close()


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        _harden_sqlite(engine)
        return engine
    return create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)


def _harden_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _: Any) -> None:
        dbapi_conn.isolation_level = None  # let SQLAlchemy events issue BEGIN themselves
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Any) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE" if conn.get_execution_options().get(_WRITE) else "BEGIN")


def make_session_factory(engine: Engine) -> SessionFactory:
    return SessionFactory(engine)


def init_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
