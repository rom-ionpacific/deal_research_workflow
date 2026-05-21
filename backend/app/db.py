"""Postgres connection helper.

Uses a process-wide psycopg2.pool.ThreadedConnectionPool. FastAPI runs
sync route handlers in a thread pool, so blocking DB calls inside an
otherwise-async app are fine for V0 -- but every connect was paying a
~50-200ms TLS handshake to Neon. The pool keeps live conns warm so
each request reuses one rather than dialing fresh.

Every connection gets `SET search_path TO research, dealcloud, public`
on checkout (NOT just once on creation): Neon's connection pooler can
rotate the backing Postgres process between transactions (see
neon_pooler_search_path_drift), which drops the GUC. The 1 round-trip
on checkout is cheap (~5ms) and immune to that drift.

If a connection errors out mid-transaction we close it on putback so
the pool never hands out a poisoned conn to the next caller.
"""
from contextlib import contextmanager
from threading import Lock

import psycopg2
import psycopg2.extras
from psycopg2 import pool as _pg_pool

from .config import settings


_pool: _pg_pool.ThreadedConnectionPool | None = None
_pool_lock = Lock()


def _get_pool() -> _pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _pg_pool.ThreadedConnectionPool(
                    minconn=settings.db_pool_min,
                    maxconn=settings.db_pool_max,
                    dsn=settings.database_url,
                )
    return _pool


def _prepare(conn) -> None:
    """Apply per-request session settings. Run on every pool checkout so
    Neon's pooler rotating the backend mid-pool-lifetime can't leave us
    with the wrong search_path."""
    with conn.cursor() as cur:
        cur.execute("SET search_path TO research, dealcloud, public")
    conn.commit()


@contextmanager
def get_conn():
    """Context-managed connection from the pool. Commits on clean exit,
    rolls back and discards on exception so the pool never reuses a
    broken conn."""
    pool = _get_pool()
    conn = pool.getconn()
    # If the pool handed back a closed conn (e.g. Neon idle-killed it),
    # discard and ask again. One retry is enough; if we still get closed
    # something else is wrong and the next op will raise.
    if conn.closed:
        pool.putconn(conn, close=True)
        conn = pool.getconn()
    poisoned = False
    try:
        _prepare(conn)
        yield conn
        conn.commit()
    except Exception:
        poisoned = True
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        # close=True drops the conn from the pool entirely; we do that on
        # exception OR if it's actually closed already.
        try:
            pool.putconn(conn, close=poisoned or conn.closed != 0)
        except Exception:
            pass


@contextmanager
def get_cursor():
    """Convenience for read paths: connection + dict cursor in one go."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur


def close_pool() -> None:
    """Close all pooled connections. Called from FastAPI shutdown so
    the worker doesn't leak FDs on reload."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            finally:
                _pool = None
