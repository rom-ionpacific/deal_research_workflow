"""Postgres connection helper.

We use psycopg2 to match the existing deal_cloud_enhancer codebase. FastAPI
runs sync route handlers in a thread pool, so blocking DB calls inside an
otherwise-async app are fine for V0. If concurrent SSE streams ever
saturate that pool, switch to psycopg (v3) async.

Every connection sets search_path to research,dealcloud,public so unqualified
names resolve to research-owned tables first, with fall-through to dealcloud
for read-only cross-schema queries (organization, document, etc).
"""
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from .config import settings


def _new_conn():
    conn = psycopg2.connect(settings.database_url)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO research, dealcloud, public")
    conn.commit()
    return conn


@contextmanager
def get_conn():
    """Context-managed connection. Commits on clean exit, rollbacks on
    exception. Always closes."""
    conn = _new_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor():
    """Convenience for read paths: connection + dict cursor in one go."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
