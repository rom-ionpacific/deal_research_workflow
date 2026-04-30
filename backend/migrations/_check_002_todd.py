"""One-shot helper: syntax-check 002_todd.sql via BEGIN/ROLLBACK,
optionally apply, then run the 5 dossier functions against Moove and
Kairos to spot-check JSONB shapes.

Reads DATABASE_URL from deal_cloud_enhancer/.env so we hit the shared
Neon DB without needing a research-side .env.

Usage:
    python _check_002_todd.py syntax    # BEGIN; <migration>; ROLLBACK;
    python _check_002_todd.py apply     # commit the migration
    python _check_002_todd.py test      # run dossier on Moove + Kairos
"""
import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENHANCER_ENV = ROOT.parent.parent.parent / "deal_cloud_enhancer" / ".env"
SQL_PATH = ROOT / "002_todd.sql"

load_dotenv(ENHANCER_ENV)
DB_URL = os.environ["DATABASE_URL"]


def syntax_check():
    sql = SQL_PATH.read_text(encoding="utf-8")
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(sql)
            cur.execute("ROLLBACK;")
        print("SYNTAX CHECK: OK -- migration parsed and ran inside an aborted txn")
    except Exception as e:
        print(f"SYNTAX CHECK: FAILED\n{type(e).__name__}: {e}")
        raise
    finally:
        conn.close()


def apply():
    sql = SQL_PATH.read_text(encoding="utf-8")
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("APPLY: committed")
    except Exception as e:
        conn.rollback()
        print(f"APPLY: FAILED -- rolled back\n{type(e).__name__}: {e}")
        raise
    finally:
        conn.close()


def find_org_ids(name_substring: str) -> list[tuple[int, str, int | None]]:
    """Return (id, name, superseded_by_org_id) for orgs whose name matches."""
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO dealcloud, public")
            cur.execute(
                """
                SELECT id, name, superseded_by_org_id
                  FROM dealcloud.organization
                 WHERE LOWER(name) LIKE %s
                   AND superseded_by_org_id IS NULL
                 ORDER BY id
                 LIMIT 8
                """,
                (f"%{name_substring.lower()}%",),
            )
            return cur.fetchall()
    finally:
        conn.close()


def run_dossier(label: str, org_ids: list[int]):
    print(f"\n{'='*70}\nDOSSIER: {label} -- org_ids={org_ids}\n{'='*70}")
    if not org_ids:
        print("(no org_ids to test)")
        return
    funcs = [
        ("Q1 portfolio_status", "org_portfolio_status"),
        ("Q2 deal_history", "org_deal_history"),
        ("Q3 ion_contacts", "org_ion_contacts"),
        ("Q4 their_contacts", "org_their_contacts"),
        ("Q5 communication_timeline", "org_communication_timeline"),
    ]
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO dealcloud, public")
            for label_q, func in funcs:
                print(f"\n--- {label_q} ---")
                cur.execute(f"SELECT dealcloud.{func}(%s::int[])", (org_ids,))
                result = cur.fetchone()[0]
                print(json.dumps(result, indent=2, default=str))
    finally:
        conn.close()


def test():
    print("Looking up Moove and Kairos org IDs...")
    moove = find_org_ids("moove")
    kairos = find_org_ids("kairos")
    print(f"\nMoove candidates ({len(moove)}):")
    for r in moove:
        print(f"  id={r[0]:>6}  name={r[1]!r}")
    print(f"\nKairos candidates ({len(kairos)}):")
    for r in kairos:
        print(f"  id={r[0]:>6}  name={r[1]!r}")

    if not moove or not kairos:
        print("\nMissing one or both -- inspect manually.")
        return

    moove_ids = [r[0] for r in moove if "moove" in r[1].lower()][:3]
    kairos_ids = [r[0] for r in kairos if "kairos" in r[1].lower()][:3]

    run_dossier("Moove (top matches)", moove_ids)
    run_dossier("Kairos (top matches)", kairos_ids)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "syntax"
    {"syntax": syntax_check, "apply": apply, "test": test}[cmd]()
