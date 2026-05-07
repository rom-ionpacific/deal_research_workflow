"""One-shot apply for 004_session_starring.sql. Reads DATABASE_URL from
the deal_cloud_enhancer .env file (same pattern as _check_002_todd.py).

Usage:
    python _apply_004.py
"""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent.parent.parent / "deal_cloud_enhancer" / ".env")

sql = (ROOT / "004_session_starring.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
try:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("MIGRATION APPLIED")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_schema = 'research' AND table_name = 'session'
            ORDER BY ordinal_position
            """
        )
        print("\nresearch.session columns:")
        for r in cur.fetchall():
            print(f"  {r[0]:30s} {r[1]:20s} default={r[2]}")
finally:
    conn.close()
