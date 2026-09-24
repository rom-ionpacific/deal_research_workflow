"""One-shot apply for 005_slack_uploads.sql. Reads DATABASE_URL from
the deal_cloud_enhancer .env file (same pattern as _apply_004.py).

Usage:
    python _apply_005.py
"""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent.parent.parent / "deal_cloud_enhancer" / ".env")

sql = (ROOT / "005_slack_uploads.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
try:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("MIGRATION APPLIED")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'research'
              AND table_name = 'slack_uploaded_file'
            ORDER BY ordinal_position
            """
        )
        rows = cur.fetchall()
        print(f"\nresearch.slack_uploaded_file ({len(rows)} columns):")
        for r in rows:
            print(f"  {r[0]:16s} {r[1]:26s} null={r[2]:3s} default={r[3]}")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'research'
              AND tablename = 'slack_uploaded_file'
            ORDER BY indexname
            """
        )
        print("\nindexes:")
        for (name,) in cur.fetchall():
            print(f"  {name}")
finally:
    conn.close()
