"""Smoke test for Neon connectivity.

Skips gracefully when NEON_CONNECTION_STRING is unset so the suite can run in
environments without database credentials.
"""

import os

import pytest


@pytest.mark.skipif(
    not os.getenv("NEON_CONNECTION_STRING"),
    reason="NEON_CONNECTION_STRING not set — skipping live DB test.",
)
def test_db_select_one():
    # Imported lazily: src.config validates env vars at import time, and we only
    # want that to run when the connection string is actually present.
    from src import db

    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()
