"""Read-only SQL over the sample database.

Enforcement is done by SQLite itself, not by looking at the query text:
  1. the file is opened with mode=ro (the OS handle cannot write),
  2. PRAGMA query_only is on,
  3. an authorizer callback denies every action except SELECT, READ, FUNCTION and RECURSIVE,
  4. sqlite3 refuses multiple statements in one execute() call,
  5. a progress handler aborts queries that run too long.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
MAX_ROWS = 200
TIMEOUT_S = 2.0


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_OK if action in ALLOWED else sqlite3.SQLITE_DENY


class SqlError(Exception):
    pass


def run_sql(db_path: Path, query: str) -> dict:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        con.execute("PRAGMA query_only = ON")  # runs before the authorizer is installed
        con.set_authorizer(_authorizer)
        deadline = time.monotonic() + TIMEOUT_S
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = con.execute(query)
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as exc:
            msg = str(exc)
            if "not authorized" in msg or "readonly" in msg.lower():
                msg = f"{msg} (the database is read-only: only SELECT queries are allowed)"
            elif "interrupted" in msg:
                msg = "query took too long and was stopped"
            raise SqlError(msg) from exc
        cols = [d[0] for d in (cur.description or [])]
        truncated = len(rows) > MAX_ROWS
        return {"columns": cols, "rows": [list(r) for r in rows[:MAX_ROWS]], "truncated": truncated}
    finally:
        con.close()
