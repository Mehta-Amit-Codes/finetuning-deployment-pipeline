"""
Gateway storage: request log, routing state, and an event log.

Routing state lives in a tiny key/value table (not process memory) so it
survives restarts and the Streamlit UI and gateway always agree on it.
"""
import os
import sqlite3
import time
from contextlib import contextmanager

from app.config import DB_PATH

DEFAULT_FT_TRAFFIC_PCT = float(os.environ.get("FT_TRAFFIC_PCT", "50"))


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY, ts REAL, arm TEXT, routed_by TEXT, user_id TEXT,
            email TEXT, output TEXT, json_valid INTEGER, latency_ms REAL,
            prompt_tokens INTEGER, completion_tokens INTEGER, judge_score REAL)""")
        c.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, message TEXT)""")
        for k, v in {"ft_traffic_pct": str(DEFAULT_FT_TRAFFIC_PCT), "rollback_active": "0",
                     "demo_degrade": "0", "rollback_reset_ts": "0"}.items():
            c.execute("INSERT OR IGNORE INTO kv (key, value) VALUES (?, ?)", (k, v))


def get_state() -> dict:
    with _conn() as c:
        kv = {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM kv")}
    return {
        "ft_traffic_pct": float(kv.get("ft_traffic_pct", DEFAULT_FT_TRAFFIC_PCT)),
        "rollback_active": kv.get("rollback_active") == "1",
        "demo_degrade": kv.get("demo_degrade") == "1",
        "rollback_reset_ts": float(kv.get("rollback_reset_ts", 0)),
    }


def set_kv(key: str, value) -> None:
    with _conn() as c:
        c.execute("INSERT INTO kv (key, value) VALUES (?, ?) "
                  "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))


def insert_request(**f) -> None:
    f.setdefault("judge_score", None)
    f["ts"] = time.time()
    cols = ["id", "ts", "arm", "routed_by", "user_id", "email", "output", "json_valid",
            "latency_ms", "prompt_tokens", "completion_tokens", "judge_score"]
    with _conn() as c:
        c.execute(f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                  [f.get(k) for k in cols])


def update_judge(request_id: str, score: float) -> None:
    with _conn() as c:
        c.execute("UPDATE requests SET judge_score = ? WHERE id = ?", (score, request_id))


def recent_requests(arm: str, limit: int, since: float = 0.0) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM requests WHERE arm = ? AND ts > ? ORDER BY ts DESC LIMIT ?",
                         (arm, since, limit)).fetchall()
    return [dict(r) for r in rows]


def log_event(kind: str, message: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO events (ts, kind, message) VALUES (?, ?, ?)", (time.time(), kind, message))


def recent_events(limit: int = 30) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))]
