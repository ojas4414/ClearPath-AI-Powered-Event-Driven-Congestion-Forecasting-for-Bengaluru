"""
ClearPath :: db.py
==================
Operational data store for the LIVE side of ClearPath — reported incidents and post-event
feedback. Deliberately NOT the 8k static training CSV (that stays a file; a DB adds nothing
there). This is the dynamic, append-only operational log a real control room generates.

WHY a graceful two-backend design:
- In production / Docker we want a real RDBMS, so this uses **Postgres** when DATABASE_URL is set.
- For a zero-infra local run (`python main.py`) it falls back to a stdlib **SQLite** file, so the
  demo never depends on a database being up. Same API, two backends, picked by one env var.
"""

import os
import datetime as dt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_PG = DATABASE_URL.startswith("postgres")

if USE_PG:
    import psycopg2
    import psycopg2.extras

    def _conn():
        return psycopg2.connect(DATABASE_URL)
    PH = "%s"           # Postgres parameter placeholder
    _SERIAL = "SERIAL PRIMARY KEY"
else:
    import sqlite3
    SQLITE_PATH = os.path.join(BASE_DIR, "clearpath.db")

    def _conn():
        c = sqlite3.connect(SQLITE_PATH)
        c.row_factory = sqlite3.Row
        return c
    PH = "?"            # SQLite parameter placeholder
    _SERIAL = "INTEGER PRIMARY KEY AUTOINCREMENT"


def backend():
    """Human-readable name of the active backend (shown on /health for transparency)."""
    return "postgres" if USE_PG else "sqlite"


def init_db():
    """Create the operational tables if they don't exist. Safe to call on every startup."""
    with _conn() as c:
        cur = c.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS live_incidents (
                id {_SERIAL},
                ts            TIMESTAMP,
                corridor      TEXT,
                event_cause   TEXT,
                hour          INTEGER,
                severity      TEXT,
                impact_score  REAL,
                officers      INTEGER,
                lat           REAL,
                lon           REAL,
                reason        TEXT
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id {_SERIAL},
                ts            TIMESTAMP,
                predicted_min REAL,
                actual_min    REAL,
                residual_min  REAL
            )""")
        c.commit()


def insert_incident(d):
    """Persist one reported live incident; returns nothing (id is auto)."""
    with _conn() as c:
        cur = c.cursor()
        cur.execute(
            f"""INSERT INTO live_incidents
                (ts, corridor, event_cause, hour, severity, impact_score, officers, lat, lon, reason)
                VALUES ({PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH})""",
            (dt.datetime.now(), d["corridor"], d["event_cause"], d["hour"], d["severity"],
             d["impact_score"], d["officers"], d.get("lat"), d.get("lon"), d.get("reason", "")),
        )
        c.commit()


def recent_incidents(limit=20):
    """Return the most recent reported incidents (newest first) as a list of dicts."""
    with _conn() as c:
        cur = c.cursor()
        cur.execute(
            f"""SELECT corridor, event_cause, hour, severity, impact_score, officers, lat, lon, ts
                FROM live_incidents ORDER BY id DESC LIMIT {PH}""", (limit,))
        cols = [x[0] for x in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        if isinstance(r.get("ts"), (dt.datetime, dt.date)):
            r["ts"] = r["ts"].isoformat()
    return rows


def insert_feedback(predicted_min, actual_min, residual_min):
    """Persist one post-event feedback outcome (the live half of the learning loop)."""
    with _conn() as c:
        cur = c.cursor()
        cur.execute(
            f"""INSERT INTO feedback_log (ts, predicted_min, actual_min, residual_min)
                VALUES ({PH},{PH},{PH},{PH})""",
            (dt.datetime.now(), predicted_min, actual_min, residual_min))
        c.commit()


def feedback_count():
    """How many live feedback outcomes have been recorded (for the learning panel)."""
    with _conn() as c:
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback_log")
        return int(cur.fetchone()[0])
