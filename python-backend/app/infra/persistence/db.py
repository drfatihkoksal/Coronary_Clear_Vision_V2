"""SQLite database connection and schema management."""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS segmentations (
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    engine TEXT,
    confidence REAL,
    centerline_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, frame_index),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS qca_results (
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    pixel_spacing REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, frame_index),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS rws_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    beat_number INTEGER,
    result_json TEXT NOT NULL,
    outlier_method TEXT,
    vessel TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    details_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    is_active INTEGER DEFAULT 1,
    is_verified INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    is_revoked INTEGER DEFAULT 0,
    ip_address TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    is_used INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    is_used INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: sqlite3.Connection | None = None

    def connect(self):
        """Connect to SQLite and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("Database connected: %s", self.db_path)

    def _create_tables(self):
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Database not connected")
        return self._connection

    # Session operations
    def save_session_metadata(self, session_id: str, metadata: dict):
        now = datetime.utcnow().isoformat()
        self.connection.execute(
            "INSERT OR REPLACE INTO sessions (id, created_at, updated_at, metadata_json) VALUES (?, ?, ?, ?)",
            (session_id, now, now, json.dumps(metadata)),
        )
        self.connection.commit()

    def get_session_metadata(self, session_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row and row["metadata_json"]:
            return json.loads(row["metadata_json"])
        return None

    def list_sessions(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT id, created_at, updated_at, metadata_json FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {"id": r["id"], "created_at": r["created_at"], "updated_at": r["updated_at"],
             "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else None}
            for r in rows
        ]

    # QCA operations
    def save_qca_result(self, session_id: str, frame_index: int, metrics: dict, pixel_spacing: float):
        now = datetime.utcnow().isoformat()
        self.connection.execute(
            "INSERT OR REPLACE INTO qca_results (session_id, frame_index, metrics_json, pixel_spacing, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, frame_index, json.dumps(metrics), pixel_spacing, now),
        )
        self.connection.commit()

    def get_qca_results(self, session_id: str) -> dict[int, dict]:
        rows = self.connection.execute(
            "SELECT frame_index, metrics_json FROM qca_results WHERE session_id = ?", (session_id,)
        ).fetchall()
        return {r["frame_index"]: json.loads(r["metrics_json"]) for r in rows}

    # RWS operations
    def save_rws_result(self, session_id: str, result: dict):
        now = datetime.utcnow().isoformat()
        self.connection.execute(
            "INSERT INTO rws_results (session_id, beat_number, result_json, outlier_method, vessel, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, result.get("beat_number"), json.dumps(result),
             result.get("outlier_method"), result.get("vessel"), now),
        )
        self.connection.commit()

    def get_rws_results(self, session_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT result_json FROM rws_results WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]

    # Action log
    def log_action(self, session_id: str, action: str, details: dict | None = None):
        now = datetime.utcnow().isoformat()
        self.connection.execute(
            "INSERT INTO action_log (session_id, timestamp, action, details_json) VALUES (?, ?, ?, ?)",
            (session_id, now, action, json.dumps(details) if details else None),
        )
        self.connection.commit()

    def get_action_log(self, session_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT timestamp, action, details_json FROM action_log WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [
            {"timestamp": r["timestamp"], "action": r["action"],
             "details": json.loads(r["details_json"]) if r["details_json"] else None}
            for r in rows
        ]
