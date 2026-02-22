import tempfile
from pathlib import Path
from app.infra.persistence.db import Database


def test_database_connect_and_create_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.connect()
        # Tables should be created
        tables = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "sessions" in table_names
        assert "qca_results" in table_names
        assert "rws_results" in table_names
        assert "action_log" in table_names
        db.close()


def test_save_and_get_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.connect()
        db.save_session_metadata("test-1", {"study": "test"})
        meta = db.get_session_metadata("test-1")
        assert meta == {"study": "test"}
        db.close()


def test_list_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.connect()
        db.save_session_metadata("s1", {"a": 1})
        db.save_session_metadata("s2", {"b": 2})
        sessions = db.list_sessions()
        assert len(sessions) == 2
        db.close()


def test_action_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.connect()
        db.save_session_metadata("s1", {})
        db.log_action("s1", "upload", {"file": "test.dcm"})
        db.log_action("s1", "segment", {"frame": 0})
        log = db.get_action_log("s1")
        assert len(log) == 2
        assert log[0]["action"] == "upload"
        db.close()
