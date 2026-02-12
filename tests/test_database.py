"""Tests for database initialization."""

import sqlite3
from pathlib import Path

from src.storage.database import get_connection, initialize_database


class TestInitializeDatabase:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "books" in tables
        assert "query_history" in tables
        assert "settings" in tables

    def test_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)
        initialize_database(db_path)  # Should not raise

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "books" in tables

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "test.db"
        initialize_database(db_path)
        assert db_path.exists()

    def test_books_table_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(books)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "title" in columns
        assert "author" in columns
        assert "source_path" in columns
        assert "chunk_count" in columns
        assert "status" in columns

    def test_query_history_table_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(query_history)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "question" in columns
        assert "answer_text" in columns
        assert "sources_json" in columns
        assert "feedback" in columns


class TestGetConnection:
    def test_returns_connection_with_row_factory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = get_connection(db_path)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = get_connection(db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        assert mode == "wal"
