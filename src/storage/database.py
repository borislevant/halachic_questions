"""SQLite database initialization and connection management."""

import sqlite3
from pathlib import Path


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create a connection to the SQLite database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A sqlite3 Connection with row_factory set to Row.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database(db_path: str | Path) -> None:
    """Create the database schema if it doesn't exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT DEFAULT '',
                language TEXT DEFAULT 'he',
                source_path TEXT NOT NULL,
                file_format TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS query_history (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer_text TEXT,
                sources_json TEXT,
                model_used TEXT,
                tokens_used INTEGER,
                latency_ms INTEGER,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    finally:
        conn.close()
