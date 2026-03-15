"""SQLite database initialization and connection management."""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.models.book import Book
from src.models.user import User


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
    """Create the database schema and apply any pending migrations.

    Safe to call on an already-initialised database: ``CREATE TABLE IF NOT
    EXISTS`` skips existing tables, and column migrations are guarded by a
    ``PRAGMA table_info`` check so they never run twice.

    Args:
        db_path: Path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)
    try:
        # ---- Create tables (no-op if they already exist) ----
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );

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

        # ---- Column migrations (idempotent) ----
        _add_column_if_missing(conn, "books", "user_id", "TEXT")
        _add_column_if_missing(conn, "query_history", "user_id", "TEXT")

        # ---- Indexes (IF NOT EXISTS is safe to run every time) ----
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_books_user_id
                ON books(user_id);
            CREATE INDEX IF NOT EXISTS idx_query_history_user_id
                ON query_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_query_history_created_at
                ON query_history(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_users_username
                ON users(username);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add a column to a table only when it does not already exist.

    Uses ``PRAGMA table_info`` to inspect the live schema so this is safe
    to run against both fresh and pre-existing databases.

    Args:
        conn: Open SQLite connection.
        table: Table name to alter.
        column: Column name to add.
        col_type: SQL type string (e.g. ``"TEXT"``, ``"INTEGER"``).
    """
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def upsert_book(db_path: str | Path, book: Book) -> None:
    """Insert or update a book record.

    Args:
        db_path: Path to the SQLite database file.
        book: Book instance to persist.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO books (id, title, author, language, source_path, file_format, chunk_count, ingested_at, status, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                author = excluded.author,
                language = excluded.language,
                source_path = excluded.source_path,
                file_format = excluded.file_format,
                chunk_count = excluded.chunk_count,
                ingested_at = excluded.ingested_at,
                status = excluded.status,
                user_id = excluded.user_id
            """,
            (
                book.id,
                book.title,
                book.author,
                book.language,
                book.source_path,
                book.file_format,
                book.chunk_count,
                book.ingested_at.isoformat(),
                book.status,
                book.user_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_book_by_id(db_path: str | Path, book_id: str) -> Book | None:
    """Retrieve a book by its ID.

    Args:
        db_path: Path to the SQLite database file.
        book_id: The book's UUID.

    Returns:
        Book instance if found, None otherwise.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        if row:
            return Book(
                id=row["id"],
                title=row["title"],
                author=row["author"] or "",
                language=row["language"] or "he",
                source_path=row["source_path"],
                file_format=row["file_format"],
                chunk_count=row["chunk_count"] or 0,
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                status=row["status"] or "active",
                user_id=row["user_id"],
            )
        return None
    finally:
        conn.close()


def list_books(db_path: str | Path, user_id: str | None = None) -> list[Book]:
    """Retrieve books visible to a user.

    Args:
        db_path: Path to the SQLite database file.
        user_id: If provided, returns shared books (user_id IS NULL) + user's private books.
                 If None, returns all books.

    Returns:
        List of Book instances.
    """
    conn = get_connection(db_path)
    try:
        if user_id is not None:
            # Return shared books + user's private books
            query = "SELECT * FROM books WHERE user_id IS NULL OR user_id = ? ORDER BY ingested_at DESC"
            rows = conn.execute(query, (user_id,)).fetchall()
        else:
            # Return all books
            rows = conn.execute("SELECT * FROM books ORDER BY ingested_at DESC").fetchall()
        
        return [
            Book(
                id=row["id"],
                title=row["title"],
                author=row["author"] or "",
                language=row["language"] or "he",
                source_path=row["source_path"],
                file_format=row["file_format"],
                chunk_count=row["chunk_count"] or 0,
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                status=row["status"] or "active",
                user_id=row["user_id"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def delete_book(db_path: str | Path, book_id: str) -> bool:
    """Delete a book record.

    Args:
        db_path: Path to the SQLite database file.
        book_id: The book's UUID.

    Returns:
        True if a record was deleted, False otherwise.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_book_status(db_path: str | Path, book_id: str, status: str) -> None:
    """Update the status of a book.

    Args:
        db_path: Path to the SQLite database file.
        book_id: The book's UUID.
        status: New status ("active", "ingesting", "error").
    """
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE books SET status = ? WHERE id = ?", (status, book_id))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# User CRUD Operations
# ---------------------------------------------------------------------------


def create_user(db_path: str | Path, user: User) -> bool:
    """Create a new user record.

    Args:
        db_path: Path to the SQLite database file.
        user: User instance to persist.

    Returns:
        True if user was created successfully, False otherwise.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO users (id, username, email, password_hash, role, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.id,
                user.username,
                user.email,
                user.password_hash,
                user.role,
                user.created_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    finally:
        conn.close()


def get_user_by_username(db_path: str | Path, username: str) -> User | None:
    """Retrieve a user by username.

    Args:
        db_path: Path to the SQLite database file.
        username: The user's username (case-insensitive).

    Returns:
        User instance if found, None otherwise.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE LOWER(username) = LOWER(?)",
            (username,)
        ).fetchone()
        if row:
            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                role=row["role"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            )
        return None
    finally:
        conn.close()


def get_user_by_id(db_path: str | Path, user_id: str) -> User | None:
    """Retrieve a user by ID.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user's UUID.

    Returns:
        User instance if found, None otherwise.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                role=row["role"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            )
        return None
    finally:
        conn.close()


def update_last_login(db_path: str | Path, user_id: str) -> None:
    """Update the last_login timestamp for a user.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user's UUID.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query History Operations
# ---------------------------------------------------------------------------


def save_query(
    db_path: str | Path,
    query_result: "QueryResult",
    user_id: str,
) -> bool:
    """Save a query result to history.

    Args:
        db_path: Path to the SQLite database file.
        query_result: QueryResult instance to save.
        user_id: The user who created this query.

    Returns:
        True if saved successfully, False otherwise.
    """
    import json
    from src.models.query_result import QueryResult
    
    conn = get_connection(db_path)
    try:
        # Serialize sources and answer to JSON
        sources_json = json.dumps([
            {
                "chunk_id": src.chunk.id,
                "book_title": src.chunk.book_title,
                "section_path": src.chunk.section_path,
                "text": src.chunk.text,
                "similarity_score": src.similarity_score,
                "rerank_score": src.rerank_score,
            }
            for src in query_result.sources
        ])
        
        answer_text = query_result.answer.text if query_result.answer else None
        model_used = query_result.answer.model_used if query_result.answer else None
        tokens_used = query_result.answer.tokens_used if query_result.answer else None
        latency_ms = query_result.answer.latency_ms if query_result.answer else None
        
        conn.execute(
            """
            INSERT INTO query_history 
            (id, question, answer_text, sources_json, model_used, tokens_used, latency_ms, feedback, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_result.id,
                query_result.question,
                answer_text,
                sources_json,
                model_used,
                tokens_used,
                latency_ms,
                query_result.feedback,
                query_result.timestamp.isoformat(),
                user_id,
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_query_history(
    db_path: str | Path,
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """Retrieve query history for a user.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user's UUID.
        limit: Maximum number of queries to return (default: 50).

    Returns:
        List of dictionaries with query data (id, question, created_at).
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, question, created_at, answer_text
            FROM query_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "created_at": row["created_at"],
                "has_answer": bool(row["answer_text"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def search_query_history(
    db_path: str | Path,
    user_id: str,
    search_term: str,
    limit: int = 50,
) -> list[dict]:
    """Search query history by question text.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user's UUID.
        search_term: Text to search for in questions.
        limit: Maximum number of results (default: 50).

    Returns:
        List of dictionaries with matching query data.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, question, created_at, answer_text
            FROM query_history
            WHERE user_id = ? AND question LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, f"%{search_term}%", limit),
        ).fetchall()
        
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "created_at": row["created_at"],
                "has_answer": bool(row["answer_text"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_query_by_id(
    db_path: str | Path,
    query_id: str,
    user_id: str,
) -> dict | None:
    """Retrieve a specific query by ID (with ownership check).

    Args:
        db_path: Path to the SQLite database file.
        query_id: The query's UUID.
        user_id: The user's UUID (for ownership verification).

    Returns:
        Dictionary with full query data, or None if not found or not owned by user.
    """
    import json
    
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT * FROM query_history
            WHERE id = ? AND user_id = ?
            """,
            (query_id, user_id),
        ).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "question": row["question"],
            "answer_text": row["answer_text"],
            "sources_json": json.loads(row["sources_json"]) if row["sources_json"] else [],
            "model_used": row["model_used"],
            "tokens_used": row["tokens_used"],
            "latency_ms": row["latency_ms"],
            "feedback": row["feedback"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def delete_query(
    db_path: str | Path,
    query_id: str,
    user_id: str,
) -> bool:
    """Delete a query from history (with ownership check).

    Args:
        db_path: Path to the SQLite database file.
        query_id: The query's UUID.
        user_id: The user's UUID (for ownership verification).

    Returns:
        True if deleted, False if not found or not owned by user.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM query_history WHERE id = ? AND user_id = ?",
            (query_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

