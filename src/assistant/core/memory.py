"""Conversation history in SQLite."""
import sqlite3
from contextlib import closing
from pathlib import Path

from assistant.core.llm.base import Message


class ConversationMemory:
    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._run(
            """CREATE TABLE IF NOT EXISTS messages (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   conversation_id TEXT NOT NULL,
                   role TEXT NOT NULL,
                   content TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
               )"""
        )

    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        # Short-lived connection per call: simple and safe across threads.
        with closing(sqlite3.connect(self._db_path)) as conn, conn:  # commits on success
            return conn.execute(sql, params).fetchall()

    def add(self, conversation_id: str, role: str, content: str) -> None:
        self._run(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )

    def history(self, conversation_id: str, limit: int = 20) -> list[Message]:
        """Return the most recent `limit` messages, oldest first."""
        rows = self._run(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        return [Message(role=r, content=c) for r, c in reversed(rows)]
