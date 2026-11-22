"""Notes tools, stored in SQLite."""
import sqlite3
from contextlib import closing
from pathlib import Path

from assistant.core.config import settings
from assistant.tools.base import Risk, tool


def _run(sql: str, params: tuple = ()) -> list[tuple]:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(settings.db_path)) as conn, conn:  # commits on success
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
        return conn.execute(sql, params).fetchall()


@tool(name="save_note", description="Save a note with a title and a body.", risk=Risk.WRITE)
def save_note(title: str, body: str) -> str:
    title, body = title.strip(), body.strip()
    if not title:
        raise ValueError("Title must not be empty.")
    if len(title) > 200 or len(body) > 10_000:
        raise ValueError("Note too long (title max 200 chars, body max 10000).")
    _run("INSERT INTO notes (title, body) VALUES (?, ?)", (title, body))
    return f"Saved note: {title}"


@tool(name="list_notes", description="List the titles of all saved notes, newest first.")
def list_notes() -> str:
    rows = _run("SELECT title FROM notes ORDER BY id DESC LIMIT 100")
    return "\n".join(r[0] for r in rows) or "No notes saved yet."


@tool(
    name="search_notes",
    description="Find saved notes whose title or body contains the given text. Returns full notes.",
)
def search_notes(query: str) -> str:
    like = f"%{query}%"
    rows = _run(
        "SELECT title, body FROM notes WHERE title LIKE ? OR body LIKE ? ORDER BY id DESC LIMIT 10",
        (like, like),
    )
    if not rows:
        return "No matching notes."
    return "\n\n".join(f"## {title}\n{body}" for title, body in rows)
