"""Sandboxed file tools. They can only see the folder set in SANDBOX_DIR."""
from pathlib import Path

from assistant.core.config import settings
from assistant.tools.base import Risk, tool

MAX_CHARS = 20_000
MAX_LISTED = 200


def _sandbox_root() -> Path:
    root = Path(settings.sandbox_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve_inside(relative: str) -> Path:
    """Resolve first, then check. This catches '../', absolute paths, and symlinks."""
    root = _sandbox_root()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path is outside the allowed folder.")
    return target


@tool(
    name="read_file",
    description="Read a text file from the sandbox folder. Path is relative to that folder.",
    risk=Risk.READ_ONLY,
)
def read_file(path: str) -> str:
    target = _resolve_inside(path)
    if not target.is_file():
        raise ValueError(f"File not found: {path}")
    with target.open(encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_CHARS + 1)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n[truncated]"
    return text


@tool(
    name="list_files",
    description="List the files in the sandbox folder, so you know which ones read_file can open.",
    risk=Risk.READ_ONLY,
)
def list_files() -> str:
    root = _sandbox_root()
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    if not names:
        return "(the sandbox folder is empty)"
    return "\n".join(names[:MAX_LISTED])
