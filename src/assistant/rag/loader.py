"""PDF text extraction, page by page."""
from pathlib import Path

from pypdf import PdfReader

from assistant.core.logging import get_logger

log = get_logger(__name__)


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has extractable text."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    if not pages:
        log.warning("no_text_extracted", extra={"fields": {"file": str(path)}})
    return pages
