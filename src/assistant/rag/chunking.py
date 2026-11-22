"""Splits page text into overlapping chunks that keep their source and page."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


def split_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Sliding window over the text. Cuts at spaces so words are never split."""
    if overlap >= size // 2:
        raise ValueError("overlap must be less than half of size")
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while True:
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # Move forward to a word boundary so the next chunk doesn't start mid-word.
        if text[start - 1] != " ":
            nxt = text.find(" ", start)
            if nxt != -1 and nxt < end:
                start = nxt + 1
    return chunks


def chunk_pages(
    source: str, pages: list[tuple[int, str]], size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for i, piece in enumerate(split_text(text, size, overlap)):
            chunks.append(Chunk(id=f"{source}:p{page_no}:c{i}", text=piece, source=source, page=page_no))
    return chunks
