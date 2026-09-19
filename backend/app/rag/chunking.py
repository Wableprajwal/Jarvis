"""Recursive character splitter (~500-token chunks, ~15% overlap).

Tokens are approximated as chars/4 (no tokenizer dependency; close enough for English prose and
code, and the size only needs to be roughly right). The splitter prefers the coarsest boundary
that fits: paragraph, then line, then sentence, then word, and only hard-cuts as a last resort.
Every chunk after the first begins with a tail of the previous chunk (snapped to a word
boundary) so a fact straddling a boundary is retrievable from either side.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_RATIO = 0.15
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]


@dataclass
class TextChunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)  # {source, date, doc_type}


def _split(text: str, max_len: int, seps: list[str]) -> list[str]:
    """Split into pieces of at most `max_len` chars using the coarsest separator available.
    Separators stay attached to the piece that precedes them, so "".join(pieces) == text."""
    if len(text) <= max_len:
        return [text]
    if not seps:  # nothing left to split on: hard cut
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _split(text, max_len, rest)
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    out: list[str] = []
    for piece in pieces:
        out.extend(_split(piece, max_len, rest) if len(piece) > max_len else [piece])
    return out


def _merge(pieces: list[str], max_len: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > max_len:
            chunks.append(cur)
            cur = ""
        cur += piece
    if cur:
        chunks.append(cur)
    return chunks


def _overlap_tail(prev: str, overlap_chars: int) -> str:
    """Last ~overlap_chars of `prev`, starting at a word boundary."""
    if overlap_chars <= 0 or not prev:
        return ""
    tail = prev[-overlap_chars:]
    if len(tail) < len(prev) and not prev[-overlap_chars - 1].isspace():
        # we cut mid-word: drop the partial leading word
        for i, ch in enumerate(tail):
            if ch.isspace():
                tail = tail[i + 1 :]
                break
        else:
            return ""
    return tail.lstrip()


def chunk_text(
    text: str,
    *,
    source: str = "",
    date: str | None = None,
    doc_type: str = "text",
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[TextChunk]:
    size = chunk_tokens * CHARS_PER_TOKEN
    overlap = int(size * overlap_ratio)
    if not 0 <= overlap < size // 2:
        raise ValueError("overlap_ratio must be in [0, 0.5)")
    text = text.strip()
    if not text:
        return []
    base = _merge(_split(text, size - overlap, _SEPARATORS), size - overlap)
    metadata = {"source": source, "date": date, "doc_type": doc_type}
    chunks: list[TextChunk] = []
    prev = ""
    for raw in base:
        body = raw.strip()
        if not body:
            continue
        tail = _overlap_tail(prev, overlap)
        full = f"{tail} {body}" if tail else body
        chunks.append(TextChunk(text=full, index=len(chunks), metadata=dict(metadata)))
        prev = raw
    return chunks
