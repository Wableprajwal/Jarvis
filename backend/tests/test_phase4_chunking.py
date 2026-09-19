"""Chunker: size, overlap, boundaries, metadata. Pure functions; no DB, no models."""
import pytest

from app.rag.chunking import CHARS_PER_TOKEN, chunk_text

PARA = (
    "The gateway retries failed calls with exponential backoff and full jitter. "
    "Each attempt waits longer than the last one. The cap is ten seconds. "
)


def _doc(n_paragraphs: int) -> str:
    return "\n\n".join(f"Section {i}. " + PARA * 4 for i in range(n_paragraphs))


def test_short_text_is_one_chunk_with_metadata():
    chunks = chunk_text("Hello world.", source="a.md", date="2026-01-01", doc_type="notes")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].metadata == {"source": "a.md", "date": "2026-01-01", "doc_type": "notes"}
    assert chunks[0].index == 0


def test_empty_and_whitespace_produce_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_budget_and_are_numbered():
    size = 500 * CHARS_PER_TOKEN
    chunks = chunk_text(_doc(30), source="big.md")
    assert len(chunks) > 3
    assert all(len(c.text) <= size for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "big.md" for c in chunks)


def test_chunks_overlap_about_fifteen_percent_on_word_boundaries():
    size = 500 * CHARS_PER_TOKEN
    chunks = chunk_text(_doc(30))
    for prev, cur in zip(chunks, chunks[1:]):
        # the head of `cur` is a tail of `prev`
        shared = 0
        for n in range(min(len(prev.text), len(cur.text)), 0, -1):
            if prev.text.endswith(cur.text[:n]):
                shared = n
                break
        assert 0.05 * size <= shared <= 0.15 * size + 1, shared
        # snapped to a word boundary: the overlap starts at the start of a word in `prev`
        assert prev.text[-shared - 1].isspace()


def test_no_content_is_lost():
    text = _doc(12)
    joined = " ".join(c.text for c in chunk_text(text, chunk_tokens=100))
    for i in range(12):
        assert f"Section {i}." in joined


def test_prefers_paragraph_boundaries():
    paras = [f"Paragraph number {i}. " + "word " * 120 for i in range(6)]  # ~600 chars each
    chunks = chunk_text("\n\n".join(paras), chunk_tokens=200)  # 800-char budget
    # no chunk should start mid-paragraph except through the overlap tail; each base piece
    # ends at a paragraph end, so every paragraph header appears intact
    for i in range(6):
        assert any(f"Paragraph number {i}." in c.text for c in chunks)


def test_oversized_unbroken_text_is_hard_cut():
    chunks = chunk_text("x" * 10_000, chunk_tokens=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", overlap_ratio=0.9)
