"""Tiny shared text helpers for lexical components (BM25, the lexical reranker)."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its of on or "
    "our so that the their then there these this to us was we were what when where which who why "
    "will with you your".split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercased word tokens. `snake_case_identifiers` are kept whole AND split into their
    parts, so a query for `max_client_conn` matches both the identifier and prose about it."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        parts = tok.split("_") if "_" in tok else []
        for t in [tok, *parts]:
            if t and not (drop_stopwords and t in STOPWORDS):
                out.append(t)
    return out
