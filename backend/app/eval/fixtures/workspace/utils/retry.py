"""Retry helpers used by the service clients."""
import random


def retry_delay(attempt: int, base: float = 0.2, cap: float = 10.0, rng=random.random) -> float:
    """Full-jitter exponential backoff: random(0, min(cap, base * 2**attempt))."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    return rng() * min(cap, base * (2 ** attempt))


def should_retry(status: int) -> bool:
    return status in (429, 502, 503, 504)
