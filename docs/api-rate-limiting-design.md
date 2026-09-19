---
doc_type: design
date: 2026-06-12
---
# Design note: API rate limiting

## Goals

Protect shared backends from a single noisy client while keeping normal integrations unaffected. Limits are
enforced at the edge gateway, keyed per API key.

## Algorithm

We use a token bucket. Each API key gets a bucket that refills at 100 tokens per second with a burst capacity of
200 tokens. A request costs 1 token; batch endpoints cost 1 token per 10 items. State lives in Redis and is
updated by a single Lua script so the check-and-decrement is atomic. A local in-process cache absorbs the
hot path and syncs with Redis every 250 milliseconds.

## Responses

When the bucket is empty the gateway returns HTTP 429 with a `Retry-After` header (whole seconds, minimum 1) and
a JSON body `{"error": "rate_limited"}`. Clients following ADR-0007 will honour `Retry-After`.

## Tiers

| Tier       | Sustained rate | Burst |
|------------|----------------|-------|
| free       | 10 rps         | 20    |
| standard   | 100 rps        | 200   |
| enterprise | 1,000 rps      | 2,000 |

## Failure mode

If Redis is unreachable the gateway fails open with a conservative local limit of 20 rps per key, rather than
rejecting all traffic, and raises the `RateLimiterDegraded` alert.
