---
doc_type: adr
date: 2026-03-04
---
# ADR-0007: Retry and backoff policy for service-to-service calls

Status: Accepted. Deciders: platform team. Supersedes ADR-0003.

## Context

During the February incident, clients retried failed calls immediately and in lockstep, tripling the load on an
already degraded ledger service (a retry storm). We need one shared policy so that retries help recovery instead
of preventing it.

## Decision

- Retry only idempotent operations, or operations carrying an `Idempotency-Key` header. POSTs without a key are
  never retried automatically.
- Use exponential backoff with full jitter: delay = random(0, min(cap, base * 2^attempt)).
  Base delay is 200 ms, the multiplier is 2, and the cap is 10 seconds.
- Maximum 5 attempts in total (1 original call plus 4 retries). The overall deadline for the whole retry sequence
  is 30 seconds.
- Retry on connection errors, timeouts, HTTP 429, 502, 503 and 504. Do not retry other 4xx responses.
- If the response carries a `Retry-After` header, honour it instead of the computed delay (but never wait longer
  than the 30 second deadline).
- Every service wraps outbound calls in a circuit breaker. The breaker opens when at least 50% of the last 20
  requests failed. While open it fails fast for 30 seconds, then allows 3 probe requests (half-open). If all
  probes succeed the breaker closes.

## Consequences

Worst-case added latency is bounded at 30 seconds. Full jitter spreads retries and removes the lockstep pattern.
The cost is that callers must handle fast-failing breaker errors (`CircuitOpenError`) explicitly.

## Rejected alternatives

Fixed-interval retry (causes synchronized bursts) and decorrelated jitter (harder to reason about in tests).
