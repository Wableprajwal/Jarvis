---
doc_type: postmortem
date: 2026-02-27
---
# Postmortem: ledger retry storm (2026-02-24)

Severity: SEV1. Duration: 47 minutes (09:12 to 09:59 UTC). Impact: about 18% of card payments failed with HTTP 503.

## Timeline (UTC)

- 09:12 A routine vacuum on `ledger-pg-1` caused lock contention; ledger p99 rose from 90 ms to 4 seconds.
- 09:14 Upstream services timed out and retried immediately, three times each, without jitter.
- 09:19 Ledger request volume reached 4.1x normal. `HalcyonErrorBudgetBurn` paged the primary on-call.
- 09:31 The secondary on-call joined; the team began shedding load at the gateway.
- 09:48 Retries were disabled by feature flag `retries.enabled=false`.
- 09:59 Ledger recovered and error rates returned to baseline.

## Root cause

The contention was a trigger, but the outage was caused by unbounded synchronized client retries amplifying load
on a degraded dependency.

## Action items

1. Adopt a shared retry policy with exponential backoff and full jitter (became ADR-0007). Owner: platform team.
2. Add circuit breakers to every outbound client. Owner: each service team, due 2026-04-30.
3. Schedule vacuum outside business hours and set `autovacuum_vacuum_cost_limit` to 400. Owner: DBA team.
4. Add an alert on ledger request volume above 2x its weekly baseline.
