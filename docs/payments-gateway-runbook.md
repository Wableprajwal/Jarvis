---
doc_type: runbook
date: 2026-02-11
---
# Runbook: Halcyon Payments Gateway

The Halcyon gateway (`halcyon-gw`) terminates all card-payment API traffic. It runs as 6 pods in the
`payments-prod` namespace and listens on port 8443 (TLS) with a plaintext health endpoint on port 8081.

## Health and alerts

- Liveness: `GET :8081/healthz` must return 200 within 500 ms. Readiness also checks the ledger database.
- Alert `HalcyonP99High` fires when p99 latency is above 800 ms for 5 minutes. Normal p99 is about 220 ms.
- Alert `HalcyonErrorBudgetBurn` fires when the 5xx rate exceeds 1.5% over 10 minutes.
- Alert `HalcyonQueueDepth` fires when the `settlement-outbox` queue holds more than 12,000 messages.

## Restarting the gateway

1. Check the current rollout: `kubectl -n payments-prod rollout status deploy/halcyon-gw`.
2. Restart one pod at a time: `kubectl -n payments-prod delete pod <name>`. Wait for readiness (about 40 seconds)
   before deleting the next pod. Never restart more than 2 pods at once; the gateway needs at least 4 healthy pods
   to carry peak load of 3,200 requests per second.
3. If more than 3 pods are crash-looping, do NOT keep restarting. Roll back with
   `kubectl -n payments-prod rollout undo deploy/halcyon-gw` and page the payments on-call.

## Ledger database failover

The ledger is a Postgres cluster with one primary (`ledger-pg-1`) and two replicas. Automatic failover promotes a
replica after 30 seconds of primary unavailability. Manual promotion: `patronictl -c /etc/patroni.yml failover`.
After failover, connection pools take up to 90 seconds to drain; expect a burst of `ledger_conn_reset` errors
during that window. This is normal and does not need a page.

## Known failure: settlement outbox stuck

Symptom: `HalcyonQueueDepth` firing and settlement reports late. Cause is usually the `settlement-worker`
holding a stale advisory lock. Fix: restart `settlement-worker` (a single-replica deployment). The outbox drains
at roughly 900 messages per minute, so a 12,000 message backlog takes about 13 minutes to clear.
