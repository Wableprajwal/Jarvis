---
doc_type: checklist
date: 2026-04-08
---
# CI and deploy checklist

## Before merging

- [ ] All checks green: lint (ruff), type check (mypy), unit tests, integration tests.
- [ ] Line coverage on changed files is at least 85%. The gate is enforced in the `coverage` stage of CI.
- [ ] At least 1 approving review, and 2 approvals for anything under `payments/` or `auth/`.
- [ ] Database migrations are backwards compatible: add columns as nullable first, drop columns one release later.

## Pipeline stages

1. `lint` (about 1 minute)
2. `unit` (about 4 minutes, sharded across 8 runners)
3. `integration` (about 9 minutes, uses ephemeral Postgres 16)
4. `build` (container image, tagged with the git SHA)
5. `deploy-staging` (automatic)
6. `deploy-prod` (manual approval by the release manager)

## Production rollout

- Deploy to a canary first: 5% of traffic for 15 minutes.
- Promote automatically only if the canary error rate stays below 0.5% and p95 latency stays within 10% of baseline.
- Then roll out in steps of 25% with a 5 minute pause between steps.
- Rollback: `make rollback SERVICE=<name>` restores the previous image in under 2 minutes. Roll back first,
  investigate later.

## Freeze windows

No production deploys after 14:00 on Fridays, on public holidays, or during the last week of the fiscal
quarter, unless a SEV1 fix is approved by the on-call engineering manager.
