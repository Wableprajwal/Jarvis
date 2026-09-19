---
doc_type: policy
date: 2026-05-20
---
# On-call escalation policy

## Rotation

Primary and secondary on-call rotate weekly. The handover happens every Monday at 10:00 UTC. The PagerDuty
schedule is named `platform-primary`. Engineers may swap shifts but must record the swap in PagerDuty at least
24 hours ahead.

## Severity levels

| Level | Definition | Acknowledge within | Update cadence |
|-------|------------|--------------------|----------------|
| SEV1  | Customer-facing outage or data loss | 5 minutes | every 30 minutes |
| SEV2  | Major degradation, workaround exists | 15 minutes | every 60 minutes |
| SEV3  | Minor issue, no customer impact | next business day | daily |

## Escalation ladder for SEV1

1. Page the primary on-call.
2. If there is no acknowledgement after 10 minutes, the system pages the secondary on-call.
3. If there is still none after 20 minutes, page the engineering manager on duty.
4. At 30 minutes without mitigation, the incident commander must notify the VP of Engineering and open a
   customer status page entry.

## Roles

The incident commander (IC) coordinates and does not debug. The communications lead posts updates in
`#inc-war-room` and on the status page. Postmortems are due within 5 business days for every SEV1 and SEV2, and
must be blameless.

## Compensation

On-call engineers receive a flat weekly stipend of 400 EUR plus 1 day of compensatory time off for each SEV1
handled outside working hours.
