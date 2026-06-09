# Release Governance

## Required Approvers

- ML Lead
- Robotics Lead

Both approvals are required for production release.

## Release Window

- Monday-Friday
- 09:00-17:00 UTC

Outside this window, release requires explicit incident/exception approval.

## Mandatory Evidence Bundle

A production release ticket must include:

- commit SHA and release tag
- artifact digest(s) from staging validation
- CI run URL with all checks green
- `smoke-release` result
- latency evidence (p95 <= 10 ms)
- replay gate evidence (`max_replay_error <= 1e-4`)
- SBOM attachment
- signature verification result
- exception register status

## Promotion Rule

- Promote only artifacts validated in staging.
- Rebuilding artifacts between staging and production is prohibited.
- Production rollout strategy is canary (no all-at-once promotion by default).
- Platform profile for production releases: GPU primary, CPU fallback enabled, RAM class 16 GB.
- ROS2/OS baseline for releases: ROS2 Humble on `cogniseer-ros2-humble:2026.06`.

## Rollback Rule

- Keep the previous known-good signed artifact set immediately deployable.
- Rollback decision authority: Robotics Lead.
