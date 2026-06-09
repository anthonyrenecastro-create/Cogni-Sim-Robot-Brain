# Environment Topology

## Staging

- Runtime: simulation plus one edge robot lane.
- Staging robot identifier: `CogniSeer-V1`.
- ROS2 distro: Humble.
- OS image baseline: `cogniseer-ros2-humble:2026.06`.
- Purpose: release-candidate validation, latency regression checks, and smoke-release verification.
- Required checks before promotion:
  - `make ci-gate`
  - `make smoke-release`
  - p95 latency <= 10 ms
  - replay determinism gate `max_replay_error <= 1e-4`

## Production

- Runtime: edge robot fleet.
- Fleet identifier: `edge-fleet`.
- ROS2 distro: Humble.
- OS image baseline: `cogniseer-ros2-humble:2026.06`.
- Hardware profile: GPU primary, CPU fallback enabled, RAM class 16 GB.
- Rollout mode: canary.
- Promotion path: staging -> production only; no direct promotion from dev.

## Outstanding Topology Details

- None.
