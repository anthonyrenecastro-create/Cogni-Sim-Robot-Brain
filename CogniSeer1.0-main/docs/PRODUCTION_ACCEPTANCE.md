# Production Acceptance Criteria (Steps 1-6)

This document defines required gates before promoting a model/runtime bundle to production.

## 1) Release Criteria (Blocking)

A release is allowed only if all checks below pass:

- CI status is green for `lint`, `typecheck`, `security`, `test`, and `smoke`.
- `make smoke-release` passes against release-candidate artifacts.
- Artifact metadata includes runtime traceability (`python`, `platform`, lockfile hash).
- Environment is explicitly set (`COGNISEER_ENV=staging|prod`) with matching config overlay.
- No unapproved vulnerability findings from `pip-audit` and `bandit`.
- Any no-fix CVE exception must be explicit and time-bounded (current temporary ignore: `PYSEC-2026-139`).

## Production Thresholds (Approved)

- Inference latency target: p95 <= 10 ms per control tick.
- Promotion replay determinism gate: `max_replay_error <= 1e-4`.
- Safety limits (production):
  - `safety.max_linear_x=0.5`
  - `safety.max_angular_z=1.0`
  - `safety.max_linear_delta=0.10`
  - `safety.max_angular_delta=0.20`
  - `safety.sensor_timeout_sec=0.50`

## 2) Configuration Discipline

- Base config: `config/defaults.json`.
- Environment overlays:
  - `config/environments/dev.json`
  - `config/environments/staging.json`
  - `config/environments/prod.json`
- Runtime environment selection:
  - `COGNISEER_ENV=dev|staging|prod`
  - Optional override path: `COGNISEER_ENV_CONFIG=/path/to/file.json`

## 3) Reproducibility

- Use pinned dependencies from `requirements.lock` and `requirements-dev.lock`.
- Include lockfile fingerprint in exported metadata.
- Keep model manifest + artifact together as deployment unit.

## 4) CI Security and Quality Gates

Minimum required checks:

- Lint: `make lint`
- Typecheck: `make typecheck`
- Security: `make security`
- Tests: `make test`
- Smoke: `make smoke`

Latency and replay thresholds must also be validated in staging before production promotion.

## 5) Release Smoke Policy

Production promotion must use strict smoke test:

```bash
make smoke-release MODEL_PATH=deploy/robot_brain_int8_stateful.pt MANIFEST_PATH=deploy/deployment_manifest.json
```

`smoke-release` does not bootstrap artifacts and fails if candidate artifacts are missing.

Production promotion requires the exact artifact digest tested in staging.

## 6) Failure-Mode Test Expectations

Failure-mode tests must cover at least:

- Invalid JSONL records.
- Shape mismatch in telemetry vectors.
- Invalid split ratios and malformed dataset rows.
- Existing safety clamp/stale/non-finite behavior must remain green.

## Governance Sign-off (Approved)

- Required approvers: ML Lead and Robotics Lead.
- Change window: Monday-Friday, 09:00-17:00 UTC.
- Required platform baseline: ROS2 Humble.
- Staging robot identifier: `CogniSeer-V1`.
- Production hardware class: GPU with optional CPU fallback, RAM class 16 GB.
- Production rollout mode: canary.

