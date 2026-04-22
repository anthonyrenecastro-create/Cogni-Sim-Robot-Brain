#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_runtime import EdgeRuntimeBrain
from observability import MetricsTimer, log_event, log_failure


def parse_args():
    parser = argparse.ArgumentParser(
        description="CI smoke test for stateful deployment bundle determinism"
    )
    parser.add_argument("--model-path", default="deploy/robot_brain_int8_stateful.pt")
    parser.add_argument("--manifest-path", default="deploy/deployment_manifest.json")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-replay-error", type=float, default=1e-6)
    return parser.parse_args()


def _build_inputs(steps, input_dim, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(input_dim, dtype=np.float32) for _ in range(steps)]


def _rollout(runtime, inputs, session_id):
    outputs = []
    runtime.start_session(session_id=session_id, reset=True)
    for obs in inputs:
        outputs.append(runtime.step(obs, session_id=session_id))
    return np.stack(outputs, axis=0)


def main():
    args = parse_args()
    timer = MetricsTimer()
    log_event(
        "smoke_start",
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        steps=args.steps,
        seed=args.seed,
    )

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model_path}")
    if not Path(args.manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")

    runtime = EdgeRuntimeBrain(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
    )

    inputs = _build_inputs(args.steps, runtime.input_dim, args.seed)

    run_a = _rollout(runtime, inputs, session_id="smoke-a")
    run_b = _rollout(runtime, inputs, session_id="smoke-a")

    replay_error = float(np.max(np.abs(run_a - run_b)))
    finite = bool(np.isfinite(run_a).all() and np.isfinite(run_b).all())

    result = {
        "steps": args.steps,
        "seed": args.seed,
        "input_dim": runtime.input_dim,
        "finite_outputs": finite,
        "max_replay_error": replay_error,
        "threshold": args.max_replay_error,
        "session_info": runtime.session_info(),
    }
    log_event("smoke_metrics", **result)

    if not finite:
        raise RuntimeError("Smoke test failed: non-finite output detected.")
    if replay_error > args.max_replay_error:
        raise RuntimeError(
            f"Smoke test failed: replay error {replay_error:.8f} exceeds threshold {args.max_replay_error:.8f}."
        )

    log_event("smoke_complete", elapsed_sec=round(timer.elapsed_sec(), 4))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log_failure("smoke_failed", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
