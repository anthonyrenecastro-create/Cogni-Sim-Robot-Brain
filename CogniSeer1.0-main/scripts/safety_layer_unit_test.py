#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_layer import ActuationSafetyLayer, SafetyConfig


def _assert_close(actual, expected, tol=1e-6, msg=""):
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{msg} expected={expected} actual={actual}")


def test_clamp_behavior():
    layer = ActuationSafetyLayer(
        SafetyConfig(
            motor_min=-1.0,
            motor_max=1.0,
            max_linear_x=2.0,
            max_angular_z=2.0,
            max_linear_delta=5.0,
            max_angular_delta=5.0,
        )
    )
    out = layer.apply(
        motor=np.array([1.8, -3.4, 0.25], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    if out["reason"] != "ok":
        raise AssertionError(f"clamp: unexpected reason {out['reason']}")

    expected_motor = np.array([1.0, -1.0, 0.25], dtype=np.float32)
    if not np.allclose(out["safe_motor"], expected_motor, atol=1e-6):
        raise AssertionError(f"clamp: safe_motor mismatch {out['safe_motor']} != {expected_motor}")

    _assert_close(out["linear_x"], 1.0, msg="clamp linear")
    _assert_close(out["angular_z"], -1.0, msg="clamp angular")


def test_slew_rate_limiting():
    layer = ActuationSafetyLayer(
        SafetyConfig(
            max_linear_x=2.0,
            max_angular_z=2.0,
            max_linear_delta=0.05,
            max_angular_delta=0.10,
        )
    )

    first = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )
    second = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    _assert_close(first["linear_x"], 0.05, msg="slew first linear")
    _assert_close(first["angular_z"], 0.10, msg="slew first angular")
    _assert_close(second["linear_x"], 0.10, msg="slew second linear")
    _assert_close(second["angular_z"], 0.20, msg="slew second angular")


def test_nan_fail_safe():
    layer = ActuationSafetyLayer(
        SafetyConfig(
            max_linear_x=2.0,
            max_angular_z=2.0,
            max_linear_delta=0.20,
            max_angular_delta=0.20,
        )
    )

    _ = layer.apply(
        motor=np.array([0.6, 0.4], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    nan_out = layer.apply(
        motor=np.array([np.nan, 0.2], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    if nan_out["reason"] != "non_finite_model_output":
        raise AssertionError(f"nan fail-safe: unexpected reason {nan_out['reason']}")
    _assert_close(nan_out["linear_x"], 0.0, msg="nan fail-safe linear")
    _assert_close(nan_out["angular_z"], 0.0, msg="nan fail-safe angular")

    post = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    _assert_close(post["linear_x"], 0.20, msg="nan reset linear")
    _assert_close(post["angular_z"], 0.20, msg="nan reset angular")


def test_stale_sensor_stop():
    layer = ActuationSafetyLayer(
        SafetyConfig(
            max_linear_x=2.0,
            max_angular_z=2.0,
            max_linear_delta=0.15,
            max_angular_delta=0.15,
            stop_on_stale_sensor=True,
        )
    )

    _ = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    stale = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=False,
    )

    if stale["reason"] != "stale_sensor_fail_safe_stop":
        raise AssertionError(f"stale stop: unexpected reason {stale['reason']}")
    _assert_close(stale["linear_x"], 0.0, msg="stale stop linear")
    _assert_close(stale["angular_z"], 0.0, msg="stale stop angular")

    post = layer.apply(
        motor=np.array([1.0, 1.0], dtype=np.float32),
        cmd_scale_linear=1.0,
        cmd_scale_angular=1.0,
        sensor_fresh=True,
    )

    _assert_close(post["linear_x"], 0.15, msg="stale reset linear")
    _assert_close(post["angular_z"], 0.15, msg="stale reset angular")


def main():
    tests = [
        ("clamp", test_clamp_behavior),
        ("slew", test_slew_rate_limiting),
        ("nan_fail_safe", test_nan_fail_safe),
        ("stale_sensor_stop", test_stale_sensor_stop),
    ]

    results = []
    for name, fn in tests:
        fn()
        results.append({"test": name, "status": "passed"})

    print(json.dumps({"status": "passed", "results": results}, indent=2))


if __name__ == "__main__":
    main()
