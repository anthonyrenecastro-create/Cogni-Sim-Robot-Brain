import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_layer import ActuationSafetyLayer, SafetyConfig


class TestSafetyLayerUnit(unittest.TestCase):
    def test_clamp_behavior(self):
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

        self.assertEqual(out["reason"], "ok")
        np.testing.assert_allclose(out["safe_motor"], np.array([1.0, -1.0, 0.25], dtype=np.float32))
        self.assertAlmostEqual(out["linear_x"], 1.0, places=6)
        self.assertAlmostEqual(out["angular_z"], -1.0, places=6)

    def test_slew_rate_limiting(self):
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

        self.assertAlmostEqual(first["linear_x"], 0.05, places=6)
        self.assertAlmostEqual(first["angular_z"], 0.10, places=6)
        self.assertAlmostEqual(second["linear_x"], 0.10, places=6)
        self.assertAlmostEqual(second["angular_z"], 0.20, places=6)

    def test_nan_fail_safe(self):
        layer = ActuationSafetyLayer(
            SafetyConfig(
                max_linear_x=2.0,
                max_angular_z=2.0,
                max_linear_delta=0.20,
                max_angular_delta=0.20,
            )
        )

        layer.apply(
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

        self.assertEqual(nan_out["reason"], "non_finite_model_output")
        self.assertAlmostEqual(nan_out["linear_x"], 0.0, places=6)
        self.assertAlmostEqual(nan_out["angular_z"], 0.0, places=6)

        post = layer.apply(
            motor=np.array([1.0, 1.0], dtype=np.float32),
            cmd_scale_linear=1.0,
            cmd_scale_angular=1.0,
            sensor_fresh=True,
        )
        self.assertAlmostEqual(post["linear_x"], 0.20, places=6)
        self.assertAlmostEqual(post["angular_z"], 0.20, places=6)

    def test_stale_sensor_stop(self):
        layer = ActuationSafetyLayer(
            SafetyConfig(
                max_linear_x=2.0,
                max_angular_z=2.0,
                max_linear_delta=0.15,
                max_angular_delta=0.15,
                stop_on_stale_sensor=True,
            )
        )

        layer.apply(
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

        self.assertEqual(stale["reason"], "stale_sensor_fail_safe_stop")
        self.assertAlmostEqual(stale["linear_x"], 0.0, places=6)
        self.assertAlmostEqual(stale["angular_z"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
