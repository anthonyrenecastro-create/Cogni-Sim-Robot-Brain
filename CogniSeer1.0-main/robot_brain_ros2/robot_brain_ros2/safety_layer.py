from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SafetyConfig:
    motor_min: float = -1.0
    motor_max: float = 1.0
    max_linear_x: float = 0.5
    max_angular_z: float = 1.0
    max_linear_delta: float = 0.10
    max_angular_delta: float = 0.20
    sensor_timeout_sec: float = 0.50
    stop_on_stale_sensor: bool = True


class ActuationSafetyLayer:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self._last_linear = 0.0
        self._last_angular = 0.0

    def reset(self):
        self._last_linear = 0.0
        self._last_angular = 0.0

    def apply(
        self,
        motor,
        cmd_scale_linear: float,
        cmd_scale_angular: float,
        sensor_fresh: bool,
    ) -> Dict[str, object]:
        vec = np.asarray(motor, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            vec = np.zeros(2, dtype=np.float32)

        if not np.isfinite(vec).all():
            self.reset()
            return {
                "safe_motor": np.zeros_like(vec, dtype=np.float32),
                "linear_x": 0.0,
                "angular_z": 0.0,
                "reason": "non_finite_model_output",
            }

        vec = np.clip(vec, self.cfg.motor_min, self.cfg.motor_max)

        if self.cfg.stop_on_stale_sensor and not sensor_fresh:
            self.reset()
            return {
                "safe_motor": np.zeros_like(vec, dtype=np.float32),
                "linear_x": 0.0,
                "angular_z": 0.0,
                "reason": "stale_sensor_fail_safe_stop",
            }

        desired_linear = 0.0
        desired_angular = 0.0
        if vec.size >= 1:
            desired_linear = float(vec[0]) * float(cmd_scale_linear)
        if vec.size >= 2:
            desired_angular = float(vec[1]) * float(cmd_scale_angular)

        desired_linear = float(np.clip(desired_linear, -self.cfg.max_linear_x, self.cfg.max_linear_x))
        desired_angular = float(np.clip(desired_angular, -self.cfg.max_angular_z, self.cfg.max_angular_z))

        linear_delta = np.clip(
            desired_linear - self._last_linear,
            -self.cfg.max_linear_delta,
            self.cfg.max_linear_delta,
        )
        angular_delta = np.clip(
            desired_angular - self._last_angular,
            -self.cfg.max_angular_delta,
            self.cfg.max_angular_delta,
        )

        safe_linear = self._last_linear + float(linear_delta)
        safe_angular = self._last_angular + float(angular_delta)

        self._last_linear = safe_linear
        self._last_angular = safe_angular

        return {
            "safe_motor": vec.astype(np.float32),
            "linear_x": safe_linear,
            "angular_z": safe_angular,
            "reason": "ok",
        }
