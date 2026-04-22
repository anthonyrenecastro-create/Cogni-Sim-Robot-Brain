import math
import numpy as np
from dataclasses import dataclass


# ============================================================
# Utility
# ============================================================

def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

def safe_log_abs(x, eps=1e-6):
    return np.log(np.abs(x) + eps)

def softclip(x, limit=5.0):
    return np.clip(x, -limit, limit)


# ============================================================
# Edge-friendly 2D convolution helpers
# ============================================================

def laplacian_2d(x):
    """
    5-point stencil Laplacian with wrap-around boundaries.
    """
    return (
        np.roll(x, 1, axis=0) +
        np.roll(x, -1, axis=0) +
        np.roll(x, 1, axis=1) +
        np.roll(x, -1, axis=1) -
        4.0 * x
    )

def biharmonic_2d(x):
    return laplacian_2d(laplacian_2d(x))


# ============================================================
# Config
# ============================================================

@dataclass
class BrainConfig:
    grid_h: int = 16
    grid_w: int = 16
    input_dim: int = 32
    hidden_dim: int = 64
    motor_dim: int = 8
    dt: float = 0.02
    coupling_eps: float = 0.05
    resonance_omega0: float = 14.0  # compressed surrogate, not literal 1e14
    spike_decay: float = 0.92
    memory_decay: float = 0.98
    action_decay: float = 0.90
    seed: int = 7


@dataclass(frozen=True)
class SensorChannelShape:
    imu: int = 6
    wheel_odometry: int = 4
    camera_embeddings: int = 12
    microphone_features: int = 4
    proximity_lidar: int = 4
    battery_thermal: int = 2

    @property
    def total_dim(self):
        return (
            self.imu
            + self.wheel_odometry
            + self.camera_embeddings
            + self.microphone_features
            + self.proximity_lidar
            + self.battery_thermal
        )


@dataclass
class SensorChannels:
    imu: np.ndarray
    wheel_odometry: np.ndarray
    camera_embeddings: np.ndarray
    microphone_features: np.ndarray
    proximity_lidar: np.ndarray
    battery_thermal: np.ndarray


@dataclass(frozen=True)
class ControlCommandShape:
    wheel_velocities: int = 2
    arm_joint_deltas: int = 2
    gaze_servo_control: int = 2
    speech_act_selection: int = 1
    planner_state_transitions: int = 1

    @property
    def total_dim(self):
        return (
            self.wheel_velocities
            + self.arm_joint_deltas
            + self.gaze_servo_control
            + self.speech_act_selection
            + self.planner_state_transitions
        )


# ============================================================
# Lightweight linear layer
# ============================================================

class Linear:
    def __init__(self, in_dim, out_dim, rng):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.w = rng.standard_normal((in_dim, out_dim)) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x):
        return x @ self.w + self.b


# ============================================================
# Resonant coupling tensor
# ============================================================

class ResonantCoupling:
    """
    Weak rank-3 coupling tensor surrogate.
    Instead of storing full T[n,m,p,h,w], we factorize it for edge use.
    """
    def __init__(self, num_fields, rng):
        self.num_fields = num_fields
        self.A = rng.standard_normal((num_fields, num_fields, num_fields)) * 0.05
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_fields, num_fields, num_fields))

    def coupling_step(self, phi, t, omega0):
        """
        phi: [10, H, W]
        returns coupling contribution [10, H, W]
        """
        n_fields = phi.shape[0]
        out = np.zeros_like(phi)

        resonance = np.sin(omega0 * t * 0.01 + self.phase)

        # edge-friendly approximation of ε Σ_{m != p} T_nmp * phi_m * mean(phi_p)
        phi_means = phi.mean(axis=(1, 2))
        for n in range(n_fields):
            acc = 0.0
            for m in range(n_fields):
                for p in range(n_fields):
                    if m == p:
                        continue
                    acc += self.A[n, m, p] * resonance[n, m, p] * phi[m] * phi_means[p]
            out[n] = acc
        return out


# ============================================================
# Quadra Mind modules
# ============================================================

class PerceptionModule:
    def __init__(self, cfg, rng):
        self.enc1 = Linear(cfg.input_dim, cfg.hidden_dim, rng)
        self.enc2 = Linear(cfg.hidden_dim, cfg.hidden_dim, rng)

    def __call__(self, obs):
        x = relu(self.enc1(obs))
        x = tanh(self.enc2(x))
        return x


class MemoryModule:
    def __init__(self, cfg, rng):
        self.mix = Linear(cfg.hidden_dim, cfg.hidden_dim, rng)
        self.state = np.zeros(cfg.hidden_dim, dtype=np.float32)
        self.decay = cfg.memory_decay

    def __call__(self, x, phi5_gain):
        candidate = tanh(self.mix(x))
        keep = np.clip(self.decay + 0.05 * np.mean(phi5_gain), 0.80, 0.999)
        self.state = keep * self.state + (1.0 - keep) * candidate
        return self.state.copy()


class ReasoningModule:
    def __init__(self, cfg, rng):
        self.r1 = Linear(cfg.hidden_dim * 2, cfg.hidden_dim, rng)
        self.r2 = Linear(cfg.hidden_dim, cfg.hidden_dim, rng)

    def __call__(self, percept, memory, field_summary):
        x = np.concatenate([percept, memory], axis=0)
        h = relu(self.r1(x))
        h = h + 0.1 * field_summary
        return tanh(self.r2(h))


class ActionModule:
    def __init__(self, cfg, rng):
        self.head = Linear(cfg.hidden_dim, cfg.motor_dim, rng)
        self.state = np.zeros(cfg.motor_dim, dtype=np.float32)
        self.decay = cfg.action_decay

    def __call__(self, x, spike_drive):
        logits = self.head(x) + 0.2 * spike_drive[: self.state.shape[0]]
        act = tanh(logits)
        self.state = self.decay * self.state + (1.0 - self.decay) * act
        return self.state.copy()


# ============================================================
# Field core
# ============================================================

class SplitIFTFieldCore:
    """
    10 coupled fields split into:
      fast sensorimotor fields: 0,1,2,3,4
      slow memory/planning fields: 5,6,7,8,9

    Fast fields update every step.
    Slow fields update with a reduced effective rate.
    """

    FAST_IDX = [0, 1, 2, 3, 4]
    SLOW_IDX = [5, 6, 7, 8, 9]

    def __init__(self, cfg, rng):
        self.cfg = cfg
        H, W = cfg.grid_h, cfg.grid_w
        self.phi = rng.standard_normal((10, H, W)).astype(np.float32) * 0.05
        self.t = 0.0

        self.gamma = np.array([0.20, 0.15, 0.10, 0.18, 0.12, 0.13, 0.14, 0.16, 0.11, 0.10], dtype=np.float32)
        self.alpha = np.array([0.30, 0.25, 0.22, 0.18, 0.20, 0.15, 0.19, 0.24, 0.17, 0.16], dtype=np.float32)
        self.beta4 = np.array([0.12, 0.00, 0.00, 0.08, 0.00, 0.03, 0.05, 0.00, 0.03, 0.02], dtype=np.float32)

        self.fast_dt_scale = 1.0
        self.slow_dt_scale = 0.20

        self.coupling_eps_fast = 0.06
        self.coupling_eps_slow = 0.02

        self.fast_to_slow_gain = 0.04
        self.slow_to_fast_gain = 0.02

        self.resonance_omega0 = cfg.resonance_omega0

        self.A = rng.standard_normal((10, 10, 10)).astype(np.float32) * 0.03
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(10, 10, 10)).astype(np.float32)

    def laplacian_2d(self, x):
        return (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
            - 4.0 * x
        )

    def biharmonic_2d(self, x):
        return self.laplacian_2d(self.laplacian_2d(x))

    def safe_log_abs(self, x, eps=1e-6):
        return np.log(np.abs(x) + eps)

    def softclip(self, x, limit=4.0):
        return np.clip(x, -limit, limit)

    def _local_dynamics(self, idx, x):
        lap = self.laplacian_2d(x)
        bilap = self.biharmonic_2d(x)

        if idx == 0:
            dx = -self.gamma[idx] * lap - self.alpha[idx] * (x ** 3) + self.beta4[idx] * bilap
        elif idx == 1:
            dx = -self.gamma[idx] * lap + 0.15 * np.sin(x) + 0.08 * (x ** 3)
        elif idx == 2:
            dx = -self.gamma[idx] * lap + 0.20 * x * (1.0 - x ** 2)
        elif idx == 3:
            dx = -self.gamma[idx] * lap - 0.03 * (x ** 5) + self.beta4[idx] * bilap
        elif idx == 4:
            dx = -self.gamma[idx] * lap + 0.12 * x * self.safe_log_abs(x) + 0.02 * bilap
        elif idx == 5:
            dx = -self.gamma[idx] * lap + 0.08 * (x ** 3) - 0.02 * (x ** 5)
        elif idx == 6:
            dx = -self.gamma[idx] * lap + 0.06 * (x ** 4) - 0.04 * x
        elif idx == 7:
            dx = -self.gamma[idx] * lap + 0.08 * np.sinh(np.clip(x, -2.0, 2.0))
        elif idx == 8:
            dx = -self.gamma[idx] * lap + 0.05 * (x ** 2) - 0.08 * x
        else:
            dx = -self.gamma[idx] * lap + 0.02 * (x ** 6) - 0.04 * (x ** 3) + 0.01 * bilap
        return dx

    def _coupling_step(self):
        out = np.zeros_like(self.phi)
        phi_means = self.phi.mean(axis=(1, 2))
        resonance = np.sin(self.resonance_omega0 * self.t * 0.01 + self.phase)

        for n in range(10):
            acc = 0.0
            for m in range(10):
                for p in range(10):
                    if m == p:
                        continue
                    acc += self.A[n, m, p] * resonance[n, m, p] * self.phi[m] * phi_means[p]
            out[n] = acc
        return out

    def step(self, sensory_map):
        dt = self.cfg.dt
        coupling = self._coupling_step()
        dphi = np.zeros_like(self.phi)

        for i in range(10):
            dphi[i] = self._local_dynamics(i, self.phi[i])

        fast_mean = self.phi[self.FAST_IDX].mean(axis=0)
        slow_mean = self.phi[self.SLOW_IDX].mean(axis=0)

        for i in self.FAST_IDX:
            dphi[i] += self.coupling_eps_fast * coupling[i]
            dphi[i] += self.slow_to_fast_gain * slow_mean

        for i in self.SLOW_IDX:
            dphi[i] += self.coupling_eps_slow * coupling[i]
            dphi[i] += self.fast_to_slow_gain * fast_mean

        dphi[0] += 0.20 * sensory_map
        dphi[4] += 0.08 * np.abs(sensory_map)
        dphi[9] += 0.05 * sensory_map

        self.phi[self.FAST_IDX] += (dt * self.fast_dt_scale) * dphi[self.FAST_IDX]
        self.phi[self.SLOW_IDX] += (dt * self.slow_dt_scale) * dphi[self.SLOW_IDX]

        self.phi = self.softclip(self.phi, 4.0)
        self.t += dt

    def summaries(self):
        phi1 = self.phi[0]
        phi5 = self.phi[4]
        field_means = self.phi.mean(axis=(1, 2))

        return {
            "phi1_mean": float(phi1.mean()),
            "phi5_mean": float(phi5.mean()),
            "field_means": field_means.astype(np.float32),
            "fast_field_mean": self.phi[self.FAST_IDX].mean(axis=(0, 1, 2)).item(),
            "slow_field_mean": self.phi[self.SLOW_IDX].mean(axis=(0, 1, 2)).item(),
            "excitability_map": phi1.copy(),
            "plasticity_map": phi5.copy(),
        }


# ============================================================
# Spiking layer with field modulation
# ============================================================

class FieldSpikingLayer:
    def __init__(self, in_dim, out_dim, rng):
        self.ff = Linear(in_dim, out_dim, rng)
        self.mem = np.zeros(out_dim, dtype=np.float32)
        self.spikes = np.zeros(out_dim, dtype=np.float32)
        self.base_threshold = 0.5
        self.base_leak = 0.90

    def step(self, x, phi1_gain, phi5_gain):
        """
        phi1_gain controls threshold.
        phi5_gain controls leak / retention.
        """
        dyn_threshold = self.base_threshold - 0.25 * phi1_gain
        leak = np.clip(self.base_leak + 0.05 * phi5_gain, 0.70, 0.99)

        current = self.ff(x)
        self.mem = leak * self.mem + current
        self.spikes = (self.mem > dyn_threshold).astype(np.float32)
        self.mem[self.spikes > 0] *= 0.25
        return self.spikes.copy(), self.mem.copy()


# ============================================================
# Full robot brain
# ============================================================

class EdgeRobotBrain:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.sensor_shape = SensorChannelShape()
        self.control_shape = ControlCommandShape()

        if cfg.input_dim != self.sensor_shape.total_dim:
            raise ValueError(
                f"input_dim={cfg.input_dim} does not match sensor channels "
                f"total_dim={self.sensor_shape.total_dim}"
            )

        if cfg.motor_dim != self.control_shape.total_dim:
            raise ValueError(
                f"motor_dim={cfg.motor_dim} does not match control channels "
                f"total_dim={self.control_shape.total_dim}"
            )

        self.perception = PerceptionModule(cfg, self.rng)
        self.memory = MemoryModule(cfg, self.rng)
        self.reasoning = ReasoningModule(cfg, self.rng)
        self.action = ActionModule(cfg, self.rng)
        self.fields = SplitIFTFieldCore(cfg, self.rng)
        self.spike_layer = FieldSpikingLayer(cfg.hidden_dim, cfg.hidden_dim, self.rng)

        self.sensor_proj = Linear(cfg.input_dim, cfg.grid_h * cfg.grid_w, self.rng)

    def _validate_sensor_shape(self, name, arr, expected_dim):
        if arr.shape != (expected_dim,):
            raise ValueError(
                f"Sensor channel '{name}' must have shape ({expected_dim},), got {arr.shape}"
            )

    def _encode_sensor_channels(self, sensors: SensorChannels):
        imu = np.asarray(sensors.imu, dtype=np.float32)
        wheel_odometry = np.asarray(sensors.wheel_odometry, dtype=np.float32)
        camera_embeddings = np.asarray(sensors.camera_embeddings, dtype=np.float32)
        microphone_features = np.asarray(sensors.microphone_features, dtype=np.float32)
        proximity_lidar = np.asarray(sensors.proximity_lidar, dtype=np.float32)
        battery_thermal = np.asarray(sensors.battery_thermal, dtype=np.float32)

        self._validate_sensor_shape("imu", imu, self.sensor_shape.imu)
        self._validate_sensor_shape("wheel_odometry", wheel_odometry, self.sensor_shape.wheel_odometry)
        self._validate_sensor_shape("camera_embeddings", camera_embeddings, self.sensor_shape.camera_embeddings)
        self._validate_sensor_shape("microphone_features", microphone_features, self.sensor_shape.microphone_features)
        self._validate_sensor_shape("proximity_lidar", proximity_lidar, self.sensor_shape.proximity_lidar)
        self._validate_sensor_shape("battery_thermal", battery_thermal, self.sensor_shape.battery_thermal)

        return np.concatenate(
            [
                imu,
                wheel_odometry,
                camera_embeddings,
                microphone_features,
                proximity_lidar,
                battery_thermal,
            ],
            axis=0,
        )

    def _obs_to_map(self, obs):
        z = tanh(self.sensor_proj(obs))
        return z.reshape(self.cfg.grid_h, self.cfg.grid_w)

    def _decode_action_channels(self, action):
        idx = 0

        wheel_velocities = action[idx : idx + self.control_shape.wheel_velocities]
        idx += self.control_shape.wheel_velocities

        arm_joint_deltas = action[idx : idx + self.control_shape.arm_joint_deltas]
        idx += self.control_shape.arm_joint_deltas

        gaze_servo_control = action[idx : idx + self.control_shape.gaze_servo_control]
        idx += self.control_shape.gaze_servo_control

        # Convert compact policy outputs into bounded discrete decisions.
        speech_raw = action[idx]
        idx += self.control_shape.speech_act_selection
        planner_raw = action[idx]

        speech_act_selection = int(np.clip(np.round((speech_raw + 1.0) * 1.5), 0, 3))
        planner_state_transitions = int(np.clip(np.round((planner_raw + 1.0) * 2.0), 0, 4))

        return {
            "wheel_velocities": wheel_velocities.copy(),
            "arm_joint_deltas": arm_joint_deltas.copy(),
            "gaze_servo_control": gaze_servo_control.copy(),
            "speech_act_selection": speech_act_selection,
            "planner_state_transitions": planner_state_transitions,
        }

    def step(self, obs):
        """
        obs shape: [input_dim]
        """
        obs = np.asarray(obs, dtype=np.float32)
        assert obs.shape == (self.cfg.input_dim,)

        sensory_map = self._obs_to_map(obs)
        self.fields.step(sensory_map)
        fs = self.fields.summaries()

        percept = self.perception(obs)

        # Quadra Mind:
        # Perception -> Memory -> Reasoning -> Action
        memory_state = self.memory(percept, fs["plasticity_map"])
        field_summary = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
        field_summary[:10] = fs["field_means"]

        reasoning_state = self.reasoning(percept, memory_state, field_summary)

        phi1_gain = np.full(self.cfg.hidden_dim, fs["phi1_mean"], dtype=np.float32)
        phi5_gain = np.full(self.cfg.hidden_dim, fs["phi5_mean"], dtype=np.float32)

        spikes, membrane = self.spike_layer.step(reasoning_state, phi1_gain, phi5_gain)
        action = self.action(reasoning_state, spikes)
        control = self._decode_action_channels(action)

        telemetry = {
            "phi1_mean": fs["phi1_mean"],
            "phi5_mean": fs["phi5_mean"],
            "spike_rate": float(spikes.mean()),
            "memory_norm": float(np.linalg.norm(memory_state)),
            "action_norm": float(np.linalg.norm(action)),
            "field_means": fs["field_means"],
            "wheel_velocities": control["wheel_velocities"],
            "arm_joint_deltas": control["arm_joint_deltas"],
            "gaze_servo_control": control["gaze_servo_control"],
            "speech_act_selection": control["speech_act_selection"],
            "planner_state_transitions": control["planner_state_transitions"],
        }
        return telemetry

    def step_from_sensors(self, sensors: SensorChannels):
        obs = self._encode_sensor_channels(sensors)
        return self.step(obs)


# ============================================================
# Demo loop
# ============================================================

def demo():
    cfg = BrainConfig(
        grid_h=16,
        grid_w=16,
        input_dim=32,
        hidden_dim=64,
        motor_dim=8,
        dt=0.02,
        coupling_eps=0.05,
        seed=42,
    )

    brain = EdgeRobotBrain(cfg)

    print("Starting edge robot brain demo...")
    demo_rng = np.random.default_rng(123)
    for t in range(20):
        # Example packet with explicit sensor channels.
        sensors = SensorChannels(
            imu=np.array(
                [
                    0.1 * math.sin(t * 0.17),
                    0.1 * math.cos(t * 0.11),
                    9.81 + 0.05 * math.sin(t * 0.07),
                    0.03 * math.sin(t * 0.21),
                    0.02 * math.cos(t * 0.19),
                    0.01 * math.sin(t * 0.13),
                ],
                dtype=np.float32,
            ),
            wheel_odometry=np.array(
                [
                    0.4 + 0.05 * math.sin(t * 0.09),
                    0.38 + 0.05 * math.cos(t * 0.08),
                    0.02 * math.sin(t * 0.04),
                    0.03 * math.cos(t * 0.05),
                ],
                dtype=np.float32,
            ),
            camera_embeddings=demo_rng.normal(0.0, 0.15, size=12).astype(np.float32),
            microphone_features=np.array(
                [
                    0.3 + 0.1 * math.sin(t * 0.14),
                    0.2 + 0.08 * math.cos(t * 0.12),
                    0.1 + 0.05 * math.sin(t * 0.06),
                    0.05 + 0.03 * math.cos(t * 0.15),
                ],
                dtype=np.float32,
            ),
            proximity_lidar=np.array(
                [
                    1.2 + 0.2 * math.sin(t * 0.05),
                    1.0 + 0.2 * math.cos(t * 0.06),
                    0.9 + 0.2 * math.sin(t * 0.08),
                    1.1 + 0.2 * math.cos(t * 0.07),
                ],
                dtype=np.float32,
            ),
            battery_thermal=np.array(
                [
                    0.78 - 0.002 * t,
                    42.0 + 0.1 * math.sin(t * 0.03),
                ],
                dtype=np.float32,
            ),
        )

        out = brain.step_from_sensors(sensors)

        print(
            f"step={t:02d} "
            f"phi1={out['phi1_mean']:+.3f} "
            f"phi5={out['phi5_mean']:+.3f} "
            f"spike_rate={out['spike_rate']:.3f} "
            f"memory={out['memory_norm']:.3f} "
            f"wheels={np.round(out['wheel_velocities'], 3)} "
            f"arm={np.round(out['arm_joint_deltas'], 3)} "
            f"gaze={np.round(out['gaze_servo_control'], 3)} "
            f"speech={out['speech_act_selection']} "
            f"planner={out['planner_state_transitions']}"
        )


if __name__ == "__main__":
    demo()
