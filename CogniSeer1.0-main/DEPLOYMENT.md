# CogniSeer Robot Brain — Deployment Guide

End-to-end steps for training, exporting, and running the INT8-quantized edge brain on a ROS2-enabled robot.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | Pinned via `.python-version` |
| PyTorch | ≥ 2.2.0 | CPU-only build is sufficient for inference |
| NumPy | ≥ 1.26.0 | |
| ROS2 | Humble or Iron | `ament_python` build system |
| colcon | any | `sudo apt install python3-colcon-common-extensions` |

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate cogniseer-robot-brain
pip install -e .
```

---

## Step 1 — Prepare the Raw Dataset

Place your telemetry data at `datasets/raw/robot_telemetry.jsonl`.  
Each line must be a JSON object matching the schema (see [README.md](README.md#raw-record-schema)):

```json
{
  "record_id": "sample-0001",
  "timestamp_ms": 1710000000020,
  "observation": [0.1, 0.2, "... 32 floats total ..."],
  "motor_target": [0.0, "... 8 floats total ..."]
}
```

---

## Step 2 — Train

```bash
python train_brain.py
```

Override specific values without editing any file:

```bash
python train_brain.py \
  --config config/defaults.json \
  --raw-data datasets/raw/robot_telemetry.jsonl \
  --epochs 20 \
  --batch-size 64 \
  --lr 5e-4
```

On completion the checkpoint is written to `robot_brain_fp32.pt` and structured logs (JSON) are emitted to stdout.

---

## Step 3 — Export the Deployment Bundle

```bash
python quantize_brain.py \
  --config config/defaults.json \
  --checkpoint robot_brain_fp32.pt \
  --output-dir deploy \
  --validate-steps 128 \
  --max-abs-error 0.35
```

This produces two files in `deploy/`:

| File | Description |
|---|---|
| `robot_brain_int8_stateful.pt` | INT8-quantized eager state dict |
| `deployment_manifest.json` | Model config, validation metrics, and artifact metadata |

The export script validates rollout parity between the FP32 checkpoint and the INT8 bundle.  
The manifest records `artifact_format`, `schema_version`, `model_signature` (SHA-256 of config), `producer`, and `created_at_utc`.

---

## Step 4 — Run the Deployment Smoke Test

Verify the bundle loads cleanly and produces deterministic outputs before deploying to hardware:

```bash
python scripts/deployment_smoke_test.py \
  --model-path deploy/robot_brain_int8_stateful.pt \
  --manifest-path deploy/deployment_manifest.json \
  --steps 32 \
  --seed 2026 \
  --max-replay-error 1e-6
```

Exit code `0` means the bundle passes replay determinism checks.  
You can also run this via Make:

```bash
make smoke
```

---

## Step 5 — Build the ROS2 Package

Source your ROS2 installation, then build from the workspace root:

```bash
source /opt/ros/humble/setup.bash        # or /opt/ros/iron/setup.bash

cd /workspaces/Cogni-Sim-Robot-Brain
colcon build --packages-select robot_brain_ros2 --symlink-install
source install/setup.bash
```

`--symlink-install` is recommended during development so Python file edits are picked up without rebuilding.

---

## Step 6 — Copy the Deployment Bundle

Place the exported artifacts where the node can find them.  
The default parameter is `model_path = deploy/robot_brain_int8_stateful.pt` resolved relative to the working directory when you launch the node.

Option A — copy into the ROS2 install share directory:

```bash
mkdir -p install/robot_brain_ros2/share/robot_brain_ros2/deploy
cp deploy/robot_brain_int8_stateful.pt \
   deploy/deployment_manifest.json \
   install/robot_brain_ros2/share/robot_brain_ros2/deploy/
```

Option B — keep artifacts in-tree and pass the absolute path at launch (see Step 7).

---

## Step 7 — Launch the Node

Minimal launch (uses all parameter defaults):

```bash
ros2 run robot_brain_ros2 robot_brain_node
```

Override parameters inline:

```bash
ros2 run robot_brain_ros2 robot_brain_node \
  --ros-args \
  -p model_path:=/absolute/path/to/deploy/robot_brain_int8_stateful.pt \
  -p input_dim:=32 \
  -p cmd_scale_linear:=0.4 \
  -p cmd_scale_angular:=0.8
```

Override safety limits:

```bash
ros2 run robot_brain_ros2 robot_brain_node \
  --ros-args \
  -p safety.max_linear_x:=0.3 \
  -p safety.max_angular_z:=0.6 \
  -p safety.sensor_timeout_sec:=1.0
```

---

## Step 8 — Verify Node Topics

In a second terminal (with the overlay sourced):

```bash
source install/setup.bash

# Check the node is alive
ros2 node list

# Inspect the published velocity commands
ros2 topic echo /cmd_vel

# Publish a synthetic sensor frame for testing
ros2 topic pub --once /sensor_input std_msgs/msg/Float32MultiArray \
  "data: [$(python3 -c "import numpy as np; print(','.join(map(str,np.zeros(32,dtype=float))))" )]"
```

Expected `/cmd_vel` output is a `geometry_msgs/msg/Twist` with `linear.x` and `angular.z` within the configured safety limits.

---

## Step 9 — Runtime Safety Parameters

All safety parameters are ROS2 parameters and can be changed at runtime (subject to node restart):

| Parameter | Default | Description |
|---|---|---|
| `safety.motor_min` | `-1.0` | Lower clamp on raw motor output |
| `safety.motor_max` | `1.0` | Upper clamp on raw motor output |
| `safety.max_linear_x` | `0.5` | Hard linear velocity cap (m/s) |
| `safety.max_angular_z` | `1.0` | Hard angular velocity cap (rad/s) |
| `safety.max_linear_delta` | `0.10` | Max per-tick linear change (slew rate) |
| `safety.max_angular_delta` | `0.20` | Max per-tick angular change (slew rate) |
| `safety.sensor_timeout_sec` | `0.50` | Stale-sensor fail-safe threshold (s) |
| `safety.stop_on_stale_sensor` | `true` | Command full stop when sensor is stale |

Safety layer behaviour (applied in order):

1. **NaN/Inf reject** — non-finite model output → zero-motion stop
2. **Motor clamp** — raw motor vector clipped to `[motor_min, motor_max]`
3. **Stale-sensor stop** — no sensor update within `sensor_timeout_sec` → zero-motion stop
4. **Velocity cap** — `linear_x` and `angular_z` hard-limited
5. **Slew-rate limit** — per-tick change limited to `max_linear_delta` / `max_angular_delta`

---

## Configuration Reference

All model and training hyperparameters live in `config/defaults.json`.  
Any value can be overridden at the CLI without editing the file.

```json
{
  "model": {
    "grid_h": 16,   "grid_w": 16,
    "input_dim": 32, "hidden_dim": 64, "motor_dim": 8, "dt": 0.02
  },
  "train": {
    "epochs": 10, "batch_size": 32, "lr": 0.001
  },
  "export": {
    "validate_steps": 128, "max_abs_error": 0.35, "seed": 1234
  }
}
```

---

## CI Pipeline

The GitHub Actions workflow at `.github/workflows/ci.yml` runs the same steps as local development on every push and pull request to `main`:

```
make install-lock → make install → make test → make smoke
```

The smoke step auto-bootstraps a random-init bundle when no trained artifacts are present, ensuring CI always runs the full load-and-inference path.

---

## Artifact Compatibility

The runtime performs strict compatibility checks on every load:

- `artifact_format` must equal `quantized_eager_state_dict_v2`
- `schema_version` must equal `2`
- All required config keys (`grid_h`, `grid_w`, `input_dim`, `hidden_dim`, `motor_dim`, `dt`) must be present
- `input_dim` in the manifest must match the runtime `input_dim` parameter
- `model_signature` (SHA-256 of model config) must match between manifest and artifact payload

A `RuntimeError` with a descriptive message is raised on any mismatch, preventing silent deployment of stale or mismatched artifacts.
