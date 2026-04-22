# CogniSeer1.0
Prototype for an edge-computing robot brain

## Packaging And Reproducible Environments

This repository now includes:

- `pyproject.toml` for installable packaging and CLI entrypoints.
- `requirements.txt` for floating dependency installs during development.
- `requirements.lock` for reproducible pinned installs.
- `.python-version` pinning the expected interpreter major/minor.
- `environment.yml` for reproducible conda environment creation.

Install as a package (editable mode):

```bash
python -m pip install -e .
```

Reproducible pip environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
```

Reproducible conda environment:

```bash
conda env create -f environment.yml
conda activate cogniseer-robot-brain
python -m pip install -e .
```

Installed CLI commands:

- `cogniseer-train`
- `cogniseer-export`
- `cogniseer-smoke`

## Makefile Commands

A lightweight `Makefile` is provided to standardize local and CI commands.

```bash
make help
make env
make install
make install-lock
make test
make smoke
```

Smoke target parameters can be overridden:

```bash
make smoke MODEL_PATH=deploy/robot_brain_int8_stateful.pt MANIFEST_PATH=deploy/deployment_manifest.json SMOKE_STEPS=16 SMOKE_SEED=123
```

`make smoke` includes failure handling and can auto-bootstrap a deterministic smoke bundle when artifacts are missing.

## Structured Logging, Metrics, And Failure Handling

Training, export, and smoke scripts emit structured JSON logs through `observability.py`.

- Training logs: `train_start`, `dataset_ready`, `epoch_complete`, `train_complete`, `train_failed`
- Export logs: `export_start`, `export_complete`, `export_failed`
- Smoke logs: `smoke_start`, `smoke_metrics`, `smoke_complete`, `smoke_failed`

Each log line includes a UTC timestamp, level, event name, and metrics fields (for example epoch losses, runtime seconds, and replay error).

## CI Workflow

CI is defined in `.github/workflows/ci.yml` and uses the same Make targets as local development:

- `make install-lock`
- `make install`
- `make test`
- `make smoke`

This keeps local and CI command paths identical.

## Deployment Flow

Full step-by-step deployment instructions — including ROS2 workspace build, parameter configuration, topic verification, and artifact compatibility notes — are in [DEPLOYMENT.md](DEPLOYMENT.md).

Quick reference:

```bash
# 1. Train
python train_brain.py

# 2. Export
python quantize_brain.py --checkpoint robot_brain_fp32.pt --output-dir deploy

# 3. Smoke test
make smoke

# 4. Build and run the ROS2 node
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_brain_ros2 --symlink-install
source install/setup.bash
ros2 run robot_brain_ros2 robot_brain_node
```

`quantize_brain.py` exports a validated stateful deployment bundle (not TorchScript tracing).
This path preserves recurrent/internal model state and verifies rollout parity against FP32.

Bundle output in `deploy/`:

- `robot_brain_int8_stateful.pt` (dynamic-int8 eager state dict)
- `deployment_manifest.json` (model config + validation metrics)

Example:

```bash
python quantize_brain.py \
	--config config/defaults.json \
	--checkpoint robot_brain_fp32.pt \
	--output-dir deploy \
	--validate-steps 128 \
	--max-abs-error 0.35
```

Exported artifacts now include metadata (`artifact_format`, `schema_version`, `model_signature`, producer info) and runtime performs compatibility checks against the manifest before loading.

Inference runtime now supports deterministic session management:

- `start_session(session_id, reset=True)` starts a named session with deterministic reset.
- `step(obs, session_id=..., reset_session=False)` auto-resets if session changes.
- `reset()` restores a deterministic baseline state.
- `session_info()` returns current session id and step count.

CI-ready deployment smoke test command:

```bash
python scripts/deployment_smoke_test.py \
	--model-path deploy/robot_brain_int8_stateful.pt \
	--manifest-path deploy/deployment_manifest.json \
	--steps 32 \
	--seed 2026 \
	--max-replay-error 1e-6
```

This verifies fixed-input rollout replay determinism after a session reset.

## Data Pipeline

Training now uses a real file-backed pipeline instead of random in-memory samples.

- Raw dataset source: `datasets/raw/robot_telemetry.jsonl` (or CSV).
- Schema validation: every record is validated with JSON Schema before processing.
- Dataset versioning: processed data is written to `datasets/processed/<dataset_name>/vNNN/`.
- Evaluation split: deterministic train/eval split is produced from `record_id` hashing.

Install dependencies:

```bash
pip install -r requirements.txt
```

Configuration is managed via `config/defaults.json` (with built-in fallbacks in code).
Both training and export support `--config <path>` plus CLI overrides.

Run training with defaults:

```bash
python train_brain.py
```

Run training with an explicit config file:

```bash
python train_brain.py --config config/defaults.json
```

Run training with explicit paths/split:

```bash
python train_brain.py \
	--raw-data datasets/raw/robot_telemetry.jsonl \
	--dataset-dir datasets/processed \
	--dataset-name robot_telemetry \
	--eval-ratio 0.2 \
	--epochs 10 \
	--batch-size 32
```

Processed output per version:

- `train.pt`
- `eval.pt`
- `manifest.json` (fingerprint, schema version, sizes, source path)

### Raw Record Schema

Each JSONL line must be a JSON object:

```json
{
	"record_id": "sample-0001",
	"timestamp_ms": 1710000000020,
	"observation": [0.1, 0.2, "... 32 values total ..."],
	"motor_target": [0.0, "... 8 values total ..."]
}
```

- `observation` length must match model `input_dim` (default 32).
- `motor_target` length must match model `motor_dim` (default 8).

## Actuation Safety Layer

A hard safety layer is applied between model output and robot actuation (`/cmd_vel`).

Safety behavior:

- Rejects non-finite model outputs and commands a zero-motion stop.
- Clips motor command vector range (`safety.motor_min`, `safety.motor_max`).
- Clamps linear/angular velocity to hard limits (`safety.max_linear_x`, `safety.max_angular_z`).
- Applies per-tick slew-rate limiting (`safety.max_linear_delta`, `safety.max_angular_delta`).
- Applies stale-sensor fail-safe stop when no fresh sensor update is seen within `safety.sensor_timeout_sec`.

ROS2 parameters (defaults):

- `safety.motor_min=-1.0`
- `safety.motor_max=1.0`
- `safety.max_linear_x=0.5`
- `safety.max_angular_z=1.0`
- `safety.max_linear_delta=0.10`
- `safety.max_angular_delta=0.20`
- `safety.sensor_timeout_sec=0.50`
- `safety.stop_on_stale_sensor=true`

Focused safety unit test command:

```bash
python scripts/safety_layer_unit_test.py
```

This verifies clamp, slew-rate limiting, NaN fail-safe stop, and stale-sensor fail-safe stop using fixed vectors.

## Test Suite

Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Test coverage categories:

- Unit: `tests/test_unit_safety_layer.py`
- Regression: `tests/test_regression_data_pipeline.py`
- Export/Load: `tests/test_export_load.py`
- Integration: `tests/test_integration_pipeline_export_smoke.py`

Torch-dependent tests are automatically skipped when `torch` is not installed.
