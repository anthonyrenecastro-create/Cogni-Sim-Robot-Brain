import copy
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "grid_h": 16,
        "grid_w": 16,
        "input_dim": 32,
        "hidden_dim": 64,
        "motor_dim": 8,
        "dt": 0.02,
        "use_field_enhanced_snn": False,
        "chaos_reduction_enabled": True,
    },
    "train": {
        "raw_data": "datasets/raw/robot_telemetry.jsonl",
        "dataset_dir": "datasets/processed",
        "dataset_name": "robot_telemetry",
        "eval_ratio": 0.2,
        "epochs": 10,
        "batch_size": 32,
        "checkpoint": "robot_brain_fp32.pt",
        "lr": 1e-3,
    },
    "export": {
        "checkpoint": "robot_brain_fp32.pt",
        "output_dir": "deploy",
        "validate_steps": 128,
        "max_abs_error": 0.35,
        "seed": 1234,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_project_config(config_path: str | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return cfg

    path = Path(config_path)
    if not path.exists():
        return cfg

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must be a JSON object")
    return _deep_merge(cfg, payload)
