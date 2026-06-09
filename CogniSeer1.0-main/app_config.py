import copy
import json
import os
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

_SUPPORTED_ENVS = {"dev", "staging", "prod"}


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return payload


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate_config(cfg: Dict[str, Any]) -> None:
    required = {
        "model": {
            "grid_h",
            "grid_w",
            "input_dim",
            "hidden_dim",
            "motor_dim",
            "dt",
            "use_field_enhanced_snn",
            "chaos_reduction_enabled",
        },
        "train": {
            "raw_data",
            "dataset_dir",
            "dataset_name",
            "eval_ratio",
            "epochs",
            "batch_size",
            "checkpoint",
            "lr",
        },
        "export": {
            "checkpoint",
            "output_dir",
            "validate_steps",
            "max_abs_error",
            "seed",
        },
    }
    for section, keys in required.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing config section: {section}")
        missing = sorted(keys - set(cfg[section].keys()))
        if missing:
            raise ValueError(f"Missing keys in '{section}': {missing}")

    eval_ratio = float(cfg["train"]["eval_ratio"])
    if not (0.0 < eval_ratio < 1.0):
        raise ValueError("train.eval_ratio must be between 0 and 1")

    if int(cfg["model"]["input_dim"]) <= 0 or int(cfg["model"]["motor_dim"]) <= 0:
        raise ValueError("model.input_dim and model.motor_dim must be positive")

    if int(cfg["train"]["epochs"]) <= 0 or int(cfg["train"]["batch_size"]) <= 0:
        raise ValueError("train.epochs and train.batch_size must be positive")

    if int(cfg["export"]["validate_steps"]) <= 0:
        raise ValueError("export.validate_steps must be positive")


def _resolve_environment_config_path(env_name: str) -> Path:
    env_cfg_override = os.getenv("COGNISEER_ENV_CONFIG")
    if env_cfg_override:
        return Path(env_cfg_override)
    return Path("config") / "environments" / f"{env_name}.json"


def load_project_config(config_path: str | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_name = os.getenv("COGNISEER_ENV", "dev").strip().lower()
    if env_name not in _SUPPORTED_ENVS:
        allowed = ", ".join(sorted(_SUPPORTED_ENVS))
        raise ValueError(f"Unsupported COGNISEER_ENV='{env_name}'. Expected one of: {allowed}")

    base_path = Path(config_path) if config_path else Path("config/defaults.json")
    if base_path.exists():
        cfg = _deep_merge(cfg, _load_json_object(base_path))

    env_path = _resolve_environment_config_path(env_name)
    if env_path.exists():
        cfg = _deep_merge(cfg, _load_json_object(env_path))
    elif env_name != "dev":
        raise FileNotFoundError(
            f"Environment config not found for '{env_name}': {env_path}. "
            "Set COGNISEER_ENV_CONFIG or add the file."
        )

    _validate_config(cfg)
    return cfg
