import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Dict, Any


ARTIFACT_FORMAT = "quantized_eager_state_dict_v2"
ARTIFACT_SCHEMA_VERSION = 2


def cfg_to_dict(cfg) -> Dict[str, Any]:
    return {
        "input_dim": int(cfg.input_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "motor_dim": int(cfg.motor_dim),
        "grid_h": int(cfg.grid_h),
        "grid_w": int(cfg.grid_w),
        "dt": float(cfg.dt),
        "use_field_enhanced_snn": bool(cfg.use_field_enhanced_snn),
        "chaos_reduction_enabled": bool(cfg.chaos_reduction_enabled),
    }


def model_signature_from_cfg(cfg_dict: Dict[str, Any]) -> str:
    stable = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def build_metadata(cfg_dict: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": "TorchEdgeRobotBrain",
        "model_signature": model_signature_from_cfg(cfg_dict),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": checkpoint_path,
        "producer": {
            "python": platform.python_version(),
        },
    }
