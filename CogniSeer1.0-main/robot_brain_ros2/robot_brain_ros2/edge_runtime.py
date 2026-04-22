import json
import copy
from pathlib import Path

import torch

from artifact_metadata import ARTIFACT_FORMAT, ARTIFACT_SCHEMA_VERSION, model_signature_from_cfg


def _load_manifest_checked(manifest_path, input_dim):
    file = Path(manifest_path)
    if not file.exists():
        raise FileNotFoundError(f"Deployment manifest not found: {manifest_path}")

    payload = json.loads(file.read_text(encoding="utf-8"))
    if payload.get("artifact_format") != ARTIFACT_FORMAT:
        raise RuntimeError(
            "Unsupported artifact format: "
            f"{payload.get('artifact_format')} (expected {ARTIFACT_FORMAT})"
        )
    if int(payload.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported artifact schema version: "
            f"{payload.get('schema_version')} (expected {ARTIFACT_SCHEMA_VERSION})"
        )

    cfg_blob = payload.get("config", {})
    required_keys = {
        "grid_h",
        "grid_w",
        "input_dim",
        "hidden_dim",
        "motor_dim",
        "dt",
        "use_field_enhanced_snn",
        "chaos_reduction_enabled",
    }
    missing = sorted(required_keys - set(cfg_blob.keys()))
    if missing:
        raise RuntimeError(f"Manifest config missing required keys: {missing}")

    if int(cfg_blob["input_dim"]) != int(input_dim):
        raise RuntimeError(
            "Input dimension mismatch between runtime request and artifact: "
            f"runtime={input_dim} artifact={cfg_blob['input_dim']}"
        )

    expected_signature = model_signature_from_cfg(cfg_blob)
    if payload.get("model_signature") != expected_signature:
        raise RuntimeError("Manifest model_signature mismatch; artifact is not trusted")

    return payload, cfg_blob, expected_signature


class EdgeRuntimeBrain:
    def __init__(
        self,
        model_path="deploy/robot_brain_int8_stateful.pt",
        input_dim=32,
        manifest_path="deploy/deployment_manifest.json",
    ):
        self.input_dim = int(input_dim)

        if str(model_path).endswith(".ts"):
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            self._baseline_state = None
            self._active_session = None
            self._session_step = 0
            return

        try:
            from brain_torch import TorchBrainConfig, TorchEdgeRobotBrain
        except ImportError as exc:
            raise RuntimeError(
                "Stateful .pt deployment requires brain_torch imports to be available. "
                "Either provide a .ts artifact or install project modules in the ROS2 environment."
            ) from exc

        class QuantizableBrainWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, obs):
                return self.model(obs)["motor_command"]

            def reset_state(self, batch_size=1, device="cpu"):
                self.model.reset_state(batch_size=batch_size, device=device)

        _manifest, cfg_blob, expected_signature = _load_manifest_checked(
            manifest_path,
            input_dim=self.input_dim,
        )
        cfg = TorchBrainConfig(
            grid_h=cfg_blob["grid_h"],
            grid_w=cfg_blob["grid_w"],
            input_dim=cfg_blob["input_dim"],
            hidden_dim=cfg_blob["hidden_dim"],
            motor_dim=cfg_blob["motor_dim"],
            dt=cfg_blob["dt"],
            use_field_enhanced_snn=cfg_blob["use_field_enhanced_snn"],
            chaos_reduction_enabled=cfg_blob["chaos_reduction_enabled"],
        )

        wrapped = QuantizableBrainWrapper(TorchEdgeRobotBrain(cfg))
        self.model = torch.quantization.quantize_dynamic(
            wrapped,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
            raise RuntimeError("Invalid artifact payload: missing state_dict and metadata")

        metadata = payload["metadata"]
        if metadata.get("artifact_format") != ARTIFACT_FORMAT:
            raise RuntimeError("Artifact metadata format mismatch")
        if int(metadata.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError("Artifact metadata schema_version mismatch")
        if metadata.get("model_signature") != expected_signature:
            raise RuntimeError("Artifact metadata model_signature mismatch")

        state_dict = payload["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.reset_state(batch_size=1, device="cpu")
        self._baseline_state = copy.deepcopy(self.model.state_dict())
        self._active_session = None
        self._session_step = 0
        self.input_dim = cfg.input_dim

    def start_session(self, session_id=None, reset=True):
        if session_id is None:
            session_id = "default"
        if reset:
            self.reset()
        self._active_session = str(session_id)
        self._session_step = 0
        return self._active_session

    def step(self, obs, session_id=None, reset_session=False):
        if session_id is None:
            if self._active_session is None:
                self.start_session("default", reset=True)
        else:
            session_id = str(session_id)
            if reset_session or self._active_session != session_id:
                self.start_session(session_id=session_id, reset=True)

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).view(1, self.input_dim)
            y = self.model(x).squeeze(0).cpu().numpy()

        self._session_step += 1
        return y

    def reset(self):
        if self._baseline_state is not None:
            self.model.load_state_dict(self._baseline_state, strict=True)
        if hasattr(self.model, "reset_state"):
            self.model.reset_state(batch_size=1, device="cpu")

    def session_info(self):
        return {
            "session_id": self._active_session,
            "step": self._session_step,
        }
