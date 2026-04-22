import argparse
import copy
import json
from pathlib import Path

import torch
from app_config import load_project_config
from artifact_metadata import (
    ARTIFACT_FORMAT,
    ARTIFACT_SCHEMA_VERSION,
    build_metadata,
    cfg_to_dict,
)
from brain_torch import TorchBrainConfig, TorchEdgeRobotBrain
from observability import MetricsTimer, log_event, log_failure


class QuantizableBrainWrapper(torch.nn.Module):
    """
    Quantization wrapper around the main trainable parts.
    Dynamic quantization is easiest for edge CPUs.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs):
        out = self.model(obs)
        return out["motor_command"]

    def reset_state(self, batch_size=1, device="cpu"):
        self.model.reset_state(batch_size=batch_size, device=device)


def _build_fp32_model(checkpoint_path: str, cfg: TorchBrainConfig):
    model = TorchEdgeRobotBrain(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in checkpoint.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    model.eval()
    model.reset_state(batch_size=1, device="cpu")
    return model


def _quantize_wrapper(wrapper: QuantizableBrainWrapper):
    quantized = torch.quantization.quantize_dynamic(
        wrapper,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    quantized.reset_state(batch_size=1, device="cpu")
    return quantized


def _validate_rollout(eager_wrapper, quantized_wrapper, steps: int, seed: int):
    torch.manual_seed(seed)
    inputs = torch.randn(steps, eager_wrapper.model.cfg.input_dim)

    eager_wrapper.reset_state(batch_size=1, device="cpu")
    quantized_wrapper.reset_state(batch_size=1, device="cpu")

    eager_outputs = []
    quant_outputs = []
    with torch.no_grad():
        for i in range(steps):
            x = inputs[i].view(1, -1)
            eager_outputs.append(eager_wrapper(x))
            quant_outputs.append(quantized_wrapper(x))

    eager_cat = torch.cat(eager_outputs, dim=0)
    quant_cat = torch.cat(quant_outputs, dim=0)
    abs_err = (eager_cat - quant_cat).abs()

    return {
        "steps": steps,
        "seed": seed,
        "mean_abs_error": float(abs_err.mean().item()),
        "max_abs_error": float(abs_err.max().item()),
    }


def _save_bundle(
    quantized_model,
    cfg: TorchBrainConfig,
    metrics: dict,
    output_dir: Path,
    checkpoint_path: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "robot_brain_int8_stateful.pt"
    manifest_path = output_dir / "deployment_manifest.json"

    cfg_dict = cfg_to_dict(cfg)
    metadata = build_metadata(cfg_dict, checkpoint_path=checkpoint_path)

    torch.save(
        {
            "state_dict": quantized_model.state_dict(),
            "metadata": metadata,
            "config": cfg_dict,
        },
        artifact_path,
    )

    manifest = {
        "artifact_format": ARTIFACT_FORMAT,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_path": str(artifact_path),
        "runtime": "edge_runtime.EdgeRuntimeBrain",
        "quantization": "dynamic_int8_linear",
        "model_family": metadata["model_family"],
        "model_signature": metadata["model_signature"],
        "config": cfg_dict,
        "validation": metrics,
        "producer": metadata["producer"],
        "created_at_utc": metadata["created_at_utc"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return artifact_path, manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create validated stateful deployment bundle")
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validate-steps", type=int, default=None)
    parser.add_argument("--max-abs-error", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_blob = load_project_config(args.config)
    model_blob = cfg_blob["model"]
    export_blob = cfg_blob["export"]

    checkpoint = args.checkpoint if args.checkpoint is not None else export_blob["checkpoint"]
    output_dir = args.output_dir if args.output_dir is not None else export_blob["output_dir"]
    validate_steps = args.validate_steps if args.validate_steps is not None else export_blob["validate_steps"]
    max_abs_error = args.max_abs_error if args.max_abs_error is not None else export_blob["max_abs_error"]
    seed = args.seed if args.seed is not None else export_blob["seed"]

    timer = MetricsTimer()
    log_event(
        "export_start",
        checkpoint=checkpoint,
        output_dir=output_dir,
        validate_steps=validate_steps,
        seed=seed,
        config=args.config,
    )
    try:
        cfg = TorchBrainConfig(
            grid_h=model_blob["grid_h"],
            grid_w=model_blob["grid_w"],
            input_dim=model_blob["input_dim"],
            hidden_dim=model_blob["hidden_dim"],
            motor_dim=model_blob["motor_dim"],
            dt=model_blob["dt"],
            use_field_enhanced_snn=model_blob["use_field_enhanced_snn"],
            chaos_reduction_enabled=model_blob["chaos_reduction_enabled"],
        )
        fp32_model = _build_fp32_model(checkpoint, cfg)
        eager_wrapper = QuantizableBrainWrapper(fp32_model)

        quantized_wrapper = _quantize_wrapper(
            QuantizableBrainWrapper(copy.deepcopy(fp32_model))
        )

        metrics = _validate_rollout(
            eager_wrapper=eager_wrapper,
            quantized_wrapper=quantized_wrapper,
            steps=validate_steps,
            seed=seed,
        )
        if metrics["max_abs_error"] > max_abs_error:
            raise RuntimeError(
                "Quantized rollout validation failed: "
                f"max_abs_error={metrics['max_abs_error']:.6f} exceeds "
                f"threshold={max_abs_error:.6f}"
            )

        artifact_path, manifest_path = _save_bundle(
            quantized_model=quantized_wrapper,
            cfg=cfg,
            metrics=metrics,
            output_dir=Path(output_dir),
            checkpoint_path=checkpoint,
        )

        log_event(
            "export_complete",
            artifact=str(artifact_path),
            manifest=str(manifest_path),
            mean_abs_error=round(metrics["mean_abs_error"], 8),
            max_abs_error=round(metrics["max_abs_error"], 8),
            elapsed_sec=round(timer.elapsed_sec(), 4),
            artifact_format=ARTIFACT_FORMAT,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )
    except Exception as exc:
        log_failure("export_failed", exc, checkpoint=checkpoint, output_dir=output_dir, config=args.config)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
