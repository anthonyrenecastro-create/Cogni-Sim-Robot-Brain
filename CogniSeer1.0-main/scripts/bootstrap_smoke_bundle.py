#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observability import fail_exit, log_event, log_failure


def parse_args():
    parser = argparse.ArgumentParser(description="Create a minimal deploy bundle for smoke testing")
    parser.add_argument("--output-dir", default="deploy")
    parser.add_argument("--checkpoint", default="deploy/_smoke_fp32.pt")
    parser.add_argument("--validate-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-abs-error", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        from brain_torch import TorchBrainConfig, TorchEdgeRobotBrain
        from quantize_brain import (
            QuantizableBrainWrapper,
            _build_fp32_model,
            _quantize_wrapper,
            _save_bundle,
            _validate_rollout,
        )
    except Exception as exc:
        log_failure("bootstrap_smoke_failed", exc)
        fail_exit(
            "Bootstrap smoke bundle requires torch. Install dependencies with 'make install-lock' first.",
            code=2,
        )

    log_event("bootstrap_smoke_start", output_dir=args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TorchBrainConfig(input_dim=32, hidden_dim=64, motor_dim=8)
    model = TorchEdgeRobotBrain(cfg).eval()

    transient_keys = {
        "memory_state",
        "spike_mem",
        "action_state",
        "fields.phi",
        "resonance.phi",
        "resonance.stability",
        "resonance.field_mean",
        "resonance.variance",
        "field_enhanced_snn.phi1",
        "field_enhanced_snn.phi5",
        "field_enhanced_snn.Phi",
        "field_enhanced_snn.mem_potential",
    }
    stable_state = {k: v for k, v in model.state_dict().items() if k not in transient_keys}
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stable_state, checkpoint)

    fp32 = _build_fp32_model(str(checkpoint), cfg)
    eager = QuantizableBrainWrapper(fp32)
    quant = _quantize_wrapper(QuantizableBrainWrapper(fp32))
    metrics = _validate_rollout(eager, quant, steps=args.validate_steps, seed=args.seed)
    if metrics["max_abs_error"] > args.max_abs_error:
        raise RuntimeError("bootstrap quantized rollout validation failed")

    _save_bundle(quant, cfg, metrics, output_dir, checkpoint_path=str(checkpoint))
    log_event("bootstrap_smoke_complete", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
