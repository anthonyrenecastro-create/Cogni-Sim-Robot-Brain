import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app_config import load_project_config
from brain_torch import TorchBrainConfig, TorchEdgeRobotBrain
from data_pipeline import TensorRobotDataset, build_or_load_versioned_dataset
from observability import MetricsTimer, log_event, log_failure


def _loss_terms(out, target):
    control_loss = F.mse_loss(out["motor_command"], target)
    sparsity_loss = 0.01 * out["spike_rate"].mean()
    stability_loss = 0.001 * (out["phi1_mean"].abs().mean() + out["phi5_mean"].abs().mean())
    return control_loss + sparsity_loss + stability_loss


def train(args):
    total_timer = MetricsTimer()
    cfg_blob = load_project_config(args.config)
    model_blob = cfg_blob["model"]
    train_blob = cfg_blob["train"]

    raw_data = args.raw_data if args.raw_data is not None else train_blob["raw_data"]
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else train_blob["dataset_dir"]
    dataset_name = args.dataset_name if args.dataset_name is not None else train_blob["dataset_name"]
    eval_ratio = args.eval_ratio if args.eval_ratio is not None else train_blob["eval_ratio"]
    epochs = args.epochs if args.epochs is not None else train_blob["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else train_blob["batch_size"]
    checkpoint = args.checkpoint if args.checkpoint is not None else train_blob["checkpoint"]
    lr = args.lr if args.lr is not None else train_blob["lr"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_event("train_start", device=device, epochs=epochs, batch_size=batch_size, config=args.config)

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
    model = TorchEdgeRobotBrain(cfg).to(device)

    artifacts = build_or_load_versioned_dataset(
        raw_path=Path(raw_data),
        output_dir=Path(dataset_dir),
        dataset_name=dataset_name,
        input_dim=cfg.input_dim,
        motor_dim=cfg.motor_dim,
        eval_ratio=eval_ratio,
    )
    log_event(
        "dataset_ready",
        dataset_version=artifacts.version,
        train_size=artifacts.train_size,
        eval_size=artifacts.eval_size,
        manifest=str(artifacts.manifest_path),
    )

    train_ds = TensorRobotDataset(artifacts.train_path)
    eval_ds = TensorRobotDataset(artifacts.eval_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_timer = MetricsTimer()
        train_loss_total = 0.0
        model.train()

        for obs, target in train_loader:
            obs = obs.to(device)
            target = target.to(device)

            if model.memory_state.shape[0] != obs.shape[0]:
                model.reset_state(batch_size=obs.shape[0], device=device)
            else:
                model.detach_state()

            out = model(obs)
            loss = _loss_terms(out, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_total += loss.item()

        model.eval()
        eval_loss_total = 0.0
        with torch.no_grad():
            for obs, target in eval_loader:
                obs = obs.to(device)
                target = target.to(device)
                model.reset_state(batch_size=obs.shape[0], device=device)
                out = model(obs)
                eval_loss_total += _loss_terms(out, target).item()

        train_loss = train_loss_total / len(train_loader)
        eval_loss = eval_loss_total / len(eval_loader)
        log_event(
            "epoch_complete",
            epoch=epoch,
            train_loss=round(float(train_loss), 6),
            eval_loss=round(float(eval_loss), 6),
            epoch_sec=round(epoch_timer.elapsed_sec(), 4),
        )

    # Exclude transient runtime buffers so the checkpoint is batch-size agnostic.
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
    stable_state = {
        k: v for k, v in model.state_dict().items() if k not in transient_keys
    }
    torch.save(stable_state, checkpoint)
    log_event(
        "train_complete",
        checkpoint=checkpoint,
        total_sec=round(total_timer.elapsed_sec(), 4),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train CogniSeer torch brain")
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--raw-data", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--eval-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train(args)
    except Exception as exc:
        log_failure("train_failed", exc, checkpoint=args.checkpoint, config=args.config)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
