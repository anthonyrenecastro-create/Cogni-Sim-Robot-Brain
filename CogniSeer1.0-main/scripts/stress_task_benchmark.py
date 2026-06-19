#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_layer import ActuationSafetyLayer, SafetyConfig
from torch_edge_robot_brain import TorchBrainConfig, TorchEdgeRobotBrain


@dataclass
class DisturbanceConfig:
    sensor_noise_std: float = 0.10
    stale_frame_prob: float = 0.08
    actuator_delay_steps: int = 2
    perturb_prob: float = 0.05
    perturb_accel_scale: float = 0.40
    obs_dropout_prob: float = 0.15
    obs_scale_min: float = 0.60


@dataclass
class TaskConfig:
    dt: float = 0.02
    horizon_steps: int = 500
    goal_radius: float = 0.25
    collision_radius: float = 0.22
    max_speed: float = 0.90
    fall_tilt_threshold: float = 0.95


def _angle_wrap(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _build_observation(
    pos: np.ndarray,
    vel: np.ndarray,
    theta: float,
    tilt: float,
    goal: np.ndarray,
    obstacles: np.ndarray,
    prev_action: np.ndarray,
    step_idx: int,
    horizon: int,
) -> np.ndarray:
    to_goal = goal - pos
    dist_goal = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / max(dist_goal, 1e-6)

    dists = np.linalg.norm(obstacles - pos[None, :], axis=1)
    nearest_obs_idx = int(np.argmin(dists))
    nearest_obs = obstacles[nearest_obs_idx] - pos
    nearest_obs_dist = float(dists[nearest_obs_idx])

    obs = np.zeros(32, dtype=np.float32)
    obs[0:2] = pos
    obs[2:4] = vel
    obs[4] = math.sin(theta)
    obs[5] = math.cos(theta)
    obs[6] = tilt
    obs[7:9] = to_goal
    obs[9] = dist_goal
    obs[10:12] = goal_dir
    obs[12:14] = nearest_obs
    obs[14] = nearest_obs_dist
    obs[15:17] = prev_action[:2]
    obs[17] = float(step_idx) / float(max(horizon, 1))

    # Remaining channels are deterministic nonlinear mixes to emulate richer telemetry.
    obs[18] = obs[0] * obs[4]
    obs[19] = obs[1] * obs[5]
    obs[20] = obs[2] * obs[3]
    obs[21] = obs[6] * obs[9]
    obs[22] = obs[10] * obs[12]
    obs[23] = obs[11] * obs[13]
    obs[24] = float(np.tanh(obs[7]))
    obs[25] = float(np.tanh(obs[8]))
    obs[26] = float(np.tanh(obs[14]))
    obs[27] = float(np.clip(obs[17] * 2.0 - 1.0, -1.0, 1.0))
    obs[28] = float(np.linalg.norm(vel))
    obs[29] = float(np.dot(goal_dir, vel))
    obs[30] = float(np.dot(goal_dir, nearest_obs / max(nearest_obs_dist, 1e-6)))
    obs[31] = float(np.linalg.norm(prev_action))
    return obs


def _apply_observation_degradation(obs: np.ndarray, cfg: DisturbanceConfig, rng: np.random.Generator) -> np.ndarray:
    degraded = obs.copy()
    if cfg.obs_dropout_prob > 0.0:
        mask = rng.random(obs.shape[0]) < cfg.obs_dropout_prob
        degraded[mask] = 0.0
    scale = rng.uniform(cfg.obs_scale_min, 1.0)
    degraded *= scale
    degraded += rng.normal(0.0, cfg.sensor_noise_std, size=degraded.shape).astype(np.float32)
    return degraded


def _task_success_risk_signal(
    pos: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    tilt: float,
    sensor_fresh: bool,
    collided: bool,
    collision_radius: float,
    goal_radius: float,
) -> tuple[float, float]:
    dist_goal = float(np.linalg.norm(goal - pos))
    min_obs_dist = float(np.min(np.linalg.norm(obstacles - pos[None, :], axis=1)))

    # Success grows as the robot closes in on the goal.
    success_signal = 1.0 - min(1.0, dist_goal / max(goal_radius * 8.0, 1e-6))
    if dist_goal <= goal_radius:
        success_signal = 1.0

    # Risk grows with proximity to obstacles, tilt, stale frames, and collisions.
    obstacle_risk = 1.0 - min(1.0, max(0.0, min_obs_dist - collision_radius) / max(collision_radius * 4.0, 1e-6))
    tilt_risk = min(1.0, abs(float(tilt)) / 1.0)
    stale_risk = 0.35 if not sensor_fresh else 0.0
    collision_risk = 1.0 if collided else 0.0

    risk_signal = min(1.0, 0.40 * obstacle_risk + 0.35 * tilt_risk + 0.15 * stale_risk + 0.10 * collision_risk)
    return float(success_signal), float(risk_signal)


def _simulate_episode(
    model: TorchEdgeRobotBrain,
    safety: ActuationSafetyLayer,
    task_cfg: TaskConfig,
    dist_cfg: DisturbanceConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    goal = rng.uniform(-2.5, 2.5, size=2).astype(np.float32)
    pos = rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
    vel = np.zeros(2, dtype=np.float32)
    theta = float(rng.uniform(-0.15, 0.15))
    tilt = float(rng.uniform(-0.05, 0.05))
    prev_action = np.zeros(8, dtype=np.float32)

    obstacles = rng.uniform(-2.8, 2.8, size=(5, 2)).astype(np.float32)
    delay_q: List[np.ndarray] = [np.zeros(2, dtype=np.float32) for _ in range(max(0, dist_cfg.actuator_delay_steps))]

    stale_events = 0
    recovery_windows = 0
    recovered_windows = 0
    collisions = 0
    collision_flag = False
    success = False
    fall = False

    path = [pos.copy()]
    latency_ms: List[float] = []
    chaos_enabled_vals: List[float] = []
    auto_chaos_risk_vals: List[float] = []
    auto_chaos_strength_vals: List[float] = []

    sensor_obs_prev = _build_observation(
        pos=pos,
        vel=vel,
        theta=theta,
        tilt=tilt,
        goal=goal,
        obstacles=obstacles,
        prev_action=prev_action,
        step_idx=0,
        horizon=task_cfg.horizon_steps,
    )

    for step_idx in range(task_cfg.horizon_steps):
        base_obs = _build_observation(
            pos=pos,
            vel=vel,
            theta=theta,
            tilt=tilt,
            goal=goal,
            obstacles=obstacles,
            prev_action=prev_action,
            step_idx=step_idx,
            horizon=task_cfg.horizon_steps,
        )

        sensor_fresh = bool(rng.random() >= dist_cfg.stale_frame_prob)
        if sensor_fresh:
            sensor_obs = _apply_observation_degradation(base_obs, dist_cfg, rng)
            sensor_obs_prev = sensor_obs.copy()
        else:
            stale_events += 1
            sensor_obs = sensor_obs_prev.copy()

        task_success_signal, task_risk_signal = _task_success_risk_signal(
            pos=pos,
            goal=goal,
            obstacles=obstacles,
            tilt=tilt,
            sensor_fresh=sensor_fresh,
            collided=collision_flag,
            collision_radius=task_cfg.collision_radius,
            goal_radius=task_cfg.goal_radius,
        )

        model.set_auto_chaos_context(
            success_signal=task_success_signal,
            risk_signal=task_risk_signal,
        )

        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(torch.from_numpy(sensor_obs).unsqueeze(0))
            latency_ms.append((time.perf_counter() - t0) * 1000.0)
            chaos_enabled_vals.append(float(out.get("chaos_enabled", torch.zeros(1)).mean().item()))
            auto_chaos_risk_vals.append(float(out.get("auto_chaos_risk", torch.zeros(1)).mean().item()))
            auto_chaos_strength_vals.append(float(out.get("auto_chaos_strength", torch.zeros(1)).mean().item()))

        motor = out["motor_command"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        safe = safety.apply(
            motor=motor,
            cmd_scale_linear=1.0,
            cmd_scale_angular=1.0,
            sensor_fresh=sensor_fresh,
        )

        cmd = np.array([safe["linear_x"], safe["angular_z"]], dtype=np.float32)
        if delay_q:
            delay_q.append(cmd)
            cmd = delay_q.pop(0)

        # Random perturbation in dynamics space.
        if rng.random() < dist_cfg.perturb_prob:
            recovery_windows += 1
            kick = rng.normal(0.0, dist_cfg.perturb_accel_scale, size=2).astype(np.float32)
            vel += kick * task_cfg.dt

        heading = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        vel = 0.86 * vel + 0.14 * cmd[0] * heading
        speed = float(np.linalg.norm(vel))
        if speed > task_cfg.max_speed:
            vel *= task_cfg.max_speed / max(speed, 1e-6)
        pos += vel * task_cfg.dt
        theta = _angle_wrap(theta + float(cmd[1]) * task_cfg.dt)

        # Tilt grows with aggressive actuation and perturbation, then relaxes.
        tilt = 0.92 * tilt + 0.06 * abs(float(cmd[1])) + 0.04 * abs(float(cmd[0]))
        if rng.random() < dist_cfg.perturb_prob:
            tilt += abs(float(rng.normal(0.0, 0.15)))
        tilt = float(max(0.0, min(1.5, tilt)))

        # Recovery credit when progress recovers after perturbation/stale events.
        if recovery_windows > recovered_windows:
            prev_dist = float(np.linalg.norm(goal - path[-1]))
            now_dist = float(np.linalg.norm(goal - pos))
            if now_dist < prev_dist:
                recovered_windows += 1

        path.append(pos.copy())
        prev_action = motor

        min_obs_dist = float(np.min(np.linalg.norm(obstacles - pos[None, :], axis=1)))
        if min_obs_dist < task_cfg.collision_radius:
            collisions += 1
            collision_flag = True

        if tilt > task_cfg.fall_tilt_threshold:
            fall = True
            break

        if float(np.linalg.norm(goal - pos)) <= task_cfg.goal_radius:
            success = True
            break

    path_np = np.stack(path, axis=0)
    start = path_np[0]
    goal_vec = goal - start
    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm < 1e-6:
        trajectory_error = float(np.sqrt(np.mean(np.sum((path_np - goal[None, :]) ** 2, axis=1))))
    else:
        unit = goal_vec / goal_norm
        rel = path_np - start[None, :]
        proj = np.sum(rel * unit[None, :], axis=1)
        closest = start[None, :] + proj[:, None] * unit[None, :]
        trajectory_error = float(np.sqrt(np.mean(np.sum((path_np - closest) ** 2, axis=1))))

    return {
        "success": float(success),
        "fall": float(fall),
        "collision": float(collision_flag),
        "recovery_rate_episode": float(recovered_windows / max(1, recovery_windows + stale_events)),
        "collision_events": float(collisions),
        "stale_events": float(stale_events),
        "trajectory_error": trajectory_error,
        "latency_ms": float(statistics.mean(latency_ms) if latency_ms else 0.0),
        "chaos_duty_cycle": float(statistics.mean(chaos_enabled_vals) if chaos_enabled_vals else 0.0),
        "auto_chaos_risk": float(statistics.mean(auto_chaos_risk_vals) if auto_chaos_risk_vals else 0.0),
        "auto_chaos_strength": float(statistics.mean(auto_chaos_strength_vals) if auto_chaos_strength_vals else 0.0),
        "steps": float(len(path) - 1),
    }


def _aggregate_episode_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    n = max(1, len(rows))
    return {
        "episodes": float(n),
        "success_rate": float(sum(r["success"] for r in rows) / n),
        "fall_rate": float(sum(r["fall"] for r in rows) / n),
        "collision_rate": float(sum(r["collision"] for r in rows) / n),
        "recovery_rate": float(sum(r["recovery_rate_episode"] for r in rows) / n),
        "trajectory_error": float(sum(r["trajectory_error"] for r in rows) / n),
        "latency_ms": float(sum(r["latency_ms"] for r in rows) / n),
        "chaos_duty_cycle": float(sum(r["chaos_duty_cycle"] for r in rows) / n),
        "auto_chaos_risk": float(sum(r["auto_chaos_risk"] for r in rows) / n),
        "auto_chaos_strength": float(sum(r["auto_chaos_strength"] for r in rows) / n),
        "avg_steps": float(sum(r["steps"] for r in rows) / n),
    }


def _robustness_score(metrics: Dict[str, float]) -> float:
    success = metrics["success_rate"]
    safety = 1.0 - 0.5 * metrics["fall_rate"] - 0.5 * metrics["collision_rate"]
    recovery = metrics["recovery_rate"]
    traj = 1.0 / (1.0 + metrics["trajectory_error"])
    return float(0.45 * success + 0.20 * max(0.0, safety) + 0.20 * recovery + 0.15 * traj)


def _build_model(
    enable_chaos: bool,
    chaos_strength: float,
    auto_chaos: bool = False,
    auto_on_threshold: float = 0.55,
    auto_off_threshold: float = 0.35,
    auto_ema_alpha: float = 0.20,
    auto_strength_scale: float = 1.0,
) -> TorchEdgeRobotBrain:
    cfg = TorchBrainConfig(
        input_dim=32,
        hidden_dim=64,
        motor_dim=8,
        grid_h=16,
        grid_w=16,
        dt=0.02,
        use_field_enhanced_snn=True,
        chaos_reduction_enabled=enable_chaos,
    )
    model = TorchEdgeRobotBrain(cfg).eval()
    model.set_chaos_control(enable_chaos)
    model.set_chaos_strength(chaos_strength)
    model.configure_auto_chaos(
        enabled=auto_chaos,
        on_threshold=auto_on_threshold,
        off_threshold=auto_off_threshold,
        ema_alpha=auto_ema_alpha,
        strength_scale=auto_strength_scale,
    )

    model.reset_state(batch_size=1)
    return model


def _evaluate_setting(
    episodes: int,
    seed: int,
    chaos_enabled: bool,
    chaos_strength: float,
    task_cfg: TaskConfig,
    dist_cfg: DisturbanceConfig,
    auto_chaos: bool = False,
    auto_on_threshold: float = 0.55,
    auto_off_threshold: float = 0.35,
    auto_ema_alpha: float = 0.20,
    auto_strength_scale: float = 1.0,
    policy_name: str = "",
) -> Dict[str, float]:
    model = _build_model(
        enable_chaos=chaos_enabled,
        chaos_strength=chaos_strength,
        auto_chaos=auto_chaos,
        auto_on_threshold=auto_on_threshold,
        auto_off_threshold=auto_off_threshold,
        auto_ema_alpha=auto_ema_alpha,
        auto_strength_scale=auto_strength_scale,
    )
    safety = ActuationSafetyLayer(SafetyConfig())

    rows = []
    for ep in range(episodes):
        rng = np.random.default_rng(seed + ep)
        model.reset_state(batch_size=1)
        safety.reset()
        rows.append(_simulate_episode(model, safety, task_cfg, dist_cfg, rng))

    metrics = _aggregate_episode_metrics(rows)
    metrics["robustness_score"] = _robustness_score(metrics)
    metrics["chaos_enabled"] = float(1.0 if chaos_enabled else 0.0)
    metrics["chaos_strength"] = float(chaos_strength)
    metrics["auto_chaos_enabled"] = float(1.0 if auto_chaos else 0.0)
    metrics["policy"] = policy_name
    return metrics


def _pct_delta(new_value: float, base_value: float) -> float:
    return 100.0 * ((new_value - base_value) / max(base_value, 1e-8))


def _parse_strengths(raw: str) -> List[float]:
    vals: List[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        vals.append(float(piece))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress/task benchmark with chaos-control ablation")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strengths", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--fixed-on-strength", type=float, default=1.0)
    parser.add_argument("--auto-on-threshold", type=float, default=0.55)
    parser.add_argument("--auto-off-threshold", type=float, default=0.35)
    parser.add_argument("--auto-ema-alpha", type=float, default=0.20)
    parser.add_argument("--auto-strength-scale", type=float, default=1.0)
    parser.add_argument("--success-improvement-threshold-pct", type=float, default=12.0)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")

    strengths = _parse_strengths(args.strengths)
    if not strengths:
        raise ValueError("--strengths must include at least one value")

    task_cfg = TaskConfig()
    dist_cfg = DisturbanceConfig()

    # Baseline OFF under stress profile.
    off = _evaluate_setting(
        episodes=args.episodes,
        seed=args.seed,
        chaos_enabled=False,
        chaos_strength=0.0,
        task_cfg=task_cfg,
        dist_cfg=dist_cfg,
        policy_name="fixed_off",
    )

    fixed_on = _evaluate_setting(
        episodes=args.episodes,
        seed=args.seed,
        chaos_enabled=True,
        chaos_strength=max(0.0, args.fixed_on_strength),
        task_cfg=task_cfg,
        dist_cfg=dist_cfg,
        policy_name="fixed_on",
    )

    auto = _evaluate_setting(
        episodes=args.episodes,
        seed=args.seed,
        chaos_enabled=False,
        chaos_strength=0.0,
        task_cfg=task_cfg,
        dist_cfg=dist_cfg,
        auto_chaos=True,
        auto_on_threshold=args.auto_on_threshold,
        auto_off_threshold=args.auto_off_threshold,
        auto_ema_alpha=args.auto_ema_alpha,
        auto_strength_scale=args.auto_strength_scale,
        policy_name="auto_chaos",
    )

    # Ablation ON at multiple strengths.
    on_rows: List[Dict[str, float]] = []
    for s in strengths:
        if s <= 0.0:
            continue
        on_rows.append(
            _evaluate_setting(
                episodes=args.episodes,
                seed=args.seed,
                chaos_enabled=True,
                chaos_strength=s,
                task_cfg=task_cfg,
                dist_cfg=dist_cfg,
                policy_name="ablation_on",
            )
        )

    if on_rows:
        best_on = max(on_rows, key=lambda r: r["robustness_score"])
    else:
        best_on = off.copy()

    policy_rows: List[Dict[str, float]] = []
    for name, row in [("fixed_off", off), ("fixed_on", fixed_on), ("auto_chaos", auto)]:
        success_gain_pct = _pct_delta(row["success_rate"], off["success_rate"])
        latency_cost_pct = _pct_delta(row["latency_ms"], off["latency_ms"])
        eligible = bool(
            name == "fixed_off"
            or (
                success_gain_pct >= args.success_improvement_threshold_pct
                and success_gain_pct > latency_cost_pct
            )
        )
        policy_rows.append(
            {
                "policy": name,
                "success_rate": row["success_rate"],
                "fall_rate": row["fall_rate"],
                "collision_rate": row["collision_rate"],
                "recovery_rate": row["recovery_rate"],
                "trajectory_error": row["trajectory_error"],
                "latency_ms": row["latency_ms"],
                "chaos_duty_cycle": row["chaos_duty_cycle"],
                "success_gain_pct_vs_off": float(round(success_gain_pct, 4)),
                "latency_cost_pct_vs_off": float(round(latency_cost_pct, 4)),
                "eligible": eligible,
                "score_margin": float(round(success_gain_pct - latency_cost_pct, 4)),
            }
        )

    eligible_candidates = [r for r in policy_rows if r["policy"] != "fixed_off" and r["eligible"]]
    if eligible_candidates:
        recommended = max(eligible_candidates, key=lambda r: (r["score_margin"], r["success_rate"]))
        recommended_policy = str(recommended["policy"])
    else:
        recommended_policy = "fixed_off"

    decision = {
        "recommended_policy": recommended_policy,
        "threshold_policy": {
            "success_improvement_threshold_pct": float(args.success_improvement_threshold_pct),
            "rule": "enable non-off policy only if success_gain_pct_vs_off >= threshold and > latency_cost_pct_vs_off",
        },
        "fixed_on_strength": float(max(0.0, args.fixed_on_strength)),
        "auto_config": {
            "on_threshold": float(args.auto_on_threshold),
            "off_threshold": float(args.auto_off_threshold),
            "ema_alpha": float(args.auto_ema_alpha),
            "strength_scale": float(args.auto_strength_scale),
        },
    }

    report = {
        "profile": {
            "name": "stress_only",
            "disturbances": {
                "sensor_noise_std": dist_cfg.sensor_noise_std,
                "stale_frame_prob": dist_cfg.stale_frame_prob,
                "actuator_delay_steps": dist_cfg.actuator_delay_steps,
                "random_perturb_prob": dist_cfg.perturb_prob,
                "degraded_obs_dropout_prob": dist_cfg.obs_dropout_prob,
            },
            "episodes": args.episodes,
            "seed": args.seed,
        },
        "off": off,
        "fixed_on": fixed_on,
        "auto": auto,
        "policy_comparison": policy_rows,
        "on_ablation": on_rows,
        "best_on": best_on,
        "decision": decision,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
