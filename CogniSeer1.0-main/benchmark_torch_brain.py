import argparse
import statistics
import time
import torch

from torch_edge_robot_brain import TorchBrainConfig, TorchEdgeRobotBrain


def run_single(mode_name: str, enable_chaos: bool, steps: int, warmup: int):
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
    model.reset_state(batch_size=1)

    x = torch.randn(1, cfg.input_dim)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            y = model(x)
    elapsed = time.perf_counter() - t0

    return {
        "mode": mode_name,
        "benchmark_steps": steps,
        "elapsed_sec": round(elapsed, 6),
        "steps_per_sec": round(steps / elapsed, 2),
        "ms_per_step": round(1000.0 * elapsed / steps, 4),
        "res_stability": round(float(y["res_stability"].mean()), 4),
        "res_variance": round(float(y["res_variance"].mean()), 6),
        "res_chaos_reduction": round(float(y["res_chaos_reduction"].mean()), 4),
        "res_lyapunov_proxy": round(float(y["res_lyapunov_proxy"].mean()), 4),
        "action_confidence": round(float(y["action_confidence"].mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CogniSeer torch brain")
    parser.add_argument("--mode", choices=["on", "off", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    def aggregate(mode_name: str, enable_chaos: bool):
        rows = [run_single(mode_name, enable_chaos, args.steps, args.warmup) for _ in range(args.runs)]
        if len(rows) == 1:
            return rows[0]
        sps = [r["steps_per_sec"] for r in rows]
        msp = [r["ms_per_step"] for r in rows]
        row = rows[-1].copy()
        row["steps_per_sec"] = round(statistics.mean(sps), 2)
        row["steps_per_sec_std"] = round(statistics.pstdev(sps), 2)
        row["ms_per_step"] = round(statistics.mean(msp), 4)
        row["ms_per_step_std"] = round(statistics.pstdev(msp), 4)
        row["runs"] = args.runs
        return row

    if args.mode == "on":
        print(aggregate("chaos_on", True))
        return

    if args.mode == "off":
        print(aggregate("chaos_off", False))
        return

    on = aggregate("chaos_on", True)
    off = aggregate("chaos_off", False)
    print(on)
    print(off)
    print(
        {
            "mode": "delta_on_minus_off",
            "steps_per_sec_pct": round(100.0 * (on["steps_per_sec"] - off["steps_per_sec"]) / off["steps_per_sec"], 3),
            "ms_per_step_pct": round(100.0 * (on["ms_per_step"] - off["ms_per_step"]) / off["ms_per_step"], 3),
            "res_stability_delta": round(on["res_stability"] - off["res_stability"], 4),
            "res_chaos_reduction_delta": round(on["res_chaos_reduction"] - off["res_chaos_reduction"], 4),
            "res_lyapunov_proxy_delta": round(on["res_lyapunov_proxy"] - off["res_lyapunov_proxy"], 4),
        }
    )


if __name__ == "__main__":
    main()
