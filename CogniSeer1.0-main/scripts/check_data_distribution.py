#!/usr/bin/env python3
"""
Analyze and visualize robot telemetry dataset statistics.

Generates summary statistics, distribution plots, and warnings about
data adequacy for training.

Usage:
    python check_data_distribution.py \
        --input datasets/raw/robot_telemetry.jsonl \
        --output-dir /tmp/telemetry_analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_records(path: Path) -> List[Dict]:
    """Load JSONL telemetry records."""
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_no}: {e}")
    return records


def analyze_distribution(
    records: List[Dict],
) -> Dict:
    """
    Compute statistics on observation and motor target distributions.

    Returns:
        Dictionary with summary statistics
    """
    if not records:
        return {}

    observations = np.array([r["observation"] for r in records])
    motor_targets = np.array([r["motor_target"] for r in records])

    stats = {
        "total_samples": len(records),
        "observation_dim": observations.shape[1] if observations.ndim > 1 else 1,
        "motor_dim": motor_targets.shape[1] if motor_targets.ndim > 1 else 1,
        "observations": {
            "mean": observations.mean(axis=0).tolist(),
            "std": observations.std(axis=0).tolist(),
            "min": observations.min(axis=0).tolist(),
            "max": observations.max(axis=0).tolist(),
            "range": (observations.max() - observations.min()).item(),
        },
        "motor_targets": {
            "mean": motor_targets.mean(axis=0).tolist(),
            "std": motor_targets.std(axis=0).tolist(),
            "min": motor_targets.min(axis=0).tolist(),
            "max": motor_targets.max(axis=0).tolist(),
            "range": (motor_targets.max() - motor_targets.min()).item(),
        },
    }

    return stats


def check_adequacy(stats: Dict) -> Tuple[bool, List[str]]:
    """
    Check if dataset meets minimum requirements for training.

    Returns:
        (is_adequate, list_of_warnings)
    """
    warnings = []
    is_adequate = True

    n_samples = stats.get("total_samples", 0)
    obs_dim = stats.get("observation_dim", 0)
    motor_dim = stats.get("motor_dim", 0)

    # Check sample count
    minimum_samples = max(5000, obs_dim * motor_dim * 20)
    if n_samples < 1000:
        warnings.append(
            f"⚠️  CRITICAL: Only {n_samples} samples. Minimum viable: 1000. "
            f"Recommended: {minimum_samples}+."
        )
        is_adequate = False
    elif n_samples < 5000:
        warnings.append(
            f"⚠️  WARNING: Only {n_samples} samples. Risk of overfitting. "
            f"Target at least 5000."
        )
        is_adequate = False
    elif n_samples < minimum_samples:
        warnings.append(
            f"ℹ️  GUIDANCE: {n_samples} samples adequate, but recommend {minimum_samples}+ "
            f"for robust policy."
        )

    # Check observation variance
    obs_stats = stats.get("observations", {})
    obs_stds = obs_stats.get("std", [])
    if obs_stds:
        low_variance_dims = [i for i, s in enumerate(obs_stds) if s < 0.01]
        if low_variance_dims:
            warnings.append(
                f"⚠️  Low variance in observation dimensions: {low_variance_dims}. "
                f"May indicate single trajectory or static sensors."
            )

    # Check motor target variance
    motor_stats = stats.get("motor_targets", {})
    motor_stds = motor_stats.get("std", [])
    if motor_stds:
        low_variance_motors = [i for i, s in enumerate(motor_stds) if s < 0.01]
        if low_variance_motors:
            warnings.append(
                f"⚠️  Low variance in motor targets: {low_variance_motors}. "
                f"Limited diversity in control commands."
            )

    # Check range
    obs_range = obs_stats.get("range", 0)
    if obs_range < 0.5:
        warnings.append(
            f"⚠️  Narrow observation range ({obs_range:.2f}). "
            f"Data may not cover full sensor operating range."
        )

    return is_adequate, warnings


def print_report(stats: Dict, adequacy_ok: bool, warnings: List[str]):
    """Pretty-print analysis report."""
    print("\n" + "=" * 70)
    print("ROBOT TELEMETRY ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDataset Size:")
    print(f"  Total Samples: {stats.get('total_samples', 0):,}")
    print(f"  Observation Dimension: {stats.get('observation_dim', 0)}")
    print(f"  Motor Output Dimension: {stats.get('motor_dim', 0)}")

    print(f"\nObservation Statistics:")
    obs_stats = stats.get("observations", {})
    obs_mean = obs_stats.get("mean", [])
    obs_std = obs_stats.get("std", [])
    obs_min = obs_stats.get("min", [])
    obs_max = obs_stats.get("max", [])

    if obs_mean:
        print(f"  Mean (first 3):  {[f'{x:.4f}' for x in obs_mean[:3]]}")
        print(f"  Std  (first 3):  {[f'{x:.4f}' for x in obs_std[:3]]}")
        print(f"  Min  (first 3):  {[f'{x:.4f}' for x in obs_min[:3]]}")
        print(f"  Max  (first 3):  {[f'{x:.4f}' for x in obs_max[:3]]}")
        print(f"  Range: {obs_stats.get('range', 0):.4f}")

    print(f"\nMotor Target Statistics:")
    motor_stats = stats.get("motor_targets", {})
    motor_mean = motor_stats.get("mean", [])
    motor_std = motor_stats.get("std", [])
    motor_min = motor_stats.get("min", [])
    motor_max = motor_stats.get("max", [])

    if motor_mean:
        print(f"  Mean (first 3):  {[f'{x:.4f}' for x in motor_mean[:3]]}")
        print(f"  Std  (first 3):  {[f'{x:.4f}' for x in motor_std[:3]]}")
        print(f"  Min  (first 3):  {[f'{x:.4f}' for x in motor_min[:3]]}")
        print(f"  Max  (first 3):  {[f'{x:.4f}' for x in motor_max[:3]]}")
        print(f"  Range: {motor_stats.get('range', 0):.4f}")

    print(f"\nData Adequacy:")
    status = "✅ ADEQUATE" if adequacy_ok else "❌ INSUFFICIENT"
    print(f"  Status: {status}")

    if warnings:
        print(f"\nWarnings & Guidance:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print(f"\nNo warnings—dataset appears well-suited for training.")

    print("\n" + "=" * 70)


def save_report_json(stats: Dict, path: Path):
    """Save analysis results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved JSON report to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input JSONL telemetry file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for output reports",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    logger.info(f"Loading records from {args.input}...")
    records = load_records(args.input)

    if not records:
        logger.error("No records loaded")
        return 1

    logger.info("Analyzing distribution...")
    stats = analyze_distribution(records)

    adequacy_ok, warnings = check_adequacy(stats)

    print_report(stats, adequacy_ok, warnings)

    if args.output_dir:
        save_report_json(stats, args.output_dir / "distribution_analysis.json")

    return 0 if adequacy_ok else 1


if __name__ == "__main__":
    exit(main())
