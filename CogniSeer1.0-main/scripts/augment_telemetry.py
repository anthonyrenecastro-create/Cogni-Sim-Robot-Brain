#!/usr/bin/env python3
"""
Data augmentation utility for robot telemetry.

Expands small datasets through temporal interpolation, noise injection, and
trajectory scaling. Useful for reaching minimum viable sample counts before
collecting real data.

WARNING: Augmentation should supplement, not replace, genuine data collection.
Augmented data can overfit—aim to reach ~5K samples via augmentation, then
collect real data to reach 50K+.

Usage:
    python augment_telemetry.py \
        --input datasets/raw/robot_telemetry.jsonl \
        --output datasets/raw/robot_telemetry_augmented.jsonl \
        --multiplier 10 \
        --noise-std 0.02 \
        --seed 42
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


def save_records(records: List[Dict], path: Path):
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def interpolate_record(
    rec_a: Dict, rec_b: Dict, alpha: float, noise_std: float, rng: np.random.Generator
) -> Dict:
    """
    Interpolate between two records with noise injection.

    Args:
        rec_a: First record (earlier in trajectory)
        rec_b: Second record (later in trajectory)
        alpha: Interpolation weight [0, 1]. 0 = rec_a, 1 = rec_b
        noise_std: Standard deviation of Gaussian noise
        rng: NumPy random generator
    """
    obs_a = np.array(rec_a["observation"])
    obs_b = np.array(rec_b["observation"])
    motor_a = np.array(rec_a["motor_target"])
    motor_b = np.array(rec_b["motor_target"])

    # Linear interpolation
    obs = (1 - alpha) * obs_a + alpha * obs_b
    motor = (1 - alpha) * motor_a + alpha * motor_b

    # Add realistic noise
    obs = obs + noise_std * rng.standard_normal(obs.shape)
    motor = motor + (noise_std * 0.5) * rng.standard_normal(motor.shape)

    # Timestamp interpolation
    ts_a = rec_a.get("timestamp_ms", 0)
    ts_b = rec_b.get("timestamp_ms", ts_a + 20)  # Default 20ms interval
    ts = int((1 - alpha) * ts_a + alpha * ts_b)

    return {
        "record_id": f"{rec_a.get('record_id', 'aug')}-interp-{alpha:.2f}",
        "timestamp_ms": ts,
        "observation": obs.tolist(),
        "motor_target": motor.tolist(),
    }


def augment_via_interpolation(
    records: List[Dict], factor: int, noise_std: float, rng: np.random.Generator
) -> List[Dict]:
    """
    Create new records by interpolating between consecutive samples.

    Args:
        records: Input telemetry records (assumed sequential)
        factor: Number of interpolated samples per original pair
        noise_std: Noise standard deviation
        rng: Random generator
    """
    augmented = []
    for record in records:
        augmented.append(record)

    # Interpolate between consecutive pairs
    for i in range(len(records) - 1):
        rec_a = records[i]
        rec_b = records[i + 1]

        for j in range(1, factor + 1):
            alpha = j / (factor + 1)
            interp_rec = interpolate_record(rec_a, rec_b, alpha, noise_std, rng)
            augmented.append(interp_rec)

    return augmented


def augment_via_noise(
    records: List[Dict], noise_std: float, rng: np.random.Generator
) -> List[Dict]:
    """
    Create new records by adding sensor noise to existing samples.

    Args:
        records: Input telemetry
        noise_std: Observation noise standard deviation
        rng: Random generator
    """
    augmented = []
    for i, record in enumerate(records):
        augmented.append(record)

        obs = np.array(record["observation"])
        motor = np.array(record["motor_target"])

        # Create noisy variant
        obs_noisy = obs + noise_std * rng.standard_normal(obs.shape)
        motor_noisy = motor + (noise_std * 0.5) * rng.standard_normal(motor.shape)

        noisy_record = record.copy()
        noisy_record["record_id"] = f"{record.get('record_id', 'aug')}-noise-{i}"
        noisy_record["observation"] = obs_noisy.tolist()
        noisy_record["motor_target"] = motor_noisy.tolist()
        augmented.append(noisy_record)

    return augmented


def augment_via_scaling(
    records: List[Dict], factor: float, rng: np.random.Generator
) -> List[Dict]:
    """
    Create new records by scaling observation and motor baseline.

    Simulates similar tasks under different operating conditions (different
    baseline sensor readings, slightly different motor gain, etc.).

    Args:
        records: Input telemetry
        factor: Scale factor for baseline shifts (0.1 = ±10% shifts)
        rng: Random generator
    """
    augmented = []
    for i, record in enumerate(records):
        augmented.append(record)

        obs = np.array(record["observation"])
        motor = np.array(record["motor_target"])

        # Additive baseline shift (simulates sensor offset)
        obs_shift = obs + factor * rng.standard_normal(1) * np.ones_like(obs)
        motor_shift = motor + (factor * 0.5) * rng.standard_normal(1) * np.ones_like(motor)

        scaled_record = record.copy()
        scaled_record["record_id"] = f"{record.get('record_id', 'aug')}-scale-{i}"
        scaled_record["observation"] = obs_shift.tolist()
        scaled_record["motor_target"] = motor_shift.tolist()
        augmented.append(scaled_record)

    return augmented


def augment_dataset(
    records: List[Dict],
    multiplier: int,
    noise_std: float = 0.02,
    seed: int = 42,
) -> List[Dict]:
    """
    Augment dataset to reach target multiplier of original size.

    Combines interpolation, noise injection, and scaling to generate diverse
    synthetic samples.

    Args:
        records: Original telemetry records
        multiplier: Target expansion (e.g., 10 = 10× original size)
        noise_std: Gaussian noise standard deviation
        seed: Random seed for reproducibility

    Returns:
        Augmented dataset
    """
    rng = np.random.default_rng(seed)
    original_count = len(records)
    target_count = original_count * multiplier
    gap = target_count - original_count

    logger.info(f"Original: {original_count} samples")
    logger.info(f"Target: {target_count} samples (multiplier={multiplier}×)")
    logger.info(f"New samples needed: {gap}")

    # Strategy: allocate gap across interpolation, noise, and scaling
    # Roughly: 40% interpolation, 40% noise, 20% scaling
    interp_count = int(0.4 * gap)
    noise_count = int(0.4 * gap)
    scale_count = gap - interp_count - noise_count

    augmented = records.copy()

    # Phase 1: Interpolation (most faithful augmentation)
    if interp_count > 0:
        # Distribute interpolations across all pairs
        interp_factor = max(1, interp_count // (len(records) - 1))
        logger.info(f"  Interpolating with factor {interp_factor}...")
        interp_records = augment_via_interpolation(
            records, interp_factor, noise_std, rng
        )
        # Take first interp_count new records
        new_interp = [r for r in interp_records if r not in records][:interp_count]
        augmented.extend(new_interp)

    # Phase 2: Noise injection
    if noise_count > 0:
        logger.info(f"  Adding noise to {noise_count} copies...")
        # Repeat noise augmentation across original records
        repeats = max(1, noise_count // len(records))
        for _ in range(repeats):
            noise_records = augment_via_noise(records, noise_std, rng)
            new_noise = [r for r in noise_records if r not in records]
            augmented.extend(new_noise[: noise_count - len(augmented) + len(records)])

    # Phase 3: Scaling (least faithful but still useful)
    if scale_count > 0:
        logger.info(f"  Scaling with baseline shifts ({scale_count} samples)...")
        scale_records = augment_via_scaling(records, 0.1, rng)
        new_scale = [r for r in scale_records if r not in records][:scale_count]
        augmented.extend(new_scale)

    logger.info(f"Total after augmentation: {len(augmented)} samples")
    return augmented


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
        "--output",
        required=True,
        type=Path,
        help="Path to output augmented JSONL file",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=10,
        help="Target size multiplier (default: 10×)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise standard deviation (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
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
        logger.error("No records loaded from input file")
        return 1

    logger.info(f"Augmenting dataset (multiplier={args.multiplier}×)...")
    augmented = augment_dataset(
        records,
        multiplier=args.multiplier,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    logger.info(f"Saving to {args.output}...")
    save_records(augmented, args.output)

    logger.info(f"✓ Augmentation complete: {len(records)} → {len(augmented)} samples")
    logger.info(
        "⚠️  Remember: Augmentation should supplement real data collection, not replace it."
    )

    return 0


if __name__ == "__main__":
    exit(main())
