#!/usr/bin/env python3
"""
Utility to merge multiple JSONL telemetry files.

Useful for combining telemetry from multiple robot runs or trajectories.
Handles deduplication, sorting, and record ID remapping.

Usage:
    python merge_telemetry.py \
        --inputs datasets/raw/trajectory_*.jsonl \
        --output datasets/raw/robot_telemetry_merged.jsonl \
        --sort-by timestamp_ms

    # Or with explicit file list
    python merge_telemetry.py \
        --inputs datasets/raw/run1.jsonl datasets/raw/run2.jsonl \
        --output datasets/raw/combined.jsonl
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List

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
                logger.warning(f"Skipping line {line_no} in {path}: {e}")
    return records


def save_records(records: List[Dict], path: Path):
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def merge_records(
    file_paths: List[Path],
    deduplicate: bool = True,
    sort_by: str = None,
    remap_ids: bool = True,
) -> List[Dict]:
    """
    Merge records from multiple JSONL files.

    Args:
        file_paths: List of input file paths
        deduplicate: Remove exact duplicate records
        sort_by: Field to sort by (e.g., 'timestamp_ms'), or None to preserve order
        remap_ids: Remap record_id to maintain uniqueness across merges

    Returns:
        Merged list of records
    """
    all_records = []

    for path in file_paths:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        logger.info(f"Loading {path}...")
        records = load_records(path)
        logger.info(f"  → {len(records)} records")
        all_records.extend(records)

    logger.info(f"Total loaded: {len(all_records)} records")

    # Deduplication
    if deduplicate:
        before = len(all_records)
        # Create set of JSON strings to identify duplicates
        seen = set()
        unique_records = []
        for record in all_records:
            # Use a tuple of key fields as hash
            key = (
                str(record.get("observation", [])),
                str(record.get("motor_target", [])),
                record.get("timestamp_ms", -1),
            )
            key_str = json.dumps(key, sort_keys=True)
            if key_str not in seen:
                seen.add(key_str)
                unique_records.append(record)

        logger.info(f"Deduplicated: {before} → {len(unique_records)} records")
        all_records = unique_records

    # Sorting
    if sort_by:
        logger.info(f"Sorting by '{sort_by}'...")
        all_records.sort(
            key=lambda r: (r.get(sort_by) is not None, r.get(sort_by))
        )

    # Remap IDs
    if remap_ids:
        logger.info("Remapping record IDs...")
        for i, record in enumerate(all_records, 1):
            record["record_id"] = f"merged-{i:06d}"

    logger.info(f"Final merged dataset: {len(all_records)} records")
    return all_records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files (supports glob patterns)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output merged JSONL file",
    )
    parser.add_argument(
        "--sort-by",
        default="timestamp_ms",
        help="Sort by field (default: timestamp_ms, set to None to skip sorting)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't remove duplicate records",
    )
    parser.add_argument(
        "--no-remap-ids",
        action="store_true",
        help="Don't remap record IDs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    # Expand glob patterns
    all_paths = []
    for pattern in args.inputs:
        expanded = glob.glob(pattern)
        if expanded:
            all_paths.extend([Path(p) for p in expanded])
        else:
            # Try as literal path
            p = Path(pattern)
            if p.exists():
                all_paths.append(p)
            else:
                logger.warning(f"No files matched: {pattern}")

    if not all_paths:
        logger.error("No input files found")
        return 1

    logger.info(f"Found {len(all_paths)} input files")

    merged = merge_records(
        all_paths,
        deduplicate=not args.no_deduplicate,
        sort_by=args.sort_by if args.sort_by != "None" else None,
        remap_ids=not args.no_remap_ids,
    )

    if not merged:
        logger.error("No records after merge")
        return 1

    logger.info(f"Saving to {args.output}...")
    save_records(merged, args.output)

    logger.info(f"✓ Merge complete: {len(all_paths)} files → {len(merged)} records")
    return 0


if __name__ == "__main__":
    exit(main())
