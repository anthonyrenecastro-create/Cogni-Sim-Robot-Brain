import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from jsonschema import Draft202012Validator
from torch.utils.data import Dataset


SCHEMA_VERSION = "1.0.0"


ROBOT_SAMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RobotTrainingSample",
    "type": "object",
    "required": ["observation", "motor_target"],
    "properties": {
        "record_id": {"type": "string", "minLength": 1},
        "timestamp_ms": {"type": "integer", "minimum": 0},
        "observation": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
        },
        "motor_target": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}


@dataclass
class DatasetArtifacts:
    dataset_root: Path
    version: str
    manifest_path: Path
    train_path: Path
    eval_path: Path
    train_size: int
    eval_size: int
    fingerprint: str


class TensorRobotDataset(Dataset):
    def __init__(self, dataset_path: Path):
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
        self.obs = payload["obs"]
        self.target = payload["target"]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int):
        return self.obs[idx], self.target[idx]


def build_or_load_versioned_dataset(
    raw_path: Path,
    output_dir: Path,
    dataset_name: str,
    input_dim: int,
    motor_dim: int,
    eval_ratio: float,
) -> DatasetArtifacts:
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. Provide JSONL/CSV robot telemetry."
        )
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        raise ValueError("eval_ratio must be between 0 and 1.")

    records = list(_read_records(raw_path))
    if not records:
        raise ValueError(f"No records found in raw dataset: {raw_path}")

    _validate_records(records, input_dim=input_dim, motor_dim=motor_dim)

    fingerprint = _fingerprint_dataset(
        raw_path=raw_path,
        schema_version=SCHEMA_VERSION,
        input_dim=input_dim,
        motor_dim=motor_dim,
        eval_ratio=eval_ratio,
    )

    dataset_root = output_dir / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_version(dataset_root=dataset_root, fingerprint=fingerprint)
    if existing is not None:
        return existing

    train_records, eval_records = _split_records(records, eval_ratio=eval_ratio)
    if not train_records or not eval_records:
        raise ValueError(
            "Split produced an empty train or eval set. Add more data or adjust eval_ratio."
        )

    version = _next_version(dataset_root)
    version_dir = dataset_root / version
    version_dir.mkdir(parents=True, exist_ok=False)

    train_obs, train_target = _to_tensors(train_records)
    eval_obs, eval_target = _to_tensors(eval_records)

    train_path = version_dir / "train.pt"
    eval_path = version_dir / "eval.pt"
    manifest_path = version_dir / "manifest.json"

    torch.save({"obs": train_obs, "target": train_target}, train_path)
    torch.save({"obs": eval_obs, "target": eval_target}, eval_path)

    manifest: Dict[str, object] = {
        "dataset_name": dataset_name,
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "input_dim": input_dim,
        "motor_dim": motor_dim,
        "eval_ratio": eval_ratio,
        "train_size": int(train_obs.shape[0]),
        "eval_size": int(eval_obs.shape[0]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dataset_root / "LATEST").write_text(version, encoding="utf-8")

    return DatasetArtifacts(
        dataset_root=dataset_root,
        version=version,
        manifest_path=manifest_path,
        train_path=train_path,
        eval_path=eval_path,
        train_size=int(train_obs.shape[0]),
        eval_size=int(eval_obs.shape[0]),
        fingerprint=fingerprint,
    )


def _read_records(raw_path: Path) -> Iterable[Dict[str, object]]:
    suffix = raw_path.suffix.lower()
    if suffix == ".jsonl":
        with raw_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no} in {raw_path}: {exc}") from exc
        return

    if suffix == ".csv":
        with raw_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_no, row in enumerate(reader, start=2):
                try:
                    observation = json.loads(row["observation"])
                    motor_target = json.loads(row["motor_target"])
                except KeyError as exc:
                    raise ValueError(
                        "CSV must include 'observation' and 'motor_target' columns encoded as JSON arrays."
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON array in CSV row {row_no}: {exc}") from exc

                record = {
                    "record_id": row.get("record_id") or f"row-{row_no}",
                    "timestamp_ms": int(row["timestamp_ms"]) if row.get("timestamp_ms") else None,
                    "observation": observation,
                    "motor_target": motor_target,
                }
                if record["timestamp_ms"] is None:
                    del record["timestamp_ms"]
                yield record
        return

    raise ValueError("Only .jsonl and .csv raw datasets are supported.")


def _validate_records(records: List[Dict[str, object]], input_dim: int, motor_dim: int):
    validator = Draft202012Validator(ROBOT_SAMPLE_SCHEMA)
    for idx, record in enumerate(records):
        errors = list(validator.iter_errors(record))
        if errors:
            msg = "; ".join(err.message for err in errors)
            raise ValueError(f"Schema validation failed for record {idx}: {msg}")

        obs = record["observation"]
        target = record["motor_target"]
        if len(obs) != input_dim:
            raise ValueError(
                f"Record {idx} has observation length {len(obs)} but expected {input_dim}."
            )
        if len(target) != motor_dim:
            raise ValueError(
                f"Record {idx} has motor_target length {len(target)} but expected {motor_dim}."
            )


def _fingerprint_dataset(
    raw_path: Path,
    schema_version: str,
    input_dim: int,
    motor_dim: int,
    eval_ratio: float,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(raw_path.read_bytes())
    hasher.update(schema_version.encode("utf-8"))
    hasher.update(str(input_dim).encode("utf-8"))
    hasher.update(str(motor_dim).encode("utf-8"))
    hasher.update(f"{eval_ratio:.6f}".encode("utf-8"))
    return hasher.hexdigest()


def _find_existing_version(dataset_root: Path, fingerprint: str):
    for manifest_path in sorted(dataset_root.glob("v*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("fingerprint") != fingerprint:
            continue

        version = str(manifest.get("version"))
        version_dir = manifest_path.parent
        train_path = version_dir / "train.pt"
        eval_path = version_dir / "eval.pt"
        if not train_path.exists() or not eval_path.exists():
            continue

        return DatasetArtifacts(
            dataset_root=dataset_root,
            version=version,
            manifest_path=manifest_path,
            train_path=train_path,
            eval_path=eval_path,
            train_size=int(manifest.get("train_size", 0)),
            eval_size=int(manifest.get("eval_size", 0)),
            fingerprint=fingerprint,
        )
    return None


def _next_version(dataset_root: Path) -> str:
    existing_versions = []
    for version_dir in dataset_root.glob("v*"):
        if not version_dir.is_dir():
            continue
        suffix = version_dir.name[1:]
        if suffix.isdigit():
            existing_versions.append(int(suffix))
    next_idx = (max(existing_versions) + 1) if existing_versions else 1
    return f"v{next_idx:03d}"


def _split_records(records: List[Dict[str, object]], eval_ratio: float):
    train_records = []
    eval_records = []

    for record in records:
        key = str(record.get("record_id") or json.dumps(record, sort_keys=True))
        frac = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if frac < eval_ratio:
            eval_records.append(record)
        else:
            train_records.append(record)

    if not eval_records:
        eval_records.append(train_records.pop())
    elif not train_records:
        train_records.append(eval_records.pop())

    return train_records, eval_records


def _to_tensors(records: List[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor]:
    obs = torch.tensor([rec["observation"] for rec in records], dtype=torch.float32)
    target = torch.tensor([rec["motor_target"] for rec in records], dtype=torch.float32)
    return obs, target