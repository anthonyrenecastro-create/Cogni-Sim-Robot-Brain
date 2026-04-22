import json
import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from data_pipeline import build_or_load_versioned_dataset


def _write_jsonl(path: Path, n: int, input_dim: int, motor_dim: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "record_id": f"rec-{i:04d}",
                "timestamp_ms": 1710000000000 + i,
                "observation": [float((i + j) % 17) / 17.0 for j in range(input_dim)],
                "motor_target": [float((i + j) % 7) / 7.0 for j in range(motor_dim)],
            }
            f.write(json.dumps(rec) + "\n")


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for data pipeline regression tests")
class TestDataPipelineRegression(unittest.TestCase):
    def test_version_reuse_and_deterministic_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            processed = root / "processed"
            _write_jsonl(raw, n=50, input_dim=32, motor_dim=8)

            first = build_or_load_versioned_dataset(
                raw_path=raw,
                output_dir=processed,
                dataset_name="robot",
                input_dim=32,
                motor_dim=8,
                eval_ratio=0.2,
            )
            second = build_or_load_versioned_dataset(
                raw_path=raw,
                output_dir=processed,
                dataset_name="robot",
                input_dim=32,
                motor_dim=8,
                eval_ratio=0.2,
            )

            self.assertEqual(first.version, second.version)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(first.train_size, second.train_size)
            self.assertEqual(first.eval_size, second.eval_size)
            self.assertGreater(first.train_size, 0)
            self.assertGreater(first.eval_size, 0)

    def test_new_version_when_raw_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            processed = root / "processed"
            _write_jsonl(raw, n=30, input_dim=32, motor_dim=8)

            first = build_or_load_versioned_dataset(
                raw_path=raw,
                output_dir=processed,
                dataset_name="robot",
                input_dim=32,
                motor_dim=8,
                eval_ratio=0.25,
            )

            with raw.open("a", encoding="utf-8") as f:
                rec = {
                    "record_id": "new-sample",
                    "timestamp_ms": 1710000009999,
                    "observation": [0.1] * 32,
                    "motor_target": [0.2] * 8,
                }
                f.write(json.dumps(rec) + "\n")

            second = build_or_load_versioned_dataset(
                raw_path=raw,
                output_dir=processed,
                dataset_name="robot",
                input_dim=32,
                motor_dim=8,
                eval_ratio=0.25,
            )

            self.assertNotEqual(first.fingerprint, second.fingerprint)
            self.assertNotEqual(first.version, second.version)


if __name__ == "__main__":
    unittest.main()
