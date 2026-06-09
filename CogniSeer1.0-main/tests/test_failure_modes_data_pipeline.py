import importlib.util
import json
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


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for data pipeline failure-mode tests")
class TestDataPipelineFailureModes(unittest.TestCase):
    def test_invalid_jsonl_line_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "bad.jsonl"
            processed = root / "processed"
            raw.write_text('{"record_id":"ok","observation":[0.1],"motor_target":[0.2]}\n{not-json}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                build_or_load_versioned_dataset(
                    raw_path=raw,
                    output_dir=processed,
                    dataset_name="robot",
                    input_dim=1,
                    motor_dim=1,
                    eval_ratio=0.2,
                )

    def test_observation_dimension_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "mismatch.jsonl"
            processed = root / "processed"

            rec = {
                "record_id": "r1",
                "timestamp_ms": 1710000000000,
                "observation": [0.1, 0.2],
                "motor_target": [0.3],
            }
            raw.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "observation length"):
                build_or_load_versioned_dataset(
                    raw_path=raw,
                    output_dir=processed,
                    dataset_name="robot",
                    input_dim=1,
                    motor_dim=1,
                    eval_ratio=0.2,
                )

    def test_invalid_eval_ratio_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            processed = root / "processed"

            rec = {
                "record_id": "r1",
                "timestamp_ms": 1710000000000,
                "observation": [0.1],
                "motor_target": [0.3],
            }
            raw.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "eval_ratio"):
                build_or_load_versioned_dataset(
                    raw_path=raw,
                    output_dir=processed,
                    dataset_name="robot",
                    input_dim=1,
                    motor_dim=1,
                    eval_ratio=1.0,
                )


if __name__ == "__main__":
    unittest.main()
