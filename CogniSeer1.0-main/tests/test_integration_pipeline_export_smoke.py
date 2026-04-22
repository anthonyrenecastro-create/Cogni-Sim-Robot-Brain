import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _write_jsonl(path: Path, n: int, input_dim: int, motor_dim: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "record_id": f"sample-{i:04d}",
                "timestamp_ms": 1710000000000 + i * 20,
                "observation": [float((i + j) % 19) / 19.0 for j in range(input_dim)],
                "motor_target": [float((i + j) % 11) / 11.0 for j in range(motor_dim)],
            }
            f.write(json.dumps(rec) + "\n")


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for integration tests")
class TestIntegrationPipelineExportSmoke(unittest.TestCase):
    def test_train_export_and_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            processed = root / "processed"
            deploy = root / "deploy"
            checkpoint = root / "brain.pt"
            _write_jsonl(raw, n=96, input_dim=32, motor_dim=8)

            train_cmd = [
                sys.executable,
                str(ROOT / "train_brain.py"),
                "--raw-data",
                str(raw),
                "--dataset-dir",
                str(processed),
                "--dataset-name",
                "robot_test",
                "--eval-ratio",
                "0.2",
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--checkpoint",
                str(checkpoint),
            ]
            subprocess.run(train_cmd, cwd=str(ROOT), check=True)
            self.assertTrue(checkpoint.exists())

            export_cmd = [
                sys.executable,
                str(ROOT / "quantize_brain.py"),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(deploy),
                "--validate-steps",
                "8",
                "--max-abs-error",
                "1.0",
                "--seed",
                "123",
            ]
            subprocess.run(export_cmd, cwd=str(ROOT), check=True)

            smoke_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "deployment_smoke_test.py"),
                "--model-path",
                str(deploy / "robot_brain_int8_stateful.pt"),
                "--manifest-path",
                str(deploy / "deployment_manifest.json"),
                "--steps",
                "8",
                "--seed",
                "123",
                "--max-replay-error",
                "1e-4",
            ]
            subprocess.run(smoke_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    unittest.main()
