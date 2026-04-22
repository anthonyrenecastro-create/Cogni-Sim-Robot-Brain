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
    import torch
    from brain_torch import TorchBrainConfig, TorchEdgeRobotBrain
    from edge_runtime import EdgeRuntimeBrain
    from quantize_brain import (
        QuantizableBrainWrapper,
        _build_fp32_model,
        _quantize_wrapper,
        _save_bundle,
        _validate_rollout,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for export/load tests")
class TestExportLoad(unittest.TestCase):
    def test_bundle_export_and_runtime_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "robot_brain_fp32.pt"
            deploy_dir = root / "deploy"

            cfg = TorchBrainConfig(input_dim=32, hidden_dim=64, motor_dim=8)
            model = TorchEdgeRobotBrain(cfg)
            model.eval()
            transient = {
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
            stable = {k: v for k, v in model.state_dict().items() if k not in transient}
            torch.save(stable, checkpoint)

            fp32_model = _build_fp32_model(str(checkpoint), cfg)
            eager = QuantizableBrainWrapper(fp32_model)
            quant = _quantize_wrapper(QuantizableBrainWrapper(fp32_model))

            metrics = _validate_rollout(eager, quant, steps=8, seed=123)
            self.assertIn("max_abs_error", metrics)

            artifact, manifest = _save_bundle(
                quant,
                cfg,
                metrics,
                deploy_dir,
                checkpoint_path=str(checkpoint),
            )
            self.assertTrue(artifact.exists())
            self.assertTrue(manifest.exists())

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_format"], "quantized_eager_state_dict_v2")
            self.assertIn("model_signature", payload)

            runtime = EdgeRuntimeBrain(
                model_path=str(artifact),
                manifest_path=str(manifest),
                input_dim=32,
            )
            obs = [0.0] * runtime.input_dim
            out = runtime.step(obs, session_id="t1", reset_session=True)
            self.assertEqual(out.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
