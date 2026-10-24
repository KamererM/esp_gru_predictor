"""Real TensorFlow smoke test for fixed-batch, fully integer conversion."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


HAS_REAL_TENSORFLOW = tf is not None and hasattr(getattr(tf, "keras", None), "Input")


@unittest.skipUnless(
    HAS_REAL_TENSORFLOW, "TensorFlow is not installed in this environment"
)
class TensorFlowIntegrationTests(unittest.TestCase):
    def test_fixed_batch_int8_conversion_and_inference(self):
        from rerun import Config, build_model, convert_int8, evaluate_tflite

        cfg = Config(steps=4, units=4, calibration_samples=8)
        rng = np.random.default_rng(cfg.seed)
        raw = rng.uniform(0.1, 2.0, size=(16, cfg.steps)).astype(np.float32)
        scaler = StandardScaler().fit(raw.reshape(-1, 1))
        normalized = (
            (raw - np.float32(scaler.mean_[0])) / np.float32(scaler.scale_[0])
        ).astype(np.float32)[..., None]

        model = build_model(cfg, batch_size=1)
        model(normalized[:1], training=False)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "tiny_int8.tflite"
            convert_int8(model, normalized, cfg, model_path)
            predictions, _, details, q_inputs, q_outputs = evaluate_tflite(
                model_path, raw[:3], scaler, warmup=0
            )

        self.assertEqual(details["input_shape_signature"], [1, cfg.steps, 1])
        self.assertEqual(details["float_tensor_count"], 0)
        self.assertEqual(q_inputs.dtype, np.int8)
        self.assertEqual(q_outputs.dtype, np.int8)
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
