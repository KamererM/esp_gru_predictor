#!/usr/bin/env python3
"""Fast unit tests for split isolation, gap handling, and bootstrap output."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# The unit tests exercise data splitting and statistics and can run before the
# heavy TensorFlow dependency is installed. The real pipeline still requires
# the pinned TensorFlow package in requirements.txt.
try:
    import tensorflow  # noqa: F401
except ModuleNotFoundError:
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(
        utils=types.SimpleNamespace(
            register_keras_serializable=lambda **kwargs: (lambda cls: cls)
        ),
        layers=types.SimpleNamespace(Layer=object),
        Model=object,
    )
    sys.modules["tensorflow"] = fake_tf

from evaluate_esp32 import app_elf_sha256_from_image, request_prediction
from rerun import (
    Config,
    block_bootstrap,
    make_windows,
    moving_block_indices,
    normalize_float32,
    quantize,
    read_source,
    split_hourly,
)


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, body):
        self.body = body

    def post(self, url, json, timeout):
        return FakeResponse(self.body)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2007-01-01", periods=500, freq="h")
        self.frame = pd.DataFrame(
            {"energy_kwh": np.linspace(0.2, 2.5, len(index)), "count": 60},
            index=index,
        )

    def test_split_order_and_disjointness(self):
        parts = split_hourly(self.frame, Config())
        self.assertLess(parts["train"].index.max(), parts["validation"].index.min())
        self.assertLess(parts["validation"].index.max(), parts["test"].index.min())
        self.assertTrue(set(parts["train"].index).isdisjoint(parts["test"].index))

    def test_scaler_uses_training_only(self):
        parts = split_hourly(self.frame, Config())
        scaler = StandardScaler().fit(parts["train"][["energy_kwh"]].to_numpy())
        self.assertAlmostEqual(float(scaler.mean_[0]), parts["train"]["energy_kwh"].mean())
        self.assertNotAlmostEqual(float(scaler.mean_[0]), self.frame["energy_kwh"].mean())

    def test_windows_do_not_bridge_gap(self):
        frame = self.frame.iloc[:100].drop(self.frame.index[50])
        scaler = StandardScaler().fit(frame[["energy_kwh"]].to_numpy())
        _, _, _, targets, _ = make_windows(frame, scaler, steps=8)
        forbidden = pd.date_range(self.frame.index[51], periods=8, freq="h")
        self.assertTrue(set(pd.to_datetime(targets)).isdisjoint(forbidden))

    def test_hourly_energy_and_incomplete_hour_removal(self):
        timestamps = pd.date_range("2007-01-01", periods=120, freq="min")
        raw = pd.DataFrame(
            {
                "Date": timestamps.strftime("%d/%m/%Y"),
                "Time": timestamps.strftime("%H:%M:%S"),
                "Global_active_power": ["2.0"] * len(timestamps),
            }
        )
        raw.loc[90, "Global_active_power"] = "?"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            raw.to_csv(path, sep=";", index=False)
            hourly, audit = read_source(
                path, "2007-01-01 00:00:00", "2007-01-01 01:59:00"
            )
        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(float(hourly.iloc[0]["energy_kwh"]), 2.0)
        self.assertEqual(audit["incomplete_hourly_rows_removed"], 1)

    def test_bootstrap_is_deterministic(self):
        cfg = Config(bootstrap_resamples=50, bootstrap_block_length=6)
        y = np.linspace(1, 2, 50)
        first = y + 0.1
        second = y + 0.2
        timestamps = pd.date_range("2007-01-01", periods=50, freq="h").to_numpy()
        a = block_bootstrap(y, first, second, timestamps, cfg, "first minus second")
        b = block_bootstrap(y, first, second, timestamps, cfg, "first minus second")
        self.assertEqual(a, b)

    def test_quantization_uses_half_away_and_clips(self):
        values = np.array([-200.0, -1.5, -0.5, 0.5, 1.5, 200.0], dtype=np.float32)
        result = quantize(values, scale=1.0, zero_point=0)
        np.testing.assert_array_equal(result, [-128, -2, -1, 1, 2, 127])

    def test_float32_normalization_contract(self):
        scaler = StandardScaler().fit(
            np.array([[0.1], [0.2], [0.4]], dtype=np.float64)
        )
        values = np.array([[0.1, 0.4]], dtype=np.float32)
        result = normalize_float32(values, scaler)
        expected = (
            (values - np.float32(scaler.mean_[0])) / np.float32(scaler.scale_[0])
        ).astype(np.float32)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_bootstrap_blocks_never_cross_timestamp_gap(self):
        left = pd.date_range("2007-01-01", periods=12, freq="h")
        right = pd.date_range("2007-01-03", periods=12, freq="h")
        timestamps = left.append(right).to_numpy()
        rng = np.random.default_rng(7)
        indices = moving_block_indices(timestamps, block=6, rng=rng)
        sampled_timestamps = timestamps[indices]
        for start in range(0, len(indices), 6):
            block = sampled_timestamps[start : start + 6]
            self.assertTrue(np.all(np.diff(block) == np.timedelta64(1, "h")))

    def test_bootstrap_includes_short_segments(self):
        short = pd.date_range("2007-01-01", periods=4, freq="h")
        long = pd.date_range("2007-01-03", periods=10, freq="h")
        timestamps = short.append(long).to_numpy()
        indices = moving_block_indices(
            timestamps, block=6, rng=np.random.default_rng(7)
        )
        self.assertEqual(len(indices), len(timestamps))
        self.assertTrue(np.any(indices < len(short)))
        self.assertEqual(int(np.sum(indices < len(short))), len(short))

    def test_timestamp_csv_npz_round_trip(self):
        timestamps = pd.date_range("2007-01-01", periods=4, freq="h")
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "predictions.csv"
            npz_path = Path(directory) / "windows.npz"
            pd.DataFrame({"timestamp": timestamps}).to_csv(
                csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%S"
            )
            np.savez_compressed(
                npz_path,
                timestamp_ns=timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            )
            csv_ns = (
                pd.to_datetime(pd.read_csv(csv_path)["timestamp"])
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            npz_ns = np.load(npz_path)["timestamp_ns"]
        np.testing.assert_array_equal(csv_ns, npz_ns)

    def test_firmware_microsecond_timing_field(self):
        prediction, device_us, _, field, q_input, q_output = request_prediction(
            FakeSession(
                {
                    "prediction": 1.25,
                    "execution_time_us": 321,
                    "run_id": "run",
                    "timestamp_ns": 123,
                    "request_id": "request",
                    "input_int8": [1] * 32,
                    "output_int8": 2,
                }
            ),
            "http://device/data",
            np.ones(32, dtype=np.float32),
            timeout=1.0,
            retries=1,
            run_id="run",
            timestamp_ns=123,
            request_id="request",
        )
        self.assertEqual(prediction, 1.25)
        self.assertEqual(device_us, 321.0)
        self.assertEqual(field, "execution_time_us")
        np.testing.assert_array_equal(q_input, np.ones(32, dtype=np.int8))
        self.assertEqual(q_output, 2)

    def test_firmware_legacy_timing_field_is_explicit(self):
        with self.assertWarns(UserWarning):
            _, device_us, _, field, _, _ = request_prediction(
                FakeSession(
                    {
                        "prediction": 1.25,
                        "execution_time_ms": 321,
                        "run_id": "run",
                        "timestamp_ns": 123,
                        "request_id": "request",
                        "input_int8": [1] * 32,
                        "output_int8": 2,
                    }
                ),
                "http://device/data",
                np.ones(32, dtype=np.float32),
                timeout=1.0,
                retries=1,
                run_id="run",
                timestamp_ns=123,
                request_id="request",
            )
        self.assertEqual(device_us, 321.0)
        self.assertEqual(field, "execution_time_ms_legacy_microseconds")

    def test_firmware_nonfinite_prediction_is_rejected(self):
        body = {
            "prediction": float("nan"),
            "execution_time_us": 321,
            "run_id": "run",
            "timestamp_ns": 123,
            "request_id": "request",
            "input_int8": [1] * 32,
            "output_int8": 2,
        }
        with self.assertRaises(RuntimeError):
            request_prediction(
                FakeSession(body),
                "http://device/data",
                np.ones(32, dtype=np.float32),
                timeout=1.0,
                retries=1,
                run_id="run",
                timestamp_ns=123,
                request_id="request",
            )

    def test_firmware_fractional_int8_diagnostic_is_rejected(self):
        body = {
            "prediction": 1.25,
            "execution_time_us": 321,
            "run_id": "run",
            "timestamp_ns": "123",
            "request_id": "request",
            "input_int8": [1] * 31 + [1.5],
            "output_int8": 2,
        }
        with self.assertRaises(RuntimeError):
            request_prediction(
                FakeSession(body),
                "http://device/data",
                np.ones(32, dtype=np.float32),
                timeout=1.0,
                retries=1,
                run_id="run",
                timestamp_ns=123,
                request_id="request",
            )

    def test_firmware_boolean_int8_diagnostic_is_rejected(self):
        body = {
            "prediction": 1.25,
            "execution_time_us": 321,
            "run_id": "run",
            "timestamp_ns": "123",
            "request_id": "request",
            "input_int8": [1] * 32,
            "output_int8": True,
        }
        with self.assertRaises(RuntimeError):
            request_prediction(
                FakeSession(body),
                "http://device/data",
                np.ones(32, dtype=np.float32),
                timeout=1.0,
                retries=1,
                run_id="run",
                timestamp_ns=123,
                request_id="request",
            )

    def test_app_elf_sha_is_extracted_from_firmware_image(self):
        image = bytearray(256)
        image[32:36] = (0xABCD5432).to_bytes(4, "little")
        expected = bytes(range(32))
        image[176:208] = expected
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "app.bin"
            path.write_bytes(image)
            self.assertEqual(app_elf_sha256_from_image(path), expected.hex())


if __name__ == "__main__":
    unittest.main()
