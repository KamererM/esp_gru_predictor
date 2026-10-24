#!/usr/bin/env python3
"""Evaluate the flashed INT8 model on exactly the held-out test windows."""

from __future__ import annotations

import argparse
import json
import struct
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rerun import (
    Config,
    block_bootstrap,
    latency_summary,
    regression_metrics,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--url", required=True, help="Example: http://192.168.1.50:8080/data")
    parser.add_argument("--metadata-url", required=True)
    parser.add_argument("--device-metadata", required=True, type=Path)
    parser.add_argument("--firmware-binary", required=True, type=Path)
    parser.add_argument("--sdkconfig", required=True, type=Path)
    parser.add_argument("--resolver-source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--golden-tolerance-kwh", type=float, default=1e-4)
    parser.add_argument("--local-device-tolerance-kwh", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    return parser.parse_args()


def request_prediction(
    session: requests.Session,
    url: str,
    values: np.ndarray,
    timeout: float,
    retries: int,
    run_id: str,
    timestamp_ns: int,
    request_id: str,
    diagnostic: bool = True,
) -> tuple[float, float, float, str, np.ndarray, int]:
    payload = {
        "run_id": run_id,
        # JSON numbers cannot represent nanosecond epoch timestamps exactly.
        "timestamp_ns": str(int(timestamp_ns)),
        "request_id": request_id,
        "size": len(values),
        "data": [float(v) for v in values],
        "diagnostic": diagnostic,
    }
    last_error = None
    for attempt in range(retries):
        try:
            start = time.perf_counter_ns()
            response = session.post(url, json=payload, timeout=timeout)
            round_trip_us = (time.perf_counter_ns() - start) / 1000.0
            response.raise_for_status()
            body = response.json()
            # Existing firmware names this field *_ms, but esp_timer_get_time()
            # returns microseconds. Preserve the value and correct the unit here.
            if "execution_time_us" in body:
                device_time_us = float(body["execution_time_us"])
                timing_field = "execution_time_us"
            elif "execution_time_ms" in body:
                warnings.warn(
                    "Legacy execution_time_ms received. The archived firmware populates "
                    "this field with esp_timer_get_time() microseconds.",
                    stacklevel=2,
                )
                device_time_us = float(body["execution_time_ms"])
                timing_field = "execution_time_ms_legacy_microseconds"
            else:
                raise KeyError("Response contains no supported execution-time field.")
            prediction = float(body["prediction"])
            if body.get("run_id") != run_id:
                raise ValueError("Firmware response has a different run_id.")
            if str(body.get("timestamp_ns")) != str(int(timestamp_ns)):
                raise ValueError("Firmware response has a different timestamp_ns.")
            if body.get("request_id") != request_id:
                raise ValueError("Firmware response has a different request_id.")
            if not np.isfinite(prediction):
                raise ValueError("Firmware returned a non-finite prediction.")
            if not np.isfinite(device_time_us) or device_time_us < 0:
                raise ValueError("Firmware returned an invalid device time.")
            if not np.isfinite(round_trip_us) or round_trip_us < 0:
                raise ValueError("Client measured an invalid round-trip time.")
            raw_q_input = body["input_int8"]
            raw_q_output = body["output_int8"]
            if not isinstance(raw_q_input, list) or len(raw_q_input) != len(values):
                raise ValueError("Firmware returned an invalid input_int8 shape.")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -128 <= value <= 127
                for value in raw_q_input
            ):
                raise ValueError("Firmware returned out-of-range input_int8 values.")
            if (
                isinstance(raw_q_output, bool)
                or not isinstance(raw_q_output, int)
                or not -128 <= raw_q_output <= 127
            ):
                raise ValueError("Firmware returned an out-of-range output_int8 value.")
            q_input = np.asarray(raw_q_input, dtype=np.int8)
            q_output = raw_q_output
            return (
                prediction,
                device_time_us,
                float(round_trip_us),
                timing_field,
                q_input.astype(np.int8),
                q_output,
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"ESP32 request failed after {retries} attempts") from last_error


def fetch_device_metadata(
    session: requests.Session, url: str, timeout: float
) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    metadata = response.json()
    if not isinstance(metadata, dict):
        raise TypeError("Firmware metadata endpoint must return a JSON object.")
    return metadata

def app_elf_sha256_from_image(path: Path) -> str:
    image = path.read_bytes()
    app_desc_offset = 24 + 8
    sha_offset = app_desc_offset + 144
    if len(image) < sha_offset + 32:
        raise ValueError("Firmware image is too short to contain esp_app_desc_t.")
    if struct.unpack_from("<I", image, app_desc_offset)[0] != 0xABCD5432:
        raise ValueError("Firmware image has no esp_app_desc_t at the expected offset.")
    return image[sha_offset : sha_offset + 32].hex()


def metadata_values_equal(key: str, observed: object, expected: object) -> bool:
    if key in {"input_scale", "output_scale"}:
        try:
            return bool(
                np.isclose(
                    float(observed),
                    float(expected),
                    rtol=1e-6,
                    atol=np.finfo(np.float32).eps,
                )
            )
        except (TypeError, ValueError):
            return False
    return observed == expected


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    destination = (args.output or results_dir / "esp32_test_predictions.csv").resolve()
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    device_metadata = json.loads(args.device_metadata.read_text(encoding="utf-8"))
    cfg = Config(**manifest["config"])
    if args.bootstrap_resamples is not None:
        cfg = replace(cfg, bootstrap_resamples=args.bootstrap_resamples)

    artifacts = manifest["artifacts"]
    if sha256(results_dir / "model_int8.tflite") != artifacts["int8_model_sha256"]:
        raise ValueError("model_int8.tflite hash does not match run_manifest.json.")
    if sha256(results_dir / "model_data.cc") != artifacts["model_data_cc_sha256"]:
        raise ValueError("model_data.cc hash does not match run_manifest.json.")
    if (
        sha256(results_dir / "preprocessing_constants.h")
        != artifacts["preprocessing_constants_h_sha256"]
    ):
        raise ValueError(
            "preprocessing_constants.h hash does not match run_manifest.json."
        )
    if sha256(results_dir / "test_windows.npz") != artifacts["test_windows_sha256"]:
        raise ValueError("test_windows.npz hash does not match run_manifest.json.")
    if sha256(results_dir / "test_predictions.csv") != artifacts["test_predictions_sha256"]:
        raise ValueError("test_predictions.csv hash does not match run_manifest.json.")
    if sha256(results_dir / "golden_vectors.npz") != artifacts["golden_vectors_sha256"]:
        raise ValueError("golden_vectors.npz hash does not match run_manifest.json.")
    local_provenance = {
        "firmware_image_sha256": sha256(args.firmware_binary),
        "firmware_image_bytes": args.firmware_binary.stat().st_size,
        "app_elf_sha256": app_elf_sha256_from_image(args.firmware_binary),
        "sdkconfig_sha256": sha256(args.sdkconfig),
        "resolver_source_sha256": sha256(args.resolver_source),
    }
    required_device_values = {
        "protocol_version": "gru-esp32-v1",
        "run_id": manifest["run_id"],
        "model_sha256": artifacts["int8_model_sha256"],
        "model_bytes": artifacts["int8_model_bytes"],
        "model_data_cc_sha256": artifacts["model_data_cc_sha256"],
        "preprocessing_constants_h_sha256": artifacts[
            "preprocessing_constants_h_sha256"
        ],
        "training_mean_kwh_float32_hex": float(
            np.float32(manifest["scaler"]["mean_kwh"])
        ).hex(),
        "training_std_kwh_float32_hex": float(
            np.float32(manifest["scaler"]["std_kwh"])
        ).hex(),
        "input_scale": manifest["quantization"]["input_scale"],
        "input_scale_float32_bits": struct.pack(
            ">f", manifest["quantization"]["input_scale"]
        ).hex(),
        "input_zero_point": manifest["quantization"]["input_zero_point"],
        "output_scale": manifest["quantization"]["output_scale"],
        "output_scale_float32_bits": struct.pack(
            ">f", manifest["quantization"]["output_scale"]
        ).hex(),
        "output_zero_point": manifest["quantization"]["output_zero_point"],
    }
    for key, expected in required_device_values.items():
        if not metadata_values_equal(key, device_metadata.get(key), expected):
            raise ValueError(
                f"Device metadata mismatch for {key}: "
                f"expected {expected!r}, received {device_metadata.get(key)!r}"
            )
    declared_timing_field = device_metadata.get("timing_field")
    if declared_timing_field not in {
        "execution_time_us",
        "execution_time_ms_legacy_microseconds",
    }:
        raise ValueError(
            "Device metadata must declare timing_field as execution_time_us or "
            "execution_time_ms_legacy_microseconds."
        )
    for key in (
        "firmware_commit",
        "firmware_build_id",
        "board_or_module",
        "idf_target",
        "cpu_frequency_mhz",
        "flash_bytes",
        "psram_bytes",
        "tflm_revision",
        "operator_resolver_id",
        "arena_used_bytes",
        "arena_capacity_bytes",
        "arena_in_psram",
        "heap_measurement_caps",
        "min_internal_free_heap_bytes",
        "free_heap_before_allocate",
        "free_heap_after_allocate",
        "firmware_image_bytes",
        "firmware_image_sha256",
        "app_elf_sha256",
        "chip_model",
        "chip_revision",
        "sdkconfig_sha256",
        "resolver_source_sha256",
    ):
        value = device_metadata.get(key)
        if value is None or str(value).startswith("REPLACE_"):
            raise ValueError(f"Device metadata field {key} has not been completed.")
    numeric_positive = (
        "cpu_frequency_mhz",
        "flash_bytes",
        "arena_used_bytes",
        "arena_capacity_bytes",
        "min_internal_free_heap_bytes",
        "free_heap_before_allocate",
        "free_heap_after_allocate",
        "firmware_image_bytes",
    )
    for key in numeric_positive:
        if not isinstance(device_metadata[key], int) or device_metadata[key] <= 0:
            raise ValueError(f"Device metadata field {key} must be a positive integer.")
    if not isinstance(device_metadata["psram_bytes"], int) or device_metadata["psram_bytes"] < 0:
        raise ValueError("Device metadata field psram_bytes must be a nonnegative integer.")
    if not isinstance(device_metadata["arena_in_psram"], bool):
        raise TypeError("Device metadata field arena_in_psram must be boolean.")
    if device_metadata["arena_used_bytes"] > device_metadata["arena_capacity_bytes"]:
        raise ValueError("arena_used_bytes exceeds arena_capacity_bytes.")
    if device_metadata["arena_in_psram"] and device_metadata["psram_bytes"] <= 0:
        raise ValueError("arena_in_psram is true while psram_bytes is zero.")
    if (
        device_metadata["heap_measurement_caps"]
        != "MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT"
    ):
        raise ValueError("Heap metadata does not measure internal 8-bit memory.")
    if (
        device_metadata["free_heap_after_allocate"]
        < device_metadata["min_internal_free_heap_bytes"]
    ):
        raise ValueError("Internal free heap is below the configured minimum.")
    for key in ("chip_model", "chip_revision"):
        if not isinstance(device_metadata[key], int) or device_metadata[key] < 0:
            raise ValueError(f"Device metadata field {key} must be nonnegative integer.")
    if device_metadata["firmware_image_bytes"] != args.firmware_binary.stat().st_size:
        raise ValueError("Firmware binary size does not match device metadata.")
    if device_metadata.get("model_bytes") != artifacts["int8_model_bytes"]:
        raise ValueError("Model length does not match run manifest.")
    for key, expected in local_provenance.items():
        if device_metadata.get(key) != expected:
            raise ValueError(f"Local {key} does not match device metadata.")

    arrays = np.load(results_dir / "test_windows.npz")
    X_raw = arrays["X_raw_kwh"]
    y = arrays["target_kwh"].astype(float)
    timestamps_ns = arrays["timestamp_ns"].astype(np.int64)
    timestamps = pd.to_datetime(timestamps_ns)
    reference = pd.read_csv(results_dir / "test_predictions.csv")
    if len(reference) != len(X_raw):
        raise ValueError("Reference predictions and test windows have different lengths.")
    reference_timestamps_ns = (
        pd.to_datetime(reference["timestamp"])
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    if not np.array_equal(reference_timestamps_ns, timestamps_ns):
        raise ValueError("Timestamp mismatch between reference predictions and test windows.")
    if not np.allclose(
        reference["observed_kwh"].to_numpy(dtype=float), y, rtol=0.0, atol=1e-7
    ):
        raise ValueError("Target mismatch between reference predictions and test windows.")

    session = requests.Session()
    live_metadata = fetch_device_metadata(session, args.metadata_url, args.timeout)
    handshake_fields = set(required_device_values) | {
        "timing_field",
        "model_bytes",
        "preprocessing_constants_h_sha256",
        "model_data_cc_sha256",
        "firmware_commit",
        "firmware_build_id",
        "board_or_module",
        "idf_target",
        "cpu_frequency_mhz",
        "flash_bytes",
        "psram_bytes",
        "tflm_revision",
        "operator_resolver_id",
        "arena_used_bytes",
        "arena_capacity_bytes",
        "arena_in_psram",
        "heap_measurement_caps",
        "min_internal_free_heap_bytes",
        "free_heap_before_allocate",
        "free_heap_after_allocate",
        "app_elf_sha256",
        "chip_model",
        "chip_revision",
        "sdkconfig_sha256",
        "resolver_source_sha256",
    }
    for key in handshake_fields:
        if not metadata_values_equal(
            key, live_metadata.get(key), device_metadata.get(key)
        ):
            raise ValueError(
                f"Live firmware metadata mismatch for {key}: "
                f"expected {device_metadata.get(key)!r}, received {live_metadata.get(key)!r}"
            )
    golden = np.load(results_dir / "golden_vectors.npz")
    golden_X = golden["X_raw_kwh"]
    golden_X_int8 = golden["X_int8"].astype(np.int8)
    golden_output_int8 = golden["output_int8"].astype(np.int8)
    golden_expected = golden["output_kwh"].astype(float)
    golden_timestamps_ns = golden["timestamp_ns"].astype(np.int64)
    golden_observed = []
    timing_fields = set()
    for i, window in enumerate(golden_X):
        request_id = f"{manifest['run_id']}:golden:{i}:{golden_timestamps_ns[i]}"
        prediction, _, _, timing_field, q_input, q_output = request_prediction(
            session,
            args.url,
            window,
            args.timeout,
            args.retries,
            manifest["run_id"],
            int(golden_timestamps_ns[i]),
            request_id,
        )
        timing_fields.add(timing_field)
        golden_observed.append(prediction)
        if not np.array_equal(q_input, golden_X_int8[i]):
            raise ValueError(f"Golden-vector input INT8 mismatch at index {i}.")
        if q_output != int(golden_output_int8[i]):
            raise ValueError(f"Golden-vector output INT8 mismatch at index {i}.")
        if abs(prediction - golden_expected[i]) > args.golden_tolerance_kwh:
            raise ValueError(
                f"Golden-vector mismatch at index {i}: ESP32={prediction}, "
                f"local INT8={golden_expected[i]}, "
                f"tolerance={args.golden_tolerance_kwh} kWh."
            )
    if timing_fields != {declared_timing_field}:
        raise ValueError(
            "Firmware timing field does not match device metadata: "
            f"declared={declared_timing_field!r}, observed={sorted(timing_fields)!r}"
        )
    for i in range(args.warmup):
        request_prediction(
            session,
            args.url,
            golden_X[i % len(golden_X)],
            args.timeout,
            args.retries,
            manifest["run_id"],
            int(golden_timestamps_ns[i % len(golden_X)]),
            f"{manifest['run_id']}:warmup:{i}",
        )

    rows = []
    for i, window in enumerate(X_raw):
        request_id = f"{manifest['run_id']}:test:{i}:{timestamps_ns[i]}"
        (
            prediction,
            device_time_us,
            round_trip_us,
            timing_field,
            q_input,
            q_output,
        ) = request_prediction(
            session,
            args.url,
            window,
            args.timeout,
            args.retries,
            manifest["run_id"],
            int(timestamps_ns[i]),
            request_id,
        )
        timing_fields.add(timing_field)
        if timing_field != declared_timing_field:
            raise ValueError(
                "Firmware timing field changed or differs from device metadata: "
                f"declared={declared_timing_field!r}, observed={timing_field!r}"
            )
        rows.append(
            {
                "timestamp": timestamps[i].isoformat(),
                "observed_kwh": y[i],
                "fp32_kwh": float(reference.loc[i, "fp32_kwh"]),
                "int8_local_kwh": float(reference.loc[i, "int8_local_kwh"]),
                "esp32_kwh": prediction,
                "device_time_us": device_time_us,
                "http_round_trip_us": round_trip_us,
                "request_id": request_id,
                "input_int8_exact_match": bool(
                    np.array_equal(q_input, arrays["X_int8"][i])
                ),
                "output_int8_exact_match": bool(
                    q_output == int(arrays["output_int8"][i])
                ),
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == len(X_raw):
            pd.DataFrame(rows).to_csv(destination, index=False)
            print(f"{i + 1}/{len(X_raw)}")

    frame = pd.DataFrame(rows)
    if not frame["input_int8_exact_match"].all():
        raise ValueError("At least one held-out input differs at the INT8 byte level.")
    if not frame["output_int8_exact_match"].all():
        raise ValueError("At least one held-out output differs at the INT8 byte level.")
    local_device_abs = np.abs(
        frame["esp32_kwh"].to_numpy() - frame["int8_local_kwh"].to_numpy()
    )
    if float(np.max(local_device_abs)) > args.local_device_tolerance_kwh:
        raise ValueError(
            "ESP32 and local INT8 predictions exceed the pre-specified tolerance."
        )
    metrics = {
        "esp32": regression_metrics(
            frame["observed_kwh"].to_numpy(),
            frame["esp32_kwh"].to_numpy(),
            cfg.mape_epsilon_kwh,
        ),
        "device_model_call_latency": latency_summary(frame["device_time_us"].to_numpy()),
        "http_round_trip_latency": latency_summary(frame["http_round_trip_us"].to_numpy()),
        "device_timing_fields_observed": sorted(timing_fields),
        "run_id": manifest["run_id"],
        "firmware_commit": device_metadata["firmware_commit"],
        "golden_vector_count": len(golden_X),
        "golden_max_abs_difference_kwh": float(
            np.max(np.abs(np.asarray(golden_observed) - golden_expected))
        ),
        "all_test_input_int8_exact_match": True,
        "all_test_output_int8_exact_match": True,
        "local_device_max_abs_difference_kwh": float(np.max(local_device_abs)),
        "local_device_mean_abs_difference_kwh": float(np.mean(local_device_abs)),
        "local_device_tolerance_kwh": args.local_device_tolerance_kwh,
    }
    (results_dir / "esp32_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    bootstrap = {
        str(block): block_bootstrap(
            frame["observed_kwh"].to_numpy(),
            frame["esp32_kwh"].to_numpy(),
            frame["fp32_kwh"].to_numpy(),
            timestamps.to_numpy(dtype="datetime64[ns]"),
            cfg,
            "ESP32 INT8 minus FP32",
            block_length=block,
        )
        for block in (24, 48, 168)
    }
    (results_dir / "bootstrap_sensitivity_esp32_vs_fp32.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
