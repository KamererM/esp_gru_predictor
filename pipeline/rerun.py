#!/usr/bin/env python3
"""Leakage-free GRU training, INT8 conversion, and held-out evaluation.

The default date range reproduces the 181-day interval used in the manuscript.
The physical target is hourly energy in kWh:
    energy_kwh = sum(minute-level power in kW) / 60.

Incomplete hours are removed and windows are accepted only when all 33
timestamps (32 inputs plus one target) are exactly one hour apart.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import struct
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Config:
    seed: int = 20260916
    steps: int = 32
    units: int = 64
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    calibration_samples: int = 500
    bootstrap_resamples: int = 10000
    bootstrap_block_length: int = 48
    mape_epsilon_kwh: float = 1e-6
    subset_start: str = "2007-01-01 00:00:00"
    subset_end: str = "2007-06-30 23:59:00"


@tf.keras.utils.register_keras_serializable(package="LeakageFree")
class UnrolledGRU(tf.keras.layers.Layer):
    def __init__(self, units: int, timesteps: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.timesteps = int(timesteps)

    def build(self, input_shape):
        features = int(input_shape[-1])
        glorot = tf.keras.initializers.GlorotUniform
        orthogonal = tf.keras.initializers.Orthogonal
        zeros = tf.keras.initializers.Zeros()
        self.Wz = self.add_weight(
            shape=(features, self.units), name="Wz", initializer=glorot(seed=101)
        )
        self.Wr = self.add_weight(
            shape=(features, self.units), name="Wr", initializer=glorot(seed=102)
        )
        self.Wh = self.add_weight(
            shape=(features, self.units), name="Wh", initializer=glorot(seed=103)
        )
        self.Uz = self.add_weight(
            shape=(self.units, self.units), name="Uz", initializer=orthogonal(seed=201)
        )
        self.Ur = self.add_weight(
            shape=(self.units, self.units), name="Ur", initializer=orthogonal(seed=202)
        )
        self.Uh = self.add_weight(
            shape=(self.units, self.units), name="Uh", initializer=orthogonal(seed=203)
        )
        self.bz = self.add_weight(shape=(self.units,), name="bz", initializer=zeros)
        self.br = self.add_weight(shape=(self.units,), name="br", initializer=zeros)
        self.bh = self.add_weight(shape=(self.units,), name="bh", initializer=zeros)
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        h = tf.zeros((batch, self.units), dtype=inputs.dtype)
        for t in range(self.timesteps):
            x_t = inputs[:, t, :]
            z = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h, self.Uz) + self.bz)
            r = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h, self.Ur) + self.br)
            candidate = tf.tanh(
                tf.matmul(x_t, self.Wh) + tf.matmul(r * h, self.Uh) + self.bh
            )
            h = (1.0 - z) * h + z * candidate
        return h

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "timesteps": self.timesteps})
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--subset-start", default=Config.subset_start)
    parser.add_argument("--subset-end", default=Config.subset_end)
    parser.add_argument("--epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--bootstrap-resamples", type=int, default=Config.bootstrap_resamples)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate a nonempty output directory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility(seed: int) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            f"Publication run requires PYTHONHASHSEED={seed} before Python starts."
        )
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("Deterministic TensorFlow operations are required.") from exc


def read_source(path: Path, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        first_line = stream.readline()
    separator = ";" if ";" in first_line else ","
    raw = pd.read_csv(path, sep=separator, low_memory=False)
    required = {"Date", "Time", "Global_active_power"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Required columns missing: {sorted(required - set(raw.columns))}")

    parsed_timestamp = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    invalid_timestamp_count = int(parsed_timestamp.isna().sum())
    raw["timestamp"] = parsed_timestamp
    raw["power_kw"] = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    in_range = raw["timestamp"].between(start_ts, end_ts, inclusive="both")
    subset = raw.loc[
        in_range,
        ["timestamp", "power_kw"],
    ].copy()
    subset.sort_values("timestamp", inplace=True)

    duplicate_count = int(subset["timestamp"].duplicated().sum())
    missing_power_count = int(subset["power_kw"].isna().sum())
    subset.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    expected_minutes = pd.date_range(start_ts, end_ts, freq="min")
    indexed = subset.set_index("timestamp").reindex(expected_minutes)
    indexed.index.name = "timestamp"
    hourly = indexed["power_kw"].resample("1h").agg(["sum", "count"])
    hourly["energy_kwh"] = hourly["sum"] / 60.0
    valid_hourly = hourly.loc[hourly["count"] == 60, ["energy_kwh", "count"]].copy()
    valid_hourly.index.name = "timestamp"

    audit = {
        "source_rows_total": len(raw),
        "source_rows_in_requested_interval_before_deduplication": int(in_range.sum()),
        "expected_minute_rows_in_requested_interval": len(expected_minutes),
        "requested_subset_start": str(start_ts),
        "requested_subset_end": str(end_ts),
        "invalid_timestamps_in_source_file": invalid_timestamp_count,
        "duplicate_timestamps_in_requested_interval_removed": duplicate_count,
        "non_numeric_power_values_in_requested_interval": missing_power_count,
        "missing_minutes_after_reindex": int(indexed["power_kw"].isna().sum()),
        "candidate_hourly_rows": len(hourly),
        "complete_hourly_rows_retained": len(valid_hourly),
        "incomplete_hourly_rows_removed": int((hourly["count"] != 60).sum()),
    }
    if valid_hourly.empty:
        raise ValueError("No complete hourly observations remain after preprocessing.")
    return valid_hourly, audit


def split_hourly(hourly: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    fractions = cfg.train_fraction + cfg.validation_fraction + cfg.test_fraction
    if not np.isclose(fractions, 1.0):
        raise ValueError("Split fractions must sum to one.")
    n = len(hourly)
    train_end = int(np.floor(n * cfg.train_fraction))
    validation_end = int(np.floor(n * (cfg.train_fraction + cfg.validation_fraction)))
    splits = {
        "train": hourly.iloc[:train_end].copy(),
        "validation": hourly.iloc[train_end:validation_end].copy(),
        "test": hourly.iloc[validation_end:].copy(),
    }
    if min(map(len, splits.values())) <= cfg.steps:
        raise ValueError("Every split must contain more rows than the lookback window.")
    return splits


def make_windows(
    frame: pd.DataFrame, scaler: StandardScaler, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]")
    raw = frame["energy_kwh"].to_numpy(dtype=np.float32)
    mean32 = np.float32(scaler.mean_[0])
    std32 = np.float32(scaler.scale_[0])
    normalized = ((raw - mean32) / std32).astype(np.float32)
    X_norm, y_norm, y_raw, target_ts, X_raw = [], [], [], [], []
    one_hour = np.timedelta64(1, "h")
    for target in range(steps, len(frame)):
        start = target - steps
        candidate_ts = timestamps[start : target + 1]
        if not np.all(np.diff(candidate_ts) == one_hour):
            continue
        X_norm.append(normalized[start:target, None])
        y_norm.append(normalized[target])
        y_raw.append(raw[target])
        target_ts.append(timestamps[target])
        X_raw.append(raw[start:target])
    return (
        np.asarray(X_norm, dtype=np.float32),
        np.asarray(y_norm, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_raw, dtype=np.float64),
        np.asarray(target_ts),
        np.asarray(X_raw, dtype=np.float32),
    )


def build_model(cfg: Config, batch_size: int | None = None) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(cfg.steps, 1), batch_size=batch_size, name="hourly_energy"
    )
    hidden = UnrolledGRU(cfg.units, cfg.steps, name="unrolled_gru")(inputs)
    outputs = tf.keras.layers.Dense(1, name="forecast")(hidden)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def regression_metrics(y: np.ndarray, pred: np.ndarray, epsilon: float) -> dict:
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.abs(y) > epsilon
    return {
        "n": len(y),
        "mae_kwh": float(mean_absolute_error(y, pred)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(y, pred))),
        "mape_percent": float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100),
        "mape_excluded_count": int((~mask).sum()),
        "r2": float(r2_score(y, pred)),
    }


def calibration_indices(X_train: np.ndarray, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    flat = X_train.reshape(len(X_train), -1)
    extreme = np.flatnonzero(
        np.any(flat == flat.min(), axis=1) | np.any(flat == flat.max(), axis=1)
    )
    remaining = np.setdiff1d(np.arange(len(X_train)), extreme)
    random_count = min(cfg.calibration_samples, len(remaining))
    sampled = rng.choice(remaining, size=random_count, replace=False)
    return np.unique(np.concatenate([extreme, sampled])).astype(int)


def convert_int8(
    model: tf.keras.Model, X_train: np.ndarray, cfg: Config, output: Path
) -> tuple[bytes, np.ndarray]:
    indices = calibration_indices(X_train, cfg)

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for index in indices:
            yield [X_train[index : index + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    output.write_bytes(model_bytes)
    return model_bytes, indices


def normalize_float32(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    values32 = np.asarray(values, dtype=np.float32)
    mean32 = np.float32(scaler.mean_[0])
    std32 = np.float32(scaler.scale_[0])
    return ((values32 - mean32) / std32).astype(np.float32)


def denormalize_float32(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    values32 = np.asarray(values, dtype=np.float32)
    mean32 = np.float32(scaler.mean_[0])
    std32 = np.float32(scaler.scale_[0])
    return (values32 * std32 + mean32).astype(np.float32)


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def quantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    transformed = values / np.float32(scale) + np.float32(zero_point)
    quantized = round_half_away_from_zero(transformed)
    return np.clip(quantized, -128, 127).astype(np.int8)


def evaluate_tflite(
    model_path: Path, X_raw: np.ndarray, scaler: StandardScaler, warmup: int = 20
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    # interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=(
        tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    input_scale, input_zero = input_info["quantization"]
    output_scale, output_zero = output_info["quantization"]
    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise TypeError("The converted model is not fully INT8 at its input and output.")
    if tuple(input_info["shape"]) != (1, X_raw.shape[1], 1):
        raise ValueError(f"Unexpected deployment input shape: {input_info['shape']}")
    if tuple(input_info["shape_signature"]) != (1, X_raw.shape[1], 1):
        raise ValueError(
            "The deployment FlatBuffer does not have a fixed batch size of one: "
            f"{input_info['shape_signature']}"
        )
    float_tensors = [
        item["name"]
        for item in interpreter.get_tensor_details()
        if item["dtype"] in (np.float32, np.float64)
    ]
    if float_tensors:
        raise TypeError(f"Floating-point tensors remain in INT8 FlatBuffer: {float_tensors}")
    op_inventory = sorted({item["op_name"] for item in interpreter._get_ops_details()})
    tflm_resolver_ops = {
        "ADD",
        "DEQUANTIZE",
        "FILL",
        "FULLY_CONNECTED",
        "LOGISTIC",
        "MUL",
        "PACK",
        "QUANTIZE",
        "SHAPE",
        "STRIDED_SLICE",
        "SUB",
        "TANH",
        "UNPACK",
    }
    unsupported = sorted(set(op_inventory) - tflm_resolver_ops)
    if unsupported:
        raise ValueError(
            f"FlatBuffer contains operators absent from the documented TFLM resolver: {unsupported}"
        )
    X = normalize_float32(X_raw, scaler)[..., None]

    for i in range(min(warmup, len(X))):
        interpreter.set_tensor(input_info["index"], quantize(X[i : i + 1], input_scale, input_zero))
        interpreter.invoke()

    predictions_norm, latency_us, quantized_inputs, quantized_outputs = [], [], [], []
    for row in X:
        q_input = quantize(row[None, ...], input_scale, input_zero)
        quantized_inputs.append(q_input.ravel())
        interpreter.set_tensor(input_info["index"], q_input)
        start = time.perf_counter_ns()
        interpreter.invoke()
        latency_us.append((time.perf_counter_ns() - start) / 1000.0)
        q_output_raw = interpreter.get_tensor(output_info["index"])
        quantized_outputs.append(q_output_raw.ravel()[0])
        q_output = q_output_raw.astype(np.float32)
        predictions_norm.append((q_output.ravel()[0] - output_zero) * output_scale)
    predictions = denormalize_float32(predictions_norm, scaler).ravel()
    details = {
        "input_dtype": str(input_info["dtype"]),
        "output_dtype": str(output_info["dtype"]),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero),
        "operators": op_inventory,
        "float_tensor_count": len(float_tensors),
        "input_shape": input_info["shape"].astype(int).tolist(),
        "input_shape_signature": input_info["shape_signature"].astype(int).tolist(),
    }
    return (
        predictions.astype(np.float64),
        np.asarray(latency_us),
        details,
        np.asarray(quantized_inputs, dtype=np.int8),
        np.asarray(quantized_outputs, dtype=np.int8),
    )


def evaluate_fp32(
    model: tf.keras.Model, X: np.ndarray, scaler: StandardScaler, warmup: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    for i in range(min(warmup, len(X))):
        model(X[i : i + 1], training=False)
    predictions_norm, latency_us = [], []
    for row in X:
        start = time.perf_counter_ns()
        value = model(row[None, ...], training=False).numpy().ravel()[0]
        latency_us.append((time.perf_counter_ns() - start) / 1000.0)
        predictions_norm.append(value)
    predictions = denormalize_float32(predictions_norm, scaler).ravel()
    return predictions.astype(np.float64), np.asarray(latency_us)


def moving_block_indices(
    timestamps: np.ndarray, block: int, rng: np.random.Generator
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    n = len(timestamps)
    boundaries = np.flatnonzero(np.diff(timestamps) != np.timedelta64(1, "h")) + 1
    segments = np.split(np.arange(n), boundaries)
    sampled_segments = []
    for segment in segments:
        effective_block = min(block, len(segment))
        starts = segment[: len(segment) - effective_block + 1]
        count = int(np.ceil(len(segment) / effective_block))
        sampled_starts = rng.choice(starts, size=count, replace=True)
        sampled = np.concatenate(
            [np.arange(start, start + effective_block) for start in sampled_starts]
        )[: len(segment)]
        sampled_segments.append(sampled)
    return np.concatenate(sampled_segments)


def metric_differences(
    y: np.ndarray, first: np.ndarray, second: np.ndarray, epsilon: float
) -> dict[str, float]:
    mask = np.abs(y) > epsilon
    return {
        "mae_difference_kwh": float(
            np.mean(np.abs(y - first)) - np.mean(np.abs(y - second))
        ),
        "rmse_difference_kwh": float(
            np.sqrt(np.mean((y - first) ** 2))
            - np.sqrt(np.mean((y - second) ** 2))
        ),
        "mape_difference_percentage_points": float(
            100
            * (
                np.mean(np.abs((y[mask] - first[mask]) / y[mask]))
                - np.mean(np.abs((y[mask] - second[mask]) / y[mask]))
            )
        ),
    }


def block_bootstrap(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    timestamps: np.ndarray,
    cfg: Config,
    label: str,
    block_length: int | None = None,
) -> dict:
    rng = np.random.default_rng(cfg.seed + 1)
    effective_block = block_length or cfg.bootstrap_block_length
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    boundaries = np.flatnonzero(np.diff(timestamps) != np.timedelta64(1, "h")) + 1
    segment_lengths = [len(segment) for segment in np.split(timestamps, boundaries)]
    estimates = metric_differences(y, first, second, cfg.mape_epsilon_kwh)
    samples = {key: [] for key in estimates}
    for _ in range(cfg.bootstrap_resamples):
        idx = moving_block_indices(timestamps, effective_block, rng)
        values = metric_differences(
            y[idx], first[idx], second[idx], cfg.mape_epsilon_kwh
        )
        for key, value in values.items():
            samples[key].append(value)
    return {
        "contrast": label,
        "definition": "first minus second; negative favors first for error metrics",
        "method": "moving-block percentile bootstrap",
        "block_length_hours": effective_block,
        "segment_lengths_hours": segment_lengths,
        "short_segment_policy": (
            "segment-stratified resampling; effective block is "
            "min(requested block, segment length)"
        ),
        "all_segments_represented_in_each_resample": True,
        "resamples": cfg.bootstrap_resamples,
        "seed": cfg.seed + 1,
        "confidence_level": 0.95,
        "estimates": {
            key: {
                "point_estimate": estimates[key],
                "ci_low": float(np.percentile(samples[key], 2.5)),
                "ci_high": float(np.percentile(samples[key], 97.5)),
            }
            for key in estimates
        },
    }


def latency_summary(values_us: np.ndarray) -> dict:
    return {
        "n": len(values_us),
        "mean_ms": float(np.mean(values_us) / 1000.0),
        "median_ms": float(np.median(values_us) / 1000.0),
        "p95_ms": float(np.percentile(values_us, 95) / 1000.0),
    }


def write_c_artifacts(model_bytes: bytes, scaler: StandardScaler, out: Path) -> None:
    rows = []
    for start in range(0, len(model_bytes), 16):
        rows.append(
            "  " + ", ".join(f"0x{byte:02x}" for byte in model_bytes[start : start + 16])
        )
    (out / "model_data.h").write_text(
        "#pragma once\n#include <cstddef>\n#include <cstdint>\n"
        "extern const unsigned char optimized_model_tflite[];\n"
        "extern const int optimized_model_tflite_len;\n",
        encoding="utf-8",
    )
    model_rows = ",\n".join(rows)
    (out / "model_data.cc").write_text(
        '#include "model_data.h"\n'
        "alignas(16) const unsigned char optimized_model_tflite[] = {\n"
        f"{model_rows}\n"
        "};\n"
        f"const int optimized_model_tflite_len = {len(model_bytes)};\n",
        encoding="utf-8",
    )
    (out / "preprocessing_constants.h").write_text(
        "#pragma once\n"
        f"constexpr float kTrainingMeanKwh = {float(np.float32(scaler.mean_[0])).hex()}f;\n"
        f"constexpr float kTrainingStdKwh = {float(np.float32(scaler.scale_[0])).hex()}f;\n"
        "constexpr bool kPreprocessingConstantsArePlaceholder = false;\n"
        "/* Quantize with roundf and clamp to [-128, 127] before int8_t cast. */\n",
        encoding="utf-8",
    )


def package_versions() -> dict:
    names = ["numpy", "pandas", "scikit-learn", "tensorflow"]
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        max_epochs=args.epochs,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    set_reproducibility(cfg.seed)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out}. Use --overwrite explicitly."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    hourly, data_audit = read_source(args.input.resolve(), cfg.subset_start, cfg.subset_end)
    splits = split_hourly(hourly, cfg)
    scaler = StandardScaler().fit(splits["train"][["energy_kwh"]].to_numpy())

    prepared = {}
    for name in ("train", "validation"):
        frame = splits[name]
        prepared[name] = make_windows(frame, scaler, cfg.steps)
        if len(prepared[name][0]) == 0:
            raise ValueError(f"No valid contiguous windows in {name} split.")

    X_train, y_train, _, train_timestamps, X_train_raw = prepared["train"]
    X_validation, y_validation, _, _, _ = prepared["validation"]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .prefetch(1)
    )
    validation_ds = (
        tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
        .batch(cfg.batch_size)
        .prefetch(1)
    )

    model = build_model(cfg)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        restore_best_weights=True,
    )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=cfg.max_epochs,
        callbacks=[early_stopping],
        verbose=2,
    )
    model_path = out / "model_fp32.keras"
    model.save(model_path)
    pd.DataFrame(history.history).to_csv(out / "training_history.csv", index_label="epoch")

    # Freeze a deployment copy with batch size 1. Its weights are identical to
    # the validation-selected training model, but its FlatBuffer input shape is
    # static for TensorFlow Lite Micro.
    deployment_model = build_model(cfg, batch_size=1)
    deployment_model.set_weights(model.get_weights())
    tflite_path = out / "model_int8.tflite"
    model_bytes, calibration_idx = convert_int8(
        deployment_model, X_train, cfg, tflite_path
    )
    frozen_model_hashes = {
        "fp32_model_sha256": sha256(model_path),
        "int8_model_sha256": sha256(tflite_path),
    }

    # Verify serialization on validation data, then discard the in-memory
    # training model. Held-out test windows are materialized only afterward.
    frozen_fp32 = tf.keras.models.load_model(model_path)
    live_validation, _ = evaluate_fp32(model, X_validation, scaler, warmup=0)
    frozen_validation, _ = evaluate_fp32(
        frozen_fp32, X_validation, scaler, warmup=0
    )
    validation_reload_max_abs_kwh = float(
        np.max(np.abs(live_validation - frozen_validation))
    )
    if validation_reload_max_abs_kwh > 1e-7:
        raise ValueError(
            "Reloaded FP32 artifact differs from the in-memory model on validation data."
        )
    del model

    prepared["test"] = make_windows(splits["test"], scaler, cfg.steps)
    if len(prepared["test"][0]) == 0:
        raise ValueError("No valid contiguous windows in test split.")
    X_test, _, y_test, test_timestamps, X_test_raw = prepared["test"]
    fp32_pred, fp32_latency = evaluate_fp32(frozen_fp32, X_test, scaler)
    int8_pred, int8_latency, quantization, test_q_inputs, test_q_outputs = evaluate_tflite(
        tflite_path, X_test_raw, scaler
    )
    baseline_pred = X_test_raw[:, -1].astype(np.float64)

    prediction_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_timestamps),
            "observed_kwh": y_test,
            "persistence_kwh": baseline_pred,
            "fp32_kwh": fp32_pred,
            "int8_local_kwh": int8_pred,
        }
    )
    prediction_path = out / "test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_timestamps),
            "fp32_latency_us": fp32_latency,
            "int8_local_latency_us": int8_latency,
        }
    ).to_csv(
        out / "test_latency.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S"
    )
    windows_path = out / "test_windows.npz"
    np.savez_compressed(
        windows_path,
        X_raw_kwh=X_test_raw,
        X_int8=test_q_inputs,
        output_int8=test_q_outputs,
        target_kwh=y_test,
        timestamp_ns=test_timestamps.astype("datetime64[ns]").astype(np.int64),
    )
    golden_count = min(16, len(calibration_idx))
    calibration_positions = np.linspace(
        0, len(calibration_idx) - 1, golden_count, dtype=int
    )
    golden_indices = np.unique(
        np.concatenate(
            [
                calibration_idx[calibration_positions],
                np.linspace(
                    0,
                    len(X_train_raw) - 1,
                    min(16, len(X_train_raw)),
                    dtype=int,
                ),
            ]
        )
    )
    golden_pred, _, _, golden_q_inputs, golden_q_outputs = evaluate_tflite(
        tflite_path, X_train_raw[golden_indices], scaler, warmup=0
    )
    golden_path = out / "golden_vectors.npz"
    np.savez_compressed(
        golden_path,
        source_split=np.asarray(["train"]),
        X_raw_kwh=X_train_raw[golden_indices],
        X_int8=golden_q_inputs,
        output_int8=golden_q_outputs,
        output_kwh=golden_pred,
        timestamp_ns=train_timestamps[golden_indices]
        .astype("datetime64[ns]")
        .astype(np.int64),
    )

    split_export = []
    for name, frame in splits.items():
        part = frame.reset_index()
        part["split"] = name
        split_export.append(part)
    pd.concat(split_export, ignore_index=True).to_csv(out / "hourly_splits.csv", index=False)

    metrics = {
        "persistence": regression_metrics(y_test, baseline_pred, cfg.mape_epsilon_kwh),
        "fp32": regression_metrics(y_test, fp32_pred, cfg.mape_epsilon_kwh),
        "int8_local": regression_metrics(y_test, int8_pred, cfg.mape_epsilon_kwh),
        "latency": {
            "fp32_single_window": latency_summary(fp32_latency),
            "int8_local_single_window": latency_summary(int8_latency),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    bootstrap = {
        str(block): block_bootstrap(
            y_test,
            int8_pred,
            fp32_pred,
            test_timestamps,
            cfg,
            "INT8 local minus FP32",
            block_length=block,
        )
        for block in (24, 48, 168)
    }
    (out / "bootstrap_sensitivity_int8_vs_fp32.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    write_c_artifacts(model_bytes, scaler, out)

    split_summary = {}
    for name, frame in splits.items():
        windows = prepared[name][0]
        targets = prepared[name][3]
        split_summary[name] = {
            "hourly_rows": len(frame),
            "valid_windows": len(windows),
            "hourly_start": str(frame.index.min()),
            "hourly_end": str(frame.index.max()),
            "first_target": str(pd.Timestamp(targets.min())),
            "last_target": str(pd.Timestamp(targets.max())),
        }

    manifest = {
        "run_id": run_id,
        "config": asdict(cfg),
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256(args.input.resolve()),
        "data_audit": data_audit,
        "splits": split_summary,
        "leakage_controls": {
            "scaler_fit_split": "train",
            "early_stopping_split": "validation",
            "final_metrics_split": "test",
            "int8_calibration_split": "train",
            "windows_cross_split_boundaries": False,
            "test_windows_materialized_after_model_artifacts_hashed": True,
            "fp32_predictions_loaded_from_saved_artifact": True,
        },
        "scaler": {
            "mean_kwh": float(scaler.mean_[0]),
            "std_kwh": float(scaler.scale_[0]),
        },
        "calibration_window_count": len(calibration_idx),
        "quantization": quantization,
        "fp32_reload_validation_max_abs_difference_kwh": validation_reload_max_abs_kwh,
        "artifacts": {
            **frozen_model_hashes,
            "int8_model_bytes": len(model_bytes),
            "test_windows_sha256": sha256(windows_path),
            "test_predictions_sha256": sha256(prediction_path),
            "golden_vectors_sha256": sha256(golden_path),
            "model_data_cc_sha256": sha256(out / "model_data.cc"),
            "model_data_h_sha256": sha256(out / "model_data.h"),
            "preprocessing_constants_h_sha256": sha256(
                out / "preprocessing_constants.h"
            ),
            "requirements_sha256": sha256(Path(__file__).with_name("requirements.txt")),
            "pipeline_code_sha256": sha256(Path(__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "tensorflow_build": tf.sysconfig.get_build_info(),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    device_template = {
        "run_id": run_id,
        "model_sha256": frozen_model_hashes["int8_model_sha256"],
        "model_bytes": len(model_bytes),
        "model_data_cc_sha256": sha256(out / "model_data.cc"),
        "preprocessing_constants_h_sha256": sha256(
            out / "preprocessing_constants.h"
        ),
        "training_mean_kwh_float32_hex": float(
            np.float32(scaler.mean_[0])
        ).hex(),
        "training_std_kwh_float32_hex": float(np.float32(scaler.scale_[0])).hex(),
        "input_scale": quantization["input_scale"],
        "input_scale_float32_bits": struct.pack(
            ">f", quantization["input_scale"]
        ).hex(),
        "input_zero_point": quantization["input_zero_point"],
        "output_scale": quantization["output_scale"],
        "output_scale_float32_bits": struct.pack(
            ">f", quantization["output_scale"]
        ).hex(),
        "output_zero_point": quantization["output_zero_point"],
        "timing_field": "execution_time_us",
        "protocol_version": "gru-esp32-v1",
        "firmware_commit": "REPLACE_WITH_GIT_COMMIT",
        "firmware_build_id": "REPLACE_WITH_REPRODUCIBLE_BUILD_ID",
        "board_or_module": "REPLACE_WITH_EXACT_IDENTIFIER",
        "idf_target": "REPLACE_WITH_IDF_TARGET",
        "cpu_frequency_mhz": "REPLACE_WITH_POSITIVE_INTEGER",
        "flash_bytes": "REPLACE_WITH_POSITIVE_INTEGER",
        "psram_bytes": "REPLACE_WITH_NONNEGATIVE_INTEGER",
        "tflm_revision": "REPLACE_WITH_TFLM_REVISION",
        "operator_resolver_id": "REPLACE_WITH_RESOLVER_BUILD_ID",
        "chip_model": "REPLACE_WITH_RUNTIME_VALUE",
        "chip_revision": "REPLACE_WITH_RUNTIME_VALUE",
        "arena_capacity_bytes": "REPLACE_WITH_RUNTIME_VALUE",
        "arena_used_bytes": "REPLACE_WITH_RUNTIME_VALUE",
        "arena_in_psram": "REPLACE_WITH_RUNTIME_VALUE",
        "heap_measurement_caps": "MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT",
        "min_internal_free_heap_bytes": "REPLACE_WITH_CONFIG_VALUE",
        "free_heap_before_allocate": "REPLACE_WITH_RUNTIME_VALUE",
        "free_heap_after_allocate": "REPLACE_WITH_RUNTIME_VALUE",
        "firmware_image_bytes": "REPLACE_WITH_BUILD_VALUE",
        "firmware_image_sha256": "REPLACE_WITH_SHA256",
        "app_elf_sha256": "REPLACE_WITH_EMBEDDED_ELF_SHA256",
        "sdkconfig_sha256": "REPLACE_WITH_SHA256",
        "resolver_source_sha256": "REPLACE_WITH_SHA256",
    }
    (out / "device_metadata_template.json").write_text(
        json.dumps(device_template, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
