#!/usr/bin/env python3
"""Install rerun artifacts and generate reproducible firmware metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_different(source: Path, destination: Path) -> None:
    if destination.exists() and source.read_bytes() == destination.read_bytes():
        return
    shutil.copy2(source, destination)


def write_text_if_different(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def cpp_string(value: str) -> str:
    return json.dumps(value)

def float32_hex(value: float) -> str:
    return float(struct.unpack("<f", struct.pack("<f", value))[0]).hex()

def float32_bits_hex(value: float) -> str:
    return struct.pack("<f", value)[::-1].hex()


def sdkconfig_value(text: str, name: str) -> str:
    match = re.search(rf"^CONFIG_{re.escape(name)}=(.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"CONFIG_{name} is absent from sdkconfig")
    return match.group(1).strip().strip('"')

def git_head(path: Path) -> str:
    root = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != path.resolve():
        raise ValueError(
            "firmware-root is not the root of a Git checkout; provide "
            "--firmware-commit explicitly when working from an archive"
        )
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("git rev-parse HEAD did not return a full commit SHA")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--firmware-root", default=Path(__file__).parents[1], type=Path)
    parser.add_argument(
        "--firmware-bin",
        type=Path,
        help="Final app .bin. Omit before the build; provide after the build.",
    )
    parser.add_argument("--firmware-commit")
    parser.add_argument("--board", required=True)
    parser.add_argument(
        "--tflm-revision", default="espressif/esp-tflite-micro@1.4.1"
    )
    parser.add_argument("--resolver-id", default="gru_resolver_v1")
    return parser.parse_args()

def app_elf_sha256_from_image(path: Path) -> str:
    """Extract esp_app_desc_t.app_elf_sha256 from an ESP-IDF app image."""
    image = path.read_bytes()
    app_desc_offset = 24 + 8
    sha_offset = app_desc_offset + 144
    if len(image) < sha_offset + 32:
        raise ValueError("Firmware image is too short to contain esp_app_desc_t")
    magic = struct.unpack_from("<I", image, app_desc_offset)[0]
    if magic != 0xABCD5432:
        raise ValueError(
            "esp_app_desc_t magic was not found at the standard app-image offset"
        )
    return image[sha_offset : sha_offset + 32].hex()


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    firmware_root = args.firmware_root.resolve()
    main_dir = firmware_root / "main"
    for name, value in {
        "board": args.board,
    }.items():
        if not value.strip() or value.startswith("REPLACE_"):
            raise ValueError(f"{name} must be a concrete non-placeholder value")
    firmware_commit = args.firmware_commit or git_head(firmware_root)
    if re.fullmatch(r"[0-9a-f]{40}", firmware_commit) is None:
        raise ValueError("firmware-commit must be a full 40-character lowercase Git SHA")
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]

    checked = {
        "model_int8.tflite": "int8_model_sha256",
        "model_data.cc": "model_data_cc_sha256",
        "model_data.h": "model_data_h_sha256",
        "preprocessing_constants.h": "preprocessing_constants_h_sha256",
    }
    for filename, manifest_key in checked.items():
        if sha256(results / filename) != artifacts[manifest_key]:
            raise ValueError(f"{filename} does not match run_manifest.json")

    for filename in ("model_data.cc", "model_data.h", "preprocessing_constants.h"):
        copy_if_different(results / filename, main_dir / filename)

    preprocessing = main_dir / "preprocessing_constants.h"
    text = preprocessing.read_text(encoding="utf-8")
    if "kPreprocessingConstantsArePlaceholder = false" not in text:
        raise ValueError(
            "preprocessing_constants.h is from an older rerun package. "
            "Regenerate the experiment artifacts before deployment."
        )

    sdkconfig = firmware_root / "sdkconfig"
    sdkconfig_text = sdkconfig.read_text(encoding="utf-8")
    resolver_source = main_dir / "gru_predictor.cc"
    static_metadata = {
        "protocol_version": "gru-esp32-v1",
        "run_id": manifest["run_id"],
        "model_sha256": artifacts["int8_model_sha256"],
        "model_bytes": artifacts["int8_model_bytes"],
        "model_data_cc_sha256": sha256(main_dir / "model_data.cc"),
        "preprocessing_constants_h_sha256": sha256(preprocessing),
        "training_mean_kwh_float32_hex": float32_hex(
            manifest["scaler"]["mean_kwh"]
        ),
        "training_std_kwh_float32_hex": float32_hex(
            manifest["scaler"]["std_kwh"]
        ),
        "input_scale": manifest["quantization"]["input_scale"],
        "input_scale_float32_bits": float32_bits_hex(
            manifest["quantization"]["input_scale"]
        ),
        "input_zero_point": manifest["quantization"]["input_zero_point"],
        "output_scale": manifest["quantization"]["output_scale"],
        "output_scale_float32_bits": float32_bits_hex(
            manifest["quantization"]["output_scale"]
        ),
        "output_zero_point": manifest["quantization"]["output_zero_point"],
        "timing_field": "execution_time_us",
        "firmware_commit": firmware_commit,
        "board_or_module": args.board,
        "idf_target": sdkconfig_value(sdkconfig_text, "IDF_TARGET"),
        "cpu_frequency_mhz": int(
            sdkconfig_value(sdkconfig_text, "ESP_DEFAULT_CPU_FREQ_MHZ")
        ),
        "tflm_revision": args.tflm_revision,
        "operator_resolver_id": args.resolver_id,
        "arena_capacity_bytes": int(
            sdkconfig_value(sdkconfig_text, "GRU_TENSOR_ARENA_BYTES")
        ),
        "heap_measurement_caps": "MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT",
        "min_internal_free_heap_bytes": int(
            sdkconfig_value(sdkconfig_text, "GRU_MIN_FREE_HEAP_AFTER_MODEL")
        ),
        "sdkconfig_sha256": sha256(sdkconfig),
        "resolver_source_sha256": sha256(resolver_source),
    }
    build_identity = json.dumps(static_metadata, sort_keys=True).encode("utf-8")
    static_metadata["firmware_build_id"] = hashlib.sha256(build_identity).hexdigest()

    header = f"""#pragma once
#include <cstddef>

constexpr bool kDeploymentMetadataConfigured = true;
constexpr char kRunId[] = {cpp_string(static_metadata["run_id"])};
constexpr char kModelSha256[] = {cpp_string(static_metadata["model_sha256"])};
constexpr char kModelDataCcSha256[] = {cpp_string(static_metadata["model_data_cc_sha256"])};
constexpr char kPreprocessingConstantsSha256[] = {cpp_string(static_metadata["preprocessing_constants_h_sha256"])};
constexpr char kTrainingMeanFloat32Hex[] = {cpp_string(static_metadata["training_mean_kwh_float32_hex"])};
constexpr char kTrainingStdFloat32Hex[] = {cpp_string(static_metadata["training_std_kwh_float32_hex"])};
constexpr float kExpectedInputScale = {float(static_metadata["input_scale"]).hex()}f;
constexpr int kExpectedInputZeroPoint = {int(static_metadata["input_zero_point"])};
constexpr float kExpectedOutputScale = {float(static_metadata["output_scale"]).hex()}f;
constexpr int kExpectedOutputZeroPoint = {int(static_metadata["output_zero_point"])};
constexpr char kFirmwareCommit[] = {cpp_string(static_metadata["firmware_commit"])};
constexpr char kFirmwareBuildId[] = {cpp_string(static_metadata["firmware_build_id"])};
constexpr char kBoardOrModule[] = {cpp_string(args.board)};
constexpr char kIdfTarget[] = {cpp_string(static_metadata["idf_target"])};
constexpr int kCpuFrequencyMhz = {static_metadata["cpu_frequency_mhz"]};
constexpr char kTflmRevision[] = {cpp_string(args.tflm_revision)};
constexpr char kOperatorResolverId[] = {cpp_string(args.resolver_id)};
constexpr char kSdkconfigSha256[] = {cpp_string(static_metadata["sdkconfig_sha256"])};
constexpr char kResolverSourceSha256[] = {cpp_string(static_metadata["resolver_source_sha256"])};
"""
    write_text_if_different(main_dir / "deployment_metadata.h", header)
    if args.firmware_bin is not None:
        firmware_bin = args.firmware_bin.resolve()
        static_metadata.update(
            {
                "firmware_image_bytes": firmware_bin.stat().st_size,
                "firmware_image_sha256": sha256(firmware_bin),
                "app_elf_sha256": app_elf_sha256_from_image(firmware_bin),
            }
        )
    write_text_if_different(
        firmware_root / "device_metadata_static.json",
        json.dumps(static_metadata, indent=2),
    )
    print(json.dumps(static_metadata, indent=2))


if __name__ == "__main__":
    main()
