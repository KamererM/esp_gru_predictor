#!/usr/bin/env python3
"""Capture and validate live metadata from the flashed ESP32."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import requests


def values_equal(key: str, observed: object, expected: object) -> bool:
    if key in {"input_scale", "output_scale"}:
        try:
            return math.isclose(
                float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-7
            )
        except (TypeError, ValueError):
            return False
    return observed == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--static",
        type=Path,
        default=Path(__file__).parents[1] / "device_metadata_static.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    expected = json.loads(args.static.read_text(encoding="utf-8"))
    response = requests.get(args.url, timeout=args.timeout)
    response.raise_for_status()
    live = response.json()
    local_only = {"firmware_image_bytes", "firmware_image_sha256"}
    for key, value in expected.items():
        if key in local_only:
            continue
        if not values_equal(key, live.get(key), value):
            raise ValueError(
                f"Live metadata mismatch for {key}: "
                f"expected {value!r}, received {live.get(key)!r}"
            )
    for key in (
        "arena_used_bytes",
        "arena_capacity_bytes",
        "free_heap_before_allocate",
        "free_heap_after_allocate",
        "flash_bytes",
    ):
        if not isinstance(live.get(key), int) or live[key] <= 0:
            raise ValueError(f"Invalid runtime field: {key}")
    if not isinstance(live.get("psram_bytes"), int) or live["psram_bytes"] < 0:
        raise ValueError("Invalid runtime field: psram_bytes")
    if not isinstance(live.get("arena_in_psram"), bool):
        raise TypeError("Invalid runtime field: arena_in_psram")
    if live["arena_used_bytes"] > live["arena_capacity_bytes"]:
        raise ValueError("arena_used_bytes exceeds arena_capacity_bytes")
    if live["arena_in_psram"] and live["psram_bytes"] <= 0:
        raise ValueError("arena_in_psram is true while psram_bytes is zero")
    if live["free_heap_after_allocate"] < live["min_internal_free_heap_bytes"]:
        raise ValueError("Internal free heap is below the configured minimum")
    for key in ("chip_model", "chip_revision"):
        if not isinstance(live.get(key), int) or live[key] < 0:
            raise ValueError(f"Invalid runtime field: {key}")
    captured = dict(expected)
    captured.update(live)
    args.output.write_text(json.dumps(captured, indent=2), encoding="utf-8")
    print(json.dumps(captured, indent=2))


if __name__ == "__main__":
    main()
