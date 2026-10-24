# ESP-IDF 6.0.3 reference build

## Status

The firmware compiles successfully for target `esp32` with ESP-IDF 6.0.3.
Two clean builds produced byte-identical application and merged images.

This is a safe placeholder build. It contains no Wi-Fi SSID or password, and
`kDeploymentMetadataConfigured` remains `false`. It is therefore suitable for
validating compilation and flashing, but not for collecting manuscript
results. The leakage-free rerun artifacts, deployment metadata, and local
Wi-Fi configuration must be installed before experimental evaluation.

## Toolchain

- ESP-IDF: 6.0.3, commit `76f5dedd9950a3012fee8fb7d5586df21fc67802`
- Target: ESP32
- Xtensa compiler: `xtensa-esp-elf-g++ 15.2.0`
- Python: 3.12.13
- esptool: 5.4.0
- `espressif/esp-tflite-micro`: 1.4.1
- `espressif/esp-nn`: 1.4.0
- `espressif/cjson`: 1.7.19

## Image configuration

- Flash mode: DIO
- Flash frequency: 40 MHz
- Configured flash size: 2 MB
- Application partition: 1,048,576 bytes
- Application image: 533,520 bytes
- Free application partition space: 515,056 bytes
- Merged image: 599,056 bytes
- App version: `1.0.0-idf6`
- Reproducible-build mode: enabled
- Wi-Fi SSID/password: empty
- Embedded placeholder INT8 model: 158,760 bytes
- Model input/output: `[1,32,1]` INT8 / `[1,1]` INT8
- Embedded model SHA-256: `a725a880ad3baa63846c9e74c6cb9e398e5441911d9240f78095d5fb9bb9ffbf`

## Static memory report

| Region | Used | Available/total |
|---|---:|---:|
| Flash code | 250,046 B | n.a. |
| Flash data | 225,464 B | n.a. |
| IRAM | 46,859 B | 131,072 B |
| DRAM static data/BSS | 17,636 B | 180,736 B |
| RTC slow memory | 64 B | 8,192 B |

The 196,608-byte tensor arena is dynamically allocated at runtime and is not
included in the static DRAM figure. Physical-device startup must still confirm
successful tensor allocation and the configured 65,536-byte minimum internal
heap reserve.

## SHA-256

| Artifact | SHA-256 |
|---|---|
| `gru_predictor.bin` | `74dcd420f93e158065d07855ee5b03272a967b723efe6358c4ab27d67c549dba` |
| `gru_predictor_merged.bin` | `810a651c4d71989f194be8027d974fc0746ebeb93dfdf1b46bc41c72313143a0` |
| `gru_predictor.elf` / embedded ELF identity | `d90978654a5e58e8ac08e5fb88b4d032e45778f7570fc88da2c0c0496d84b8a6` |
| `bootloader.bin` | `d62b0e8aae6581761c334195df5da69fb3debea3efd213450408a2595763d256` |
| `partition-table.bin` | `7f00b6c042a89b15b0cac534f82ed988caf29278ff5700b0c511eb1b5bb7c820` |
| `dependencies.lock` | `b652cc726bd01f88804efb6c8627f77b03f25c64e931dcad2049edd4069e5979` |

## Flashing

The single merged image is written at offset zero:

```bash
python -m esptool \
  --chip esp32 \
  --port /dev/ttyUSB0 \
  --baud 460800 \
  --before default-reset \
  --after hard-reset \
  write-flash 0x0 gru_predictor_merged.bin
```

Alternatively, flash the three images:

```bash
python -m esptool \
  --chip esp32 \
  --port /dev/ttyUSB0 \
  --baud 460800 \
  --before default-reset \
  --after hard-reset \
  write-flash \
  --flash-mode dio \
  --flash-size 2MB \
  --flash-freq 40m \
  0x1000 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 gru_predictor.bin
```

## ESP-IDF 6.0 migration changes

- Updated the IDF dependency constraint to `>=6.0,<6.1`.
- Updated ESP-TFLite-Micro from 1.3.5 to 1.4.1.
- Replaced the removed legacy Mbed TLS SHA-256 call with PSA Crypto.
- Added explicit `esp_netif` dependency and header.
- Kept cJSON as an external managed component.
- Regenerated the lock file and project configuration under ESP-IDF 6.0.3.
- Enabled reproducible-build mode and assigned a stable application version.

## Required before experimental use

1. Run the leakage-free training and conversion pipeline.
2. Install its `model_data.*` and `preprocessing_constants.h`.
3. Generate final deployment metadata.
4. Configure Wi-Fi locally in `sdkconfig`.
5. Rebuild and archive the new binary and lock/configuration files.
6. Flash the device and verify `/metadata`.
7. Pass exact golden-vector INT8 input/output checks.
8. Run the full held-out device evaluation.
