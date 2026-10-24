# Firmware

ESP-IDF application `gru_predictor_esp` (protocol `gru-esp32-v1`, endpoints `GET /metadata` and `POST /data` on port 8080), source: https://github.com/KamererM/gru_predictor_esp

| Build | Firmware commit | Image bytes | Image SHA-256 | ELF SHA-256 | Used for |
|---|---|---|---|---|---|
| Evaluated | `15e4a5015626e639635a8b003d11ad7c3532bd15` | 1,113,872 | `b83a809f2e92b19fd8200f75e86a85dd2ddaf74086c03b340c9344c011b20af0` | `d481c3e666baefe94cebee5cd767786f3c8afbb22108cd855665fef19c3f6924` | 477 test requests, Tables 2–5 |
| Diagnostic | `8f3c2928e83d671b8bcbdd6324f4795d89ac15fc` | 1,114,592 | `f75ce03289ad14006a5cd80bfc11c00dec742ecaa301a3712d75b8ae4b64540e` | `7b43bb9306614f450274b720de0e7ac0e9890c5fd041ad28f7252a020f3afeaf` | 128 golden-vector probes, resource values |

Both builds report the same model (`ced2bd04…`), C array (`b7ff54a4…`), preprocessing constants (`e5e06f19…`), sdkconfig (`81908759…`) and resolver source (`24158b7d…`). Full hashes: `device_evaluation/device_metadata_evaluated_build.json` and `diagnostic_probes/raw/device_metadata_diagnostic_build.json`.

Configuration: ESP32-S3 rev. v0.2, 160 MHz, 16 MiB flash, 2 MiB PSRAM; ESP-IDF v6.0 (dependency lock 6.0.0), espressif/esp-tflite-micro 1.4.1, ESP-NN 1.4.0 (`CONFIG_NN_OPTIMIZED=y`), cJSON 1.7.19; tensor arena 262,144 bytes in PSRAM. Component versions: `device_evaluation/dependencies.lock`.

## Source trees

- `evaluated_build/` — tree of commit `15e4a501` with the two generated files used for the build: `main/preprocessing_constants.h` (SHA-256 `e5e06f19…`; the commit itself contains placeholder constants) and `main/deployment_metadata.h` (firmware commit `15e4a501`, build ID `989e4179…`). Both files are identical to those committed in `8f3c2928`.
- `diagnostic_build/` — tree of commit `8f3c2928` (adds `main/resource_probe.h` and two lines in `main/http.cc`) with the generated `main/deployment_metadata.h` of the diagnostic build (build ID `17413611…`).
- `sdkconfig.redacted` — build-time configuration of both builds with Wi-Fi SSID and password blanked. The unredacted file has SHA-256 `81908759986cdb4de190ba53692f44f7a8408d7385076527cf0263e1dd576ec5`, equal to the value reported by both devices.
- `dependencies.lock` — ESP-IDF component lock.

In both trees, `main/wifi.cc` had two lines copying a hard-coded SSID and password into `wifi_config`; they were overwritten by the `CONFIG_GRU_WIFI_*` values and have been removed for publication. The model C array (`b7ff54a4…`) and the resolver source `main/gru_predictor.cc` (`24158b7d…`) are unchanged.

Timing: `execution_time_us` is measured in `main/http.cc` around `predict_gru()` only (standardization, quantization, `Invoke()`, dequantization, de-standardization), identical in both commits.

To rebuild, set the Wi-Fi credentials in `idf.py menuconfig` (or copy `sdkconfig.redacted` to `sdkconfig` and fill in the two values) and run `idf.py build` with ESP-IDF v6.0.

Do not publish an `sdkconfig` with real credentials.
