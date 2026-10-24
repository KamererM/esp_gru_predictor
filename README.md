# Leakage-free INT8 GRU on ESP32-S3 — data, code, and device logs

Companion repository for the manuscript

> M. A. Barulina, M. E. Pedaev. *Leakage-free evaluation of an INT8-quantized GRU for hourly household electricity forecasting on an ESP32-S3 microcontroller: a technical case study.* Submitted to PLOS ONE (2026).

It contains everything needed to verify the reported numbers: the derived hourly data, the training/conversion/evaluation pipeline, all artifacts of the deployed run (models, C array, manifest, predictions), device metadata and logs of the evaluated and diagnostic firmware builds, the audit and control analyses, the multi-seed training study, and scripts that regenerate the data figures.

Run identifier of the deployed model: `56f38c60-e7c8-4ba0-9d07-c6a184fe6680`; INT8 model SHA-256 `ced2bd0441434481f8813e80dca49931222a8eaebd76b49f67a8cac6e9318bca` (146,168 bytes).

## Repository layout

| Folder | Content |
|---|---|
| `data/` | Hourly energy series derived from the UCI dataset with split labels (`hourly_splits.csv`, Jan–Jun 2007), the additional July–September period (`extension_hourly.csv`) and its 2,008 windows (`extension_windows.npz`). See `data/README.md`. |
| `pipeline/` | Historical `rerun.py` and `requirements.txt` (hashes match the run manifest): preprocessing, chronological split, training, INT8 conversion, held-out evaluation, block bootstrap; `evaluate_esp32.py` (device client as used) and `evaluate_esp32_prespecified.py`; unit and TensorFlow integration tests (17/17 pass). |
| `primary_run/` | Artifacts of the deployed run: `run_manifest.json`, `model_fp32.keras`, `model_int8.tflite`, `model_data.cc/.h`, `preprocessing_constants.h`, `training_history.csv`, `test_windows.npz`, `golden_vectors.npz`, `test_predictions.csv` (FP32, local INT8), `esp32_test_predictions.csv` (paired on-device outputs incl. raw INT8 input/output), `test_latency.csv`, `metrics.json`, `bootstrap_sensitivity_int8_vs_fp32.json`. |
| `device_evaluation/` | Device metadata of the evaluated firmware build (commit `15e4a501`, image 1,113,872 bytes) and the ESP-IDF component lock file. |
| `audit_and_controls/` | Artifact audit against the UCI source (`audit.py`), local kernel trace (`kernel_trace.py`), re-conversion check (`reconvert_check.py`), paired moving-block bootstrap (`paired_bootstrap.py`), the analysis plan fixed before the additional period was analysed (`plan.json`), and all results incl. ridge and persistence controls. |
| `diagnostic_probes/` | Golden-vector probes of the diagnostic build (commit `8f3c2928`, image 1,114,592 bytes): raw responses and metadata snapshots of both client sessions (`raw/`), probe client (`device_probe.py`), resource probe for the firmware (`resource_probe.h`), and `analyze_probes.py` → `probe_analysis.json`. |
| `multiseed/` | Training-variability study with 10 additional seeds plus a bit-exact retraining of the primary configuration (`run_seed.py`, `rp.py`, per-run outputs, `multiseed_summary.csv`). |
| `figures/` | `make_figures.py` regenerates Fig 1, Fig 4 and Fig 5. |
| `firmware/` | ESP-IDF source of the evaluated and diagnostic builds (credentials removed), redacted build-time `sdkconfig`, component lock. See `firmware/README.md`. |
| `environment/` | `requirements-lock.txt` (exact package versions of the analysis environment). |

## Mapping to the manuscript

| Manuscript item | Source files |
|---|---|
| Table 1, Fig 1 (data partition) | `primary_run/run_manifest.json`, `data/hourly_splits.csv`, `data/extension_hourly.csv` |
| Table 2 (test errors) | `audit_and_controls/results/all_metrics.csv`, `audit_and_controls/results/test_all_predictions.csv`, `primary_run/esp32_test_predictions.csv` |
| Table 3 (paired differences, bootstrap) | `audit_and_controls/results/paired_bootstrap.json`, `primary_run/bootstrap_sensitivity_int8_vs_fp32.json` |
| Fig 4A, Table 4 (INT8 differences, interpreter configurations) | `audit_and_controls/results/runtime_diagnostics.csv`, `audit_and_controls/results/local_kernel_trace.json` |
| Fig 4B–C, Table 4 golden column, repeatability | `diagnostic_probes/probe_analysis.json`, `diagnostic_probes/raw/` |
| Table 5 (size, memory, timing) | `device_evaluation/device_metadata_evaluated_build.json`, `primary_run/test_latency.csv`, `primary_run/esp32_test_predictions.csv`, `diagnostic_probes/raw/*/metadata_*.json` |
| Table 6 (additional period) | `audit_and_controls/results/all_metrics.csv`, `audit_and_controls/results/extension_all_predictions.csv` |
| Table 7, Fig 5 (training seeds) | `multiseed/multiseed_summary.csv`, `multiseed/runs/` |
| Ridge control | `audit_and_controls/results/ridge_validation_selection.csv`, `audit_and_controls/results/ridge_model.json` |
| Artifact provenance | `audit_and_controls/results/audit_results.json`, `audit_and_controls/results/reconversion_results.json` |

## Environment

Python 3.11; exact versions in `environment/requirements-lock.txt` (TensorFlow-CPU 2.21.0, Keras 3.15.1, NumPy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1).

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-lock.txt matplotlib pillow
```

## Reproducing the results

All commands are run from the repository root unless stated otherwise.

**0. Training run** (optional, about 2 min on a CPU):

```sh
cd pipeline && PYTHONHASHSEED=20260916 python rerun.py --input <path>/household_power_consumption.txt --output-dir ../reproduced/results_primary
```

**1. Artifact audit, controls, kernel trace, bootstrap** (no retraining; needs the UCI archive, see `data/README.md`):

```sh
mkdir -p work/in && ln -s "$PWD/primary_run" work/in/results_primary
cp device_evaluation/device_metadata_evaluated_build.json work/in/device_metadata.json
cp device_evaluation/device_metadata_static_evaluated_build.json work/in/device_metadata_static.json
cp device_evaluation/dependencies.lock work/in/
python audit_and_controls/audit.py --input work/in --uci <path>/individual+household+electric+power+consumption.zip \
  --protocol pipeline/rerun.py --plan audit_and_controls/plan.json --output reproduced --skip-private-firmware
python audit_and_controls/kernel_trace.py --results primary_run --output reproduced/local_kernel_trace.json
python audit_and_controls/reconvert_check.py --results primary_run --protocol pipeline/rerun.py --output reproduced
python audit_and_controls/paired_bootstrap.py --predictions primary_run/esp32_test_predictions.csv --output reproduced/paired_bootstrap.json
```

The audit reproduces `all_metrics.csv`, `test_all_predictions.csv`, `extension_all_predictions.csv`, `runtime_diagnostics.csv` and the ridge model in `audit_and_controls/results/` byte for byte; `reference_pipeline_hash` becomes true when `--protocol pipeline/rerun.py` is the historical script. With `--skip-private-firmware`, binary checks of the firmware image are reported as SKIPPED.

**2. Diagnostic probes** (Fig 4B–C, repeatability):

```sh
python diagnostic_probes/analyze_probes.py --golden primary_run/golden_vectors.npz \
  --model primary_run/model_int8.tflite \
  --probes diagnostic_probes/raw/session1_probe_optimized diagnostic_probes/raw/session2_probe_extension \
  --output reproduced/probe_analysis.json
```

Both probe directories come from the same boot of the diagnostic build; `session2_probe_extension` is the second client session over the same 32 training-period golden vectors, not the additional July–September period.

**3. Training variability across seeds** (Table 7):

```sh
cd multiseed
PYTHONHASHSEED=20260916 INIT_BASE=0 python run_seed.py 20260916 out_20260916   # primary configuration
for k in 1 2 3 4 5 6 7 8 9 10; do PYTHONHASHSEED=$k INIT_BASE=$((1000*k)) python run_seed.py $k out_s$k; done
```

The primary configuration reproduces the archived FP32 and local INT8 test predictions of `primary_run/` bit for bit on Linux x86_64 with one thread and deterministic operations (about 1 min per run on a CPU).

**4. Figures:** `python figures/make_figures.py` (writes `figures/output/`).

## Provenance notes and known limitations

- `pipeline/rerun.py` is the historical training script: its SHA-256 (`31740c77…`) and that of `pipeline/requirements.txt` (`66e98a86…`) match `pipeline_code_sha256` and `requirements_sha256` in `primary_run/run_manifest.json`. Run on the public UCI text file with `PYTHONHASHSEED=20260916` (TensorFlow-CPU 2.21.0, Linux x86_64), it reproduces `model_int8.tflite` byte for byte, together with `model_data.cc/.h`, `preprocessing_constants.h`, `training_history.csv`, `golden_vectors.npz`, `test_windows.npz` and `test_predictions.csv`; `model_fp32.keras` differs only in container metadata (identical weights), and latency files differ as expected.
- `requirements.txt` pins `tensorflow==2.18.0`, but the environment recorded at run time used TensorFlow 2.21.0 (CUDA-enabled build); all results refer to the recorded environment. `rerun.py` reads package versions via `importlib.metadata.version("tensorflow")`; with the `tensorflow-cpu` wheel install `tensorflow==2.21.0` instead or alias the name.
- The local INT8 reference uses `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES` (explicit in `rerun.py`); XNNPACK and reference kernels give different outputs (Table 4).
- In `rerun.py`, the kernel-initializer seeds are fixed (101–103, 201–203) independently of `--seed`; the multi-seed study offsets them by `INIT_BASE` (`multiseed/rp_vs_rerun.diff`).
- `pipeline/evaluate_esp32.py` is the client as used for data collection: the INT8 golden-vector check prints a warning instead of raising, and the kWh-level golden check was run with a 2 kWh tolerance; the final exact-match check still raises after all 477 responses are stored. `pipeline/evaluate_esp32_prespecified.py` is the prespecified strict version. All metrics were recomputed from the stored paired outputs.
- Firmware sources of both builds are in `firmware/` (see `firmware/README.md`). Two lines in `main/wifi.cc` that contained a hard-coded Wi-Fi SSID and password were removed; the build-time `sdkconfig` is provided as `firmware/sdkconfig.redacted` (credentials blanked), so its hash differs from the device-reported `81908759…`, which matches the unredacted file. Firmware binaries are not distributed because they embed the credentials.
- Resource values (minimum free RAM, PSRAM, HTTP-task stack) come from the diagnostic build, whose image differs from the evaluated one. `min_internal_free_heap_bytes = 16384` in the metadata is a configured guard threshold, not a measurement.

## Data source and licences

Source data: G. Hébrail, A. Bérard, *Individual Household Electric Power Consumption*, UCI Machine Learning Repository, https://doi.org/10.24432/C58K54, licensed CC BY 4.0. Derived data in `data/`, `primary_run/` and result tables are distributed under CC BY 4.0 (see `DATA_LICENSE.md`). Code is distributed under the licence in `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the article and the Zenodo archive of this repository.
