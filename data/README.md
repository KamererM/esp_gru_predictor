# Data

Derived from the UCI *Individual Household Electric Power Consumption* dataset (Hébrail & Bérard, https://doi.org/10.24432/C58K54, CC BY 4.0).

- Source archive: https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
- SHA-256 of the archive used: `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`
- SHA-256 of the extracted text file recorded in the run manifest: `c79e3e19348bf518748d63455f98b2f09daf9b1a72fa3f42048fadd9a588225e`

Target: hourly energy in kWh, the sum of the 60 minute-level `Global_active_power` values (kW) divided by 60. Hours with any missing minute are removed (4,269 of 4,344 hours retained for 1 Jan – 30 Jun 2007).

| File | Content |
|---|---|
| `hourly_splits.csv` | `timestamp`, `energy_kwh`, `split` (`train` 2,988 h, `validation` 640 h, `test` 641 h) |
| `extension_hourly.csv` | Complete hours of 1 Jul – 30 Sep 2007 (2,200 h), used only for the additional temporal check |
| `extension_windows.npz` | 2,008 valid 32-h windows of the additional period: `X_raw_kwh`, `X_int8`, `output_int8` (local INT8), `target_kwh`, `timestamp_ns` |

The scaler (training period only) has mean 1.2785696117804553 kWh and population SD 1.074473061836858 kWh. Windows never cross split boundaries or gaps.
