#!/usr/bin/env python3
"""Recreate the data figures of the manuscript (Fig 1, Fig 4, Fig 5) from repository files.
Run from the repository root: python figures/make_figures.py
Fig 2 (LSTM/GRU schematic) and Fig 3 (code listing) are illustrations and are not data-derived."""
import json, collections
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from PIL import Image
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OUT = Path("figures/output"); OUT.mkdir(parents=True, exist_ok=True)
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight")
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(400, 400)); plt.close(fig)

# Fig 1: data partition
h = pd.read_csv("data/hourly_splits.csv", parse_dates=["timestamp"])
e = pd.read_csv("data/extension_hourly.csv", parse_dates=["timestamp"]); e["split"] = "additional"
a = pd.concat([h, e]); full = pd.date_range("2007-01-01", "2007-09-30 23:00", freq="h")
s = a.set_index("timestamp")["energy_kwh"].reindex(full); sp = a.set_index("timestamp")["split"].reindex(full)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
cols = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868", "additional": "#8172B3"}
lab = {"train": "Training (2,988 h; 2,796 windows)", "validation": "Validation (640 h; 576 windows)",
       "test": "Test (641 h; 477 windows)", "additional": "Additional period (2,200 h; 2,008 windows, local only)"}
for k, c in cols.items(): ax.plot(full, s.where(sp == k), color=c, lw=0.45, label=lab[k])
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Hourly energy (kWh)"); ax.set_xlim(full[0], full[-1]); ax.set_ylim(0, None)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8, handlelength=1.5)
for l in ax.get_legend().get_lines(): l.set_linewidth(2.5)
save(fig, "Fig1")

# Fig 4: INT8 differences
t = pd.read_csv("audit_and_controls/results/runtime_diagnostics.csv")
dt = (t.device_q_reconstructed - t.saved_local_q).abs().value_counts().sort_index()
gm = json.load(open("diagnostic_probes/probe_analysis.json"))["outputs_int8"]; dev = np.array(gm["device"])
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1, 1.15, 1]})
ax = axs[0]; ax.bar(dt.index, dt.values, color="#4C72B0", width=0.6)
for x, v in zip(dt.index, dt.values): ax.text(x, v + 4, str(v), ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_ylim(0, 280); ax.set_xlabel("|ESP32-S3 − local INT8| (steps)"); ax.set_ylabel("Test windows")
ax.set_title("A  Test period, N = 477", loc="left", fontsize=9)
ax = axs[1]; d = collections.Counter((dev - np.array(gm["saved"])).tolist()); xs = range(-2, 3)
ax.bar(xs, [d.get(x, 0) for x in xs], color="#55A868", width=0.6)
for x in xs: ax.text(x, d.get(x, 0) + 0.3, str(d.get(x, 0)), ha="center", fontsize=8)
ax.set_xticks(list(xs)); ax.set_xlabel("ESP32-S3 − local INT8 (steps)"); ax.set_ylabel("Golden vectors")
ax.set_title("B  Diagnostic build, 32 vectors", loc="left", fontsize=9); ax.set_ylim(0, 18)
ax = axs[2]; keys = ["NODELEG", "REF", "AUTO"]; v = [int((np.array(gm[k]) != dev).sum()) for k in keys]
ax.bar(range(3), v, color="#8172B3", width=0.6)
for i, x in enumerate(v): ax.text(i, x + 0.5, f"{x}/32", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["No\ndelegates", "Reference\nkernels", "XNNPACK\n(AUTO)"], fontsize=8)
ax.set_ylabel("Differing outputs"); ax.set_ylim(0, 32); ax.set_title("C  Device vs local modes", loc="left", fontsize=9)
fig.tight_layout(w_pad=1.2); save(fig, "Fig4")

# Fig 5: training seeds
d = pd.read_csv("multiseed/multiseed_summary.csv"); x = np.arange(len(d)); lab = ["P"] + [str(k) for k in range(1, len(d))]
panels = [("A  Test MAE (kWh)", "fp_mae", "i8_mae", 0.414514, 0.406437), ("B  Test RMSE (kWh)", "fp_rmse", "i8_rmse", 0.587012, None),
          ("C  Additional period RMSE (kWh)", "ext_fp_rmse", "ext_i8_rmse", 0.578728, None)]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
for ax, (title, a_, b_, ridge, pers) in zip(axs, panels):
    ax.plot(x, d[a_], "o", color="#4C72B0", ms=4, label="GRU FP32"); ax.plot(x, d[b_], "x", color="#DD8452", ms=5, label="GRU INT8 (local)")
    ax.axhline(ridge, color="#55A868", ls="--", lw=1, label="Ridge")
    if pers: ax.axhline(pers, color="0.4", ls=":", lw=1, label="Persistence")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.set_xlabel("Training seed (P = primary)"); ax.set_title(title, loc="left", fontsize=9)
axs[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(w_pad=1.0); save(fig, "Fig5")
print("Figures written to", OUT)
