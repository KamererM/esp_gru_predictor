#!/usr/bin/env python3
"""Analyse diagnostic golden-vector probes of the ESP32-S3 build.

Compares on-device INT8 outputs (probe responses) with three local TFLite
interpreter configurations and summarises repeatability, input identity,
dequantisation consistency and timing.  Reproduces the numbers reported in
the Results subsection "Localization and repeatability of INT8 differences",
Table 4 (golden column), Table 5 (diagnostic build) and Fig 4B/4C.
"""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf

MODES = {
    "NODELEG": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    "REF": tf.lite.experimental.OpResolverType.BUILTIN_REF,
    "AUTO": tf.lite.experimental.OpResolverType.AUTO,
}

def local_outputs(model, X):
    out = {}
    for name, rt in MODES.items():
        it = tf.lite.Interpreter(model_path=str(model), num_threads=1, experimental_op_resolver_type=rt)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        ys = []
        for x in X:
            it.set_tensor(i["index"], x.reshape(1, -1, 1).astype(np.int8)); it.invoke()
            ys.append(int(it.get_tensor(o["index"]).ravel()[0]))
        out[name] = np.array(ys)
    return out

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--probes", nargs="+", required=True, type=Path, help="probe directories with responses.jsonl")
    ap.add_argument("--mean-kwh", type=float, default=1.2785696117804553)
    ap.add_argument("--std-kwh", type=float, default=1.074473061836858)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    g = np.load(a.golden); X = g["X_int8"]; saved = g["output_int8"].astype(int)
    loc = local_outputs(a.model, X)
    it = tf.lite.Interpreter(model_path=str(a.model)); it.allocate_tensors()
    s_out, z_out = it.get_output_details()[0]["quantization"]
    rows = []
    for p in a.probes:
        for r in load(p / "responses.jsonl"):
            r["session"] = p.name; rows.append(r)
    ids = [r["response"]["request_id"] for r in rows]
    dev = {}
    inputs_equal = True; deq_err = []; t_ms = {}; rt_ms = {}
    for r in rows:
        k = r["window_index"]; resp = r["response"]
        dev.setdefault(k, []).append(int(resp["output_int8"]))
        inputs_equal &= list(resp["input_int8"]) == X[k].astype(int).tolist()
        q = resp["output_int8"]
        deq = ((np.float32(q) - np.float32(z_out)) * np.float32(s_out)) * np.float32(a.std_kwh) + np.float32(a.mean_kwh)
        deq_err.append(abs(float(deq) - resp["prediction"]))
        t_ms.setdefault(r["session"], []).append(resp["execution_time_us"] / 1000)
        rt_ms.setdefault(r["session"], []).append(r["round_trip_us"] / 1000)
    deterministic = all(len(set(v)) == 1 for v in dev.values())
    device = np.array([dev[k][0] for k in range(len(X))])
    lsb = s_out * a.std_kwh
    def cmp(ref):
        d = device - ref
        vals, cnt = np.unique(d[d != 0], return_counts=True)
        return {"n_differ": int((d != 0).sum()), "n": len(d), "lsb_histogram": {int(v): int(c) for v, c in zip(vals, cnt)},
                "mean_abs_kwh": float(np.abs(d).mean() * lsb), "max_abs_kwh": float(np.abs(d).max() * lsb)}
    stat = lambda v: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                      "p95": float(np.percentile(v, 95)), "min": float(np.min(v)), "max": float(np.max(v))}
    res = {
        "requests": len(rows), "unique_request_ids": len(set(ids)),
        "inputs_identical_to_golden": bool(inputs_equal), "outputs_deterministic_across_repeats": bool(deterministic),
        "local_nodelegate_equals_saved_reference": bool((loc["NODELEG"] == saved).all()),
        "device_vs": {m: cmp(v) for m, v in loc.items()},
        "local_modes_pairwise_differ": {f"{x}_vs_{y}": int((loc[x] != loc[y]).sum()) for x in MODES for y in MODES if x < y},
        "max_dequantisation_error_kwh": float(max(deq_err)), "output_lsb_kwh": float(lsb),
        "execution_time_ms": {k: stat(v) for k, v in t_ms.items()}, "round_trip_ms": {k: stat(v) for k, v in rt_ms.items()},
        "outputs_int8": {**{m: v.tolist() for m, v in loc.items()}, "device": device.tolist(), "saved": saved.tolist()},
    }
    a.output.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "outputs_int8"}, indent=1))

if __name__ == "__main__":
    main()
