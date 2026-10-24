"""Explicit diagnostic recorder, NOT a passing publication evaluator.
Never relaxes the original evaluator. Run on a host on the ESP32's network.
No retries: a network failure preserves partial JSONL and stops the run.
"""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
import requests

def int8(value):
    if isinstance(value,bool) or not isinstance(value,int) or not -128<=value<=127:
        raise ValueError("INT8 diagnostic must be a JSON integer in [-128,127]")
    return value

def metadata_check(body,expected,resources=False):
    for key in ("run_id","model_sha256","app_elf_sha256"):
        if body.get(key)!=expected[key]:
            raise ValueError("Device identity mismatch: "+key)
    for key in ("input_scale_float32_bits","output_scale_float32_bits",
                "input_zero_point","output_zero_point"):
        if body.get(key)!=expected[key]:
            raise ValueError("Quantization mismatch: "+key)
    used,capacity=body["arena_used_bytes"],body["arena_capacity_bytes"]
    if not 0<used<=capacity:
        raise ValueError("Invalid arena measurements")
    if resources and body.get("resource_probe_version")!="gru-resource-v1":
        raise ValueError("Resource instrumentation missing")
    if resources:
        free=body["internal_free_now_bytes"];minimum=body["internal_min_free_since_boot_bytes"]
        largest=body["internal_largest_free_block_bytes"];total=body["internal_total_bytes"]
        if not 0<=minimum<=free<=total or not 0<=largest<=free:
            raise ValueError("Invalid internal RAM measurements")
    return body

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",required=True)
    p.add_argument("--metadata-url",required=True)
    p.add_argument("--expected-metadata",type=Path,required=True)
    p.add_argument("--windows",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeat",type=int,default=3)
    p.add_argument("--require-resources",action="store_true")
    a=p.parse_args()
    if a.repeat<1:raise ValueError("repeat must be positive")
    a.output.mkdir(parents=True,exist_ok=False)
    expected=json.loads(a.expected_metadata.read_text())
    if hashlib.sha256(a.model.read_bytes()).hexdigest()!=expected["model_sha256"]:
        raise ValueError("Local model hash mismatch")
    z=np.load(a.windows,allow_pickle=False)
    session=requests.Session()
    def snapshot():
        r=session.get(a.metadata_url,timeout=10);r.raise_for_status()
        return metadata_check(r.json(),expected,a.require_resources)
    before=snapshot()
    (a.output/"metadata_before.json").write_text(json.dumps(before,indent=2))
    batch=str(uuid.uuid4());count=0;miss=0
    try:
        with (a.output/"responses.jsonl").open("w") as stream:
            for repeat in range(a.repeat):
                for i,values in enumerate(z["X_raw_kwh"]):
                    stamp=str(int(z["timestamp_ns"][i]))
                    rid=f"{expected['run_id']}:probe:{batch}:{repeat}:{i}"
                    payload={"run_id":expected["run_id"],"timestamp_ns":stamp,
                             "request_id":rid,"size":len(values),
                             "data":values.tolist(),"diagnostic":True}
                    start=time.perf_counter_ns()
                    r=session.post(a.url,json=payload,timeout=10);r.raise_for_status()
                    roundtrip=(time.perf_counter_ns()-start)/1000
                    body=r.json()
                    for k in ("run_id","timestamp_ns","request_id"):
                        if body.get(k)!=payload[k]:raise ValueError("Pairing mismatch "+k)
                    qi=body["input_int8"]
                    if not isinstance(qi,list) or len(qi)!=len(values):raise ValueError("Bad input shape")
                    qi=np.array([int8(x) for x in qi],dtype=np.int8)
                    qo=int8(body["output_int8"])
                    if not np.array_equal(qi,z["X_int8"][i]):raise ValueError("Input byte mismatch")
                    if not np.isfinite(body["prediction"]):raise ValueError("Nonfinite prediction")
                    if not np.isfinite(body["execution_time_us"]) or body["execution_time_us"]<0:
                        raise ValueError("Invalid timing")
                    exact=qo==int(z["output_int8"][i]);miss+=not exact;count+=1
                    stream.write(json.dumps({"repeat":repeat,"window_index":i,
                        "reference_output_int8":int(z["output_int8"][i]),
                        "output_exact":exact,"round_trip_us":roundtrip,"response":body})+"\n")
                    stream.flush()
    finally:
        after=snapshot()
        (a.output/"metadata_after.json").write_text(json.dumps(after,indent=2))
        if "sample_uptime_us" in before and after["sample_uptime_us"]<before["sample_uptime_us"]:
            raise ValueError("Device rebooted during probe")
        (a.output/"summary.json").write_text(json.dumps({
            "status":"diagnostic_only_no_acceptance_claim",
            "completed_requests":count,"output_mismatches":miss,
            "window_sha256":hashlib.sha256(a.windows.read_bytes()).hexdigest(),
            "model_sha256":expected["model_sha256"]},indent=2))
if __name__=="__main__":main()
