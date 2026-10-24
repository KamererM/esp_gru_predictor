"""Offline audit + frozen-model temporal extension, never uploads firmware.
Run with TF 2.21.0 / Keras 3.15.1 / numpy 2.0.2.
"""
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import hashlib
import importlib.util
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p=argparse.ArgumentParser()
p.add_argument("--input",type=Path,required=True)
p.add_argument("--uci",type=Path,required=True)
p.add_argument("--protocol",type=Path,required=True)
p.add_argument("--plan",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--skip-private-firmware",action="store_true",
               help="Skip binary/config checks when those private inputs are withheld.")
a=p.parse_args(); out=a.output;out.mkdir(parents=True,exist_ok=True)
src=a.input/"results_primary"
manifest=json.loads((src/"run_manifest.json").read_text())
meta=json.loads((a.input/"device_metadata.json").read_text())
plan=json.loads(a.plan.read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(name,obj):
 def encode(v):
  if isinstance(v,np.ndarray): return v.tolist()
  if isinstance(v,np.generic): return v.item()
  return str(v)
 (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=encode))
spec=importlib.util.spec_from_file_location("protocol_reference",a.protocol)
proto=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=proto;spec.loader.exec_module(proto)
report={"run_id":manifest["run_id"],"checks":{},"scope":
 "Evidence audit, not authentication of the whole historical training process."}
checks=report["checks"]
mapping={"fp32_model":"model_fp32.keras","int8_model":"model_int8.tflite",
         "test_windows":"test_windows.npz","test_predictions":"test_predictions.csv",
         "golden_vectors":"golden_vectors.npz","model_data_cc":"model_data.cc",
         "model_data_h":"model_data.h","preprocessing_constants_h":"preprocessing_constants.h"}
checks["artifact_hashes"]={k:sha(src/f)==manifest["artifacts"][k+"_sha256"]
                          for k,f in mapping.items()}
checks["reference_pipeline_hash"]=sha(a.protocol)==manifest["artifacts"]["pipeline_code_sha256"]
if a.skip_private_firmware:
 checks["firmware_hash"]=checks["sdkconfig_hash"]=checks["elf_hash"]="SKIPPED_PRIVATE_INPUTS"
 report["firmware"]={"status":"not rechecked in this invocation"}
else:
 checks["firmware_hash"]=sha(a.input/"gru_predictor.bin")==meta["firmware_image_sha256"]
 checks["sdkconfig_hash"]=sha(a.input/"sdkconfig")==meta["sdkconfig_sha256"]
 bin_data=(a.input/"gru_predictor.bin").read_bytes()
 report["firmware"]={"bytes":len(bin_data),"sha256":sha(a.input/"gru_predictor.bin"),
  "descriptor_magic":bin_data[32:36].hex(),
  "idf_version":bin_data[144:176].split(b"\0")[0].decode(),
  "elf_sha256":bin_data[176:208].hex(),
  "int8_model_occurrences":bin_data.count((src/"model_int8.tflite").read_bytes())}
 checks["elf_hash"]=report["firmware"]["elf_sha256"]==meta["app_elf_sha256"]
# The C array and exact embedded FlatBuffer independently bind model to binary.
cc=(src/"model_data.cc").read_text()
cbytes=bytes(int(x,16) for x in re.findall(r"0x([0-9a-fA-F]{2})\b",cc))
checks["c_array_matches_model"]=cbytes==(src/"model_int8.tflite").read_bytes()
report["hardware_metadata"]=meta
report["hashes"]={f.relative_to(a.input).as_posix():{"sha256":sha(f),"bytes":f.stat().st_size}
                  for f in a.input.rglob("*") if f.is_file()}
hourly=pd.read_csv(src/"hourly_splits.csv",parse_dates=["timestamp"],float_precision="round_trip")
assert hourly.timestamp.is_unique and hourly.timestamp.is_monotonic_increasing
assert (hourly["count"]==60).all()
groups={n:df.set_index("timestamp") for n,df in hourly.groupby("split")}
train=groups["train"]
mean=float(train.energy_kwh.mean());std=float(train.energy_kwh.std(ddof=0))
checks["train_scaler_mean"]=bool(np.isclose(mean,manifest["scaler"]["mean_kwh"],atol=1e-12,rtol=0))
checks["train_scaler_std"]=bool(np.isclose(std,manifest["scaler"]["std_kwh"],atol=1e-12,rtol=0))
report["scaler_recomputed"]={"mean":mean,"std":std,
 "all_data_mean":float(hourly.energy_kwh.mean()),"all_data_std":float(hourly.energy_kwh.std(ddof=0))}
scaler=StandardScaler().fit(train[["energy_kwh"]])
cfg=proto.Config(**manifest["config"])
rebuilt={n:proto.make_windows(groups[n],scaler,cfg.steps) for n in ("train","validation","test")}
report["splits"]={}
for n,g in groups.items():
 report["splits"][n]={"hours":len(g),"windows":len(rebuilt[n][0]),
     "start":str(g.index.min()),"end":str(g.index.max()),
     "first_target":str(rebuilt[n][3][0]),"last_target":str(rebuilt[n][3][-1])}
checks["chronological_disjoint_splits"]=bool(train.index.max()<groups["validation"].index.min()
 and groups["validation"].index.max()<groups["test"].index.min())
checks["split_counts"]=all(len(groups[n])==manifest["splits"][n]["hourly_rows"]
 and len(rebuilt[n][0])==manifest["splits"][n]["valid_windows"] for n in rebuilt)
z=np.load(src/"test_windows.npz");gold=np.load(src/"golden_vectors.npz")
checks["all_test_raw_windows"]=np.array_equal(rebuilt["test"][4],z["X_raw_kwh"])
checks["all_test_targets"]=np.array_equal(rebuilt["test"][2],z["target_kwh"])
checks["all_test_timestamps"]=np.array_equal(rebuilt["test"][3].astype("datetime64[ns]").astype("int64"),z["timestamp_ns"])
cal=proto.calibration_indices(rebuilt["train"][0],cfg)
report["reconstructed_calibration"]={"count":len(cal),"source":"train",
 "min_target":str(rebuilt["train"][3][cal].min()),"max_target":str(rebuilt["train"][3][cal].max()),
 "status":"Reconstructed from reference algorithm; historical index list not retained."}
pd.DataFrame({"train_window_index":cal,"timestamp":rebuilt["train"][3][cal]}).to_csv(out/"reconstructed_calibration.csv",index=False)
gold_indices=np.unique(np.concatenate([cal[np.linspace(0,len(cal)-1,min(16,len(cal)),dtype=int)],
 np.linspace(0,len(rebuilt["train"][0])-1,min(16,len(rebuilt["train"][0])),dtype=int)]))
checks["golden_reconstruction"]=np.array_equal(rebuilt["train"][4][gold_indices],gold["X_raw_kwh"])
checks["golden_timestamps"]=np.array_equal(rebuilt["train"][3][gold_indices].astype("datetime64[ns]").astype("int64"),gold["timestamp_ns"])
hist=pd.read_csv(src/"training_history.csv")
best=int(hist.val_loss.idxmin())
report["early_stopping"]={"epochs":len(hist),"best_epoch_zero_based":best,
 "best_epoch_one_based":best+1,"min_validation_loss":float(hist.val_loss.min()),
 "epochs_after_best":len(hist)-best-1,"configured_patience":cfg.patience}
model=tf.keras.models.load_model(src/"model_fp32.keras",compile=False)
report["model"]={"parameters":model.count_params(),"fp32_file_bytes":(src/"model_fp32.keras").stat().st_size}
def fp_predict(X):
 # Match original batch-1 model-call boundary.
 y=np.array([model(v[None],training=False).numpy().reshape(-1)[0] for v in X],dtype=np.float32)
 return proto.denormalize_float32(y,scaler).reshape(-1).astype(float)
val_norm=np.array([model(v[None],training=False).numpy().reshape(-1)[0] for v in rebuilt["validation"][0]])
report["early_stopping"]["reloaded_validation_mse"]=float(np.mean((val_norm-rebuilt["validation"][1].ravel())**2))
report["early_stopping"]["best_loss_abs_difference"]=abs(report["early_stopping"]["reloaded_validation_mse"]-float(hist.val_loss.min()))

# Downloaded official dataset is independently aggregated, not inferred from predictions.
with zipfile.ZipFile(a.uci) as zz:
 with zz.open("household_power_consumption.txt") as stream:
  raw=pd.read_csv(stream,sep=";",usecols=["Date","Time","Global_active_power"],dtype=str)
timestamps=pd.to_datetime(raw.Date+" "+raw.Time,format="%d/%m/%Y %H:%M:%S")
series=pd.Series(pd.to_numeric(raw.Global_active_power,errors="coerce").to_numpy(),index=timestamps)
assert series.index.is_unique and series.index.is_monotonic_increasing
period=series.loc["2007-01-01":plan["extension_end"]]
minute_grid=pd.date_range("2007-01-01",plan["extension_end"],freq="min")
period=period.reindex(minute_grid)
agg=period.resample("h").agg(["sum","count"])
full=agg.loc[agg["count"]==60].copy()
full["energy_kwh"]=full["sum"]/60
prior=full.loc[:manifest["config"]["subset_end"]]
joined=prior.reindex(hourly.timestamp)
checks["uci_hourly_timestamps"]=np.array_equal(prior.index.to_numpy(),hourly.timestamp.to_numpy())
report["uci_hourly_max_abs_difference"]=float(np.max(np.abs(joined.energy_kwh.to_numpy()-hourly.energy_kwh.to_numpy())))
checks["uci_hourly_values"]=report["uci_hourly_max_abs_difference"]<1e-12
report["uci"]={"archive_sha256":sha(a.uci),
 "source_url":"https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
 "minute_nan_in_original_period":int(series.loc["2007-01-01":"2007-06-30 23:59"].isna().sum()),
 "full_hour_count_original":len(prior)}
ext=full.loc[plan["extension_start"]:plan["extension_end"]]
ext.index.name="timestamp"
ext[["energy_kwh","count"]].to_csv(out/"extension_hourly.csv")
rebuilt["extension"]=proto.make_windows(ext,scaler,cfg.steps)
report["extension"]={"hourly_rows":len(ext),"valid_windows":len(rebuilt["extension"][0]),
 "first_target":str(rebuilt["extension"][3][0]),"last_target":str(rebuilt["extension"][3][-1])}
save("audit_in_progress.json",report)

def tflite_predict(Xraw,kind):
 inter=tf.lite.Interpreter(model_path=str(src/"model_int8.tflite"),
   experimental_op_resolver_type=kind,num_threads=1)
 inter.allocate_tensors()
 inp=inter.get_input_details()[0];op=inter.get_output_details()[0]
 xnorm=proto.normalize_float32(Xraw,scaler)
 qi=proto.quantize(xnorm,*inp["quantization"])
 qo=[]
 for x in qi:
  inter.set_tensor(inp["index"],x.reshape(inp["shape"]))
  inter.invoke();qo.append(int(inter.get_tensor(op["index"]).ravel()[0]))
 qo=np.asarray(qo,dtype=np.int8)
 scale,zero=op["quantization"]
 yn=(qo.astype(np.float32)-np.float32(zero))*np.float32(scale)
 pred=proto.denormalize_float32(yn,scaler).reshape(-1).astype(float)
 details={"input":{k:inp[k] for k in ["shape","shape_signature","quantization"]},
 "output_quantization":op["quantization"],
 "float_tensors":sum(d["dtype"]==np.float32 for d in inter.get_tensor_details()),
 "operators":sorted(set(d["op_name"] for d in inter._get_ops_details()))}
 return pred,qi,qo,details

esp=pd.read_csv(src/"esp32_test_predictions.csv",parse_dates=["timestamp"],float_precision="round_trip")
stored=pd.read_csv(src/"test_predictions.csv",parse_dates=["timestamp"],float_precision="round_trip")
checks["csv_pairing"]=np.array_equal(esp.timestamp.to_numpy(),z["timestamp_ns"].astype("datetime64[ns]"))
checks["csv_targets"]=np.array_equal(esp.observed_kwh.to_numpy(),z["target_kwh"])
# Recover device q from saved dequantized outputs; verify <= float32 roundoff.
q=manifest["quantization"]
step=np.float32(q["output_scale"])*np.float32(std)
device_q=np.rint((esp.esp32_kwh.to_numpy()-np.float32(mean))/step+q["output_zero_point"]).astype(int)
back=proto.denormalize_float32((device_q.astype(np.float32)-q["output_zero_point"])*np.float32(q["output_scale"]),scaler)
report["device_q_reconstruction_max_abs_kwh"]=float(np.max(np.abs(back-esp.esp32_kwh)))
report["runtime_comparison"]={}
R=tf.lite.experimental.OpResolverType
outputs={}
for name,kind in [("AUTO",R.AUTO),("BUILTIN_WITHOUT_DEFAULT_DELEGATES",R.BUILTIN_WITHOUT_DEFAULT_DELEGATES),("BUILTIN_REF",R.BUILTIN_REF)]:
 pred,qi,qo,details=tflite_predict(z["X_raw_kwh"],kind)
 outputs[name]=qo
 report["runtime_comparison"][name]={"n":len(qo),"local_saved_q_mismatch":int(np.sum(qo!=z["output_int8"])),
 "device_reconstructed_q_mismatch":int(np.sum(qo!=device_q)),
 "max_abs_q_difference_device":int(np.max(np.abs(qo.astype(int)-device_q))),
 "input_matches":bool(np.array_equal(qi,z["X_int8"])),
 "max_abs_saved_local_kwh_difference":float(np.max(np.abs(pred-stored.int8_local_kwh.to_numpy()))),
 "details":details}
pd.DataFrame({"timestamp":esp.timestamp,"saved_local_q":z["output_int8"],
 "device_q_reconstructed":device_q,**outputs}).to_csv(out/"runtime_diagnostics.csv",index=False)
fp=fp_predict(rebuilt["test"][0])
report["fp32_reproduction_max_abs_kwh"]=float(np.max(np.abs(fp-stored.fp32_kwh.to_numpy())))

def metrics(y,pred):
 e=pred-y
 return {"N":len(y),"MAE_kwh":float(np.abs(e).mean()),"RMSE_kwh":float(np.sqrt(np.mean(e*e))),
 "MAPE_pct":float(100*np.mean(np.abs(e/y))),"WAPE_pct":float(100*np.abs(e).sum()/np.abs(y).sum()),
 "R2":float(1-np.sum(e*e)/np.sum((y-y.mean())**2))}
ridge_results=[];candidates=[]
for alpha in plan["ridge_alpha_grid"]:
 reg=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
 reg.fit(rebuilt["train"][4].astype(float),rebuilt["train"][2])
 mp=metrics(rebuilt["validation"][2],reg.predict(rebuilt["validation"][4].astype(float)))
 ridge_results.append({"alpha":alpha,**mp});candidates.append(reg)
best_ridge=int(np.argmin([r["RMSE_kwh"] for r in ridge_results]))
reg=candidates[best_ridge]
report["ridge"]={"selection":"validation RMSE; train-only fit","alpha":plan["ridge_alpha_grid"][best_ridge],
 "grid_results":ridge_results}
pd.DataFrame(ridge_results).to_csv(out/"ridge_validation_selection.csv",index=False)
save("ridge_model.json",{"alpha":report["ridge"]["alpha"],"mean":reg[0].mean_.tolist(),
 "scale":reg[0].scale_.tolist(),"coef":reg[1].coef_.tolist(),"intercept":float(reg[1].intercept_)})
all_metrics=[]
for n in ("test","extension"):
 X,_,y,ts,Xraw=rebuilt[n]
 fp_now=fp if n=="test" else fp_predict(X)
 # This explicit resolver reproduces every recorded local INT8 output.
 # AUTO is environment-dependent and remains a separate diagnostic above.
 ip,qi,qo,_=tflite_predict(Xraw,R.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
 frame=pd.DataFrame({"timestamp":ts,"observed_kwh":y,"persistence_kwh":Xraw[:,-1],
   "seasonal24_kwh":Xraw[:,-24],"ridge_kwh":reg.predict(Xraw.astype(float)),
   "fp32_recomputed_kwh":fp_now,"int8_recomputed_kwh":ip})
 if n=="test":
  frame["fp32_recorded_kwh"]=stored.fp32_kwh
  frame["esp32_recorded_kwh"]=esp.esp32_kwh
 for name in frame.columns[2:]:
  all_metrics.append({"period":n,"model":name,**metrics(y,frame[name].to_numpy())})
 frame.to_csv(out/f"{n}_all_predictions.csv",index=False)
 if n=="extension":
  np.savez_compressed(out/"extension_windows.npz",X_raw_kwh=Xraw,X_int8=qi,output_int8=qo,
    target_kwh=y,timestamp_ns=ts.astype("datetime64[ns]").astype("int64"))
pd.DataFrame(all_metrics).to_csv(out/"all_metrics.csv",index=False)
report["metrics"]=all_metrics
report["analysis_environment"]={"python":sys.version,"tensorflow":tf.__version__,
 "keras":tf.keras.__version__,"numpy":np.__version__,"pandas":pd.__version__}
save("audit_results.json",report)
print(json.dumps({k:v for k,v in report.items() if k not in ["hashes","hardware_metadata"]},indent=2,default=str))
