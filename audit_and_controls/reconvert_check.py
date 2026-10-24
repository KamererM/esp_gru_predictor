"""Reconstruct train-only calibration; compare quantized constants/outputs.
Supports the historical calibration hypothesis but does not recover a missing
historical calibration log or prove that test-driven decisions never occurred.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_NUM_INTRAOP_THREADS"]="1"
os.environ["TF_NUM_INTEROP_THREADS"]="1"
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
p=argparse.ArgumentParser()
p.add_argument("--results",type=Path,required=True)
p.add_argument("--protocol",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
s=importlib.util.spec_from_file_location("protocol_reference",a.protocol)
mod=importlib.util.module_from_spec(s);sys.modules[s.name]=mod;s.loader.exec_module(mod)
m=json.loads((a.results/"run_manifest.json").read_text());cfg=mod.Config(**m["config"])
d=pd.read_csv(a.results/"hourly_splits.csv",parse_dates=["timestamp"],float_precision="round_trip")
train=d[d.split=="train"].set_index("timestamp")
scaler=StandardScaler().fit(train[["energy_kwh"]])
X,_,_,_,raw=mod.make_windows(train,scaler,cfg.steps)
loaded=tf.keras.models.load_model(a.results/"model_fp32.keras",compile=False)
deploy=mod.build_model(cfg,batch_size=1);deploy.set_weights(loaded.get_weights())
a.output.mkdir(exist_ok=True,parents=True)
blob,idx=mod.convert_int8(deploy,X,cfg,a.output/"reconverted_int8.tflite")
def interpreter(f):
 x=tf.lite.Interpreter(model_path=str(f),experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
  experimental_preserve_all_tensors=True,num_threads=1)
 x.allocate_tensors();return x
old=interpreter(a.results/"model_int8.tflite")
new=interpreter(a.output/"reconverted_int8.tflite")
old_details=old.get_tensor_details();new_details=new.get_tensor_details()
z=np.load(a.results/"test_windows.npz")
preds=[]
for it in [old,new]:
 inp=it.get_input_details()[0];o=it.get_output_details()[0];q=[]
 for x in z["X_int8"]:
  it.set_tensor(inp["index"],x.reshape(inp["shape"]));it.invoke()
  q.append(int(it.get_tensor(o["index"]).ravel()[0]))
 preds.append(q)
def flat(f):
 from tensorflow.lite.python import schema_py_generated as schema
 mm=schema.Model.GetRootAsModel(Path(f).read_bytes(),0)
 constants=[]
 for k in range(mm.BuffersLength()):
  b=mm.Buffers(k).DataAsNumpy()
  if isinstance(b,np.ndarray) and len(b):
   constants.append(hashlib.sha256(b.tobytes()).hexdigest())
 return constants
result={"calibration_reconstructed_train_only":True,"calibration_count":len(idx),
 "flatbuffer_byte_identical":blob==(a.results/"model_int8.tflite").read_bytes(),
 "all_nonempty_buffers_multiset_equal":sorted(flat(a.results/"model_int8.tflite"))==sorted(flat(a.output/"reconverted_int8.tflite")),
 "test_output_q_mismatch":int(np.sum(np.array(preds[0])!=np.array(preds[1]))),
 "old_input_quantization":old.get_input_details()[0]["quantization"],
 "new_input_quantization":new.get_input_details()[0]["quantization"],
 "old_output_quantization":old.get_output_details()[0]["quantization"],
 "new_output_quantization":new.get_output_details()[0]["quantization"],
 "caveat":"Reconstruction of sufficient train-only calibration, not authentication of historical code or index log."}
(a.output/"reconversion_results.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
