"""First divergent local operator on train golden vectors only.
This is NOT a trace from the ESP32. Do not attribute its first divergence to
the device without device-side intermediate tensors.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import argparse,json
from pathlib import Path
import numpy as np
import tensorflow as tf

p=argparse.ArgumentParser()
p.add_argument("--results",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
R=tf.lite.experimental.OpResolverType
def create(kind):
 i=tf.lite.Interpreter(model_path=str(a.results/"model_int8.tflite"),
   experimental_preserve_all_tensors=True,experimental_op_resolver_type=kind,num_threads=1)
 i.allocate_tensors();return i
opt=create(R.BUILTIN_WITHOUT_DEFAULT_DELEGATES);ref=create(R.BUILTIN_REF)
z=np.load(a.results/"golden_vectors.npz")
ops=opt._get_ops_details();rows=[]
for n,x in enumerate(z["X_int8"]):
 for inter in (opt,ref):
  d=inter.get_input_details()[0]
  inter.set_tensor(d["index"],x.reshape(d["shape"]));inter.invoke()
 first=None
 for k,op in enumerate(ops):
  for tensor in op["outputs"]:
   aa,bb=opt.get_tensor(int(tensor)),ref.get_tensor(int(tensor))
   if not np.array_equal(aa,bb):
    first={"golden_index":n,"operator_index":k,"operator_name":op["op_name"],
       "tensor_index":int(tensor),"mismatched_elements":int(np.sum(aa!=bb)),
       "max_abs_integer_difference":int(np.max(np.abs(aa.astype(np.int64)-bb.astype(np.int64))))}
    break
  if first:break
 rows.append(first or {"golden_index":n,"no_divergence":True})
a.output.write_text(json.dumps({"scope":"LOCAL builtin vs LOCAL reference, NOT device",
 "golden_count":len(rows),"first_divergences":rows},indent=2))
print(a.output.read_text())
