"""Exploratory conditional gap-stratified bootstrap for original contrasts."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument("--predictions",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
d=pd.read_csv(a.predictions,parse_dates=["timestamp"])
y=d.observed_kwh.to_numpy();t=d.timestamp.to_numpy()
segments=np.split(np.arange(len(d)),np.flatnonzero(np.diff(t)!=np.timedelta64(1,"h"))+1)
out={"seed":20260929,"resamples":10000,"scope":"conditional on segments and trained model",
 "short_segment_policy":"retained as fixed when block >= segment length","comparisons":{}}
for second in ["fp32_kwh","int8_local_kwh"]:
 aa=d.esp32_kwh.to_numpy();bb=d[second].to_numpy()
 comparisons={}
 for length in (24,48,168):
  rng=np.random.default_rng(20260929);values=[]
  for _ in range(10000):
   idx=[]
   for seg in segments:
    n=min(length,len(seg));starts=rng.choice(seg[:len(seg)-n+1],size=int(np.ceil(len(seg)/n)),replace=True)
    idx.append((starts[:,None]+np.arange(n)).ravel()[:len(seg)])
   ind=np.concatenate(idx);ya=y[ind];ea=aa[ind]-ya;eb=bb[ind]-ya
   values.append([np.mean(np.abs(ea)-np.abs(eb)),
     np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2)),
     100*np.mean((np.abs(ea)-np.abs(eb))/ya)])
  ci=np.percentile(values,[2.5,97.5],axis=0)
  comparisons[str(length)]={k:{"lower":float(ci[0,i]),"upper":float(ci[1,i])}
    for i,k in enumerate(["MAE_kwh","RMSE_kwh","MAPE_pp"])}
 out["comparisons"]["esp32 minus "+second]=comparisons
a.output.write_text(json.dumps(out,indent=2))
