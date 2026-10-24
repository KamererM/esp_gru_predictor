import os, sys, json, time
import numpy as np, pandas as pd
import rp, tensorflow as tf
from sklearn.preprocessing import StandardScaler
seed=int(sys.argv[1]); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
rp.set_reproducibility(seed)
cfg=rp.Config(seed=seed)
h=pd.read_csv('../data/hourly_splits.csv',parse_dates=['timestamp']).set_index('timestamp')
splits={k:h[h.split==k][['energy_kwh']] for k in ['train','validation','test']}
scaler=StandardScaler().fit(splits['train'][['energy_kwh']].to_numpy())
P={k:rp.make_windows(splits[k],scaler,cfg.steps) for k in splits}
Xtr,ytr,_,_,Xtr_raw=P['train']; Xva,yva,_,_,_=P['validation']; Xte,_,yte,tte,Xte_raw=P['test']
tr=tf.data.Dataset.from_tensor_slices((Xtr,ytr)).shuffle(len(Xtr),seed=seed,reshuffle_each_iteration=True).batch(32).prefetch(1)
va=tf.data.Dataset.from_tensor_slices((Xva,yva)).batch(32).prefetch(1)
m=rp.build_model(cfg); t0=time.time()
es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=cfg.patience,restore_best_weights=True)
hist=m.fit(tr,validation_data=va,epochs=cfg.max_epochs,callbacks=[es],verbose=0)
vl=np.array(hist.history['val_loss'])
dm=rp.build_model(cfg,batch_size=1); dm.set_weights(m.get_weights())
tfl=os.path.join(out,'model_int8.tflite'); rp.convert_int8(dm,Xtr,cfg,__import__('pathlib').Path(tfl))
fp,_=rp.evaluate_fp32(m,Xte,scaler,warmup=0)
i8,_,det,qin,qout=rp.evaluate_tflite(__import__('pathlib').Path(tfl),Xte_raw,scaler,warmup=0)
e=np.load('../data/extension_windows.npz'); Xe_raw=e['X_raw_kwh']; ye=e['target_kwh']
Xe=rp.normalize_float32(Xe_raw,scaler)[...,None]
fpe=rp.denormalize_float32(m.predict(Xe,batch_size=256,verbose=0).ravel(),scaler).astype(np.float64)
i8e,_,_,_,_=rp.evaluate_tflite(__import__('pathlib').Path(tfl),Xe_raw,scaler,warmup=0)
M=lambda y,p: rp.regression_metrics(y,p,1e-6)
res=dict(seed=seed,init_base=rp.INIT_BASE,epochs_run=len(vl),best_epoch=int(vl.argmin())+1,best_val_loss=float(vl.min()),
 train_seconds=time.time()-t0, n_params=int(m.count_params()),
 test_fp32=M(yte,fp),test_int8=M(yte,i8),ext_fp32=M(ye,fpe),ext_int8=M(ye,i8e),
 quant={k:det[k] for k in ['input_scale','input_zero_point','output_scale','output_zero_point']},
 int8_vs_fp32_mean_abs_kwh=float(np.abs(i8-fp).mean()), int8_vs_fp32_max_abs_kwh=float(np.abs(i8-fp).max()), model_bytes=os.path.getsize(tfl))
pd.DataFrame({'timestamp':pd.to_datetime(tte),'observed_kwh':yte,'fp32_kwh':fp,'int8_local_kwh':i8,'q_out':qout}).to_csv(os.path.join(out,'test_predictions.csv'),index=False)
pd.DataFrame(hist.history).to_csv(os.path.join(out,'history.csv'),index_label='epoch')
json.dump(res,open(os.path.join(out,'summary.json'),'w'),indent=1)
print(json.dumps({k:res[k] for k in ['seed','epochs_run','best_epoch','best_val_loss','train_seconds']}), res['test_fp32']['mae_kwh'], res['test_int8']['mae_kwh'])
