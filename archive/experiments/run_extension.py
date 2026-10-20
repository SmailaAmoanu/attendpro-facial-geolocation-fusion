"""Reproducible paired extension; original generator loaded without execution."""
from pathlib import Path
import json, time, itertools, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import joblib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'extension'; OUT.mkdir(exist_ok=True)
ns={}
source=(ROOT/'AttendPro_original_simulation_recovered.py').read_text()
ns['__file__']=str(ROOT/'AttendPro_original_simulation_recovered.py')
exec(source.split('# Fit the fusion model on pooled')[0].replace('import nbformat as nbf',''),ns)
sim=ns['simulate_attendpro']; frame=ns['build_model_frame']; features=ns['FEATURES']; regimes=ns['REGIMES']
SEED=20261001

def usable(d): return ((d.gps_quality_ok==1)&(d.quality_pass==1)).to_numpy()
def x(d,kind):
 if kind=='no_reliability': return d[['observed_distance_m','face_similarity','liveness_score','frame_consistency']]
 if kind=='interactions': return frame(d)
 return d[features]
def pooled(n,seed): return pd.concat([sim(n,r,seed+i) for i,r in enumerate(regimes)],ignore_index=True)
def threshold(y,p,target=None):
 f,t,ths=roc_curve(y,p,drop_intermediate=False)
 if target is None: return float(ths[np.argmax(t-f)])
 ix=np.where(f<=target)[0]; best=ix[np.argmax(t[ix])]; return float(ths[best])
def metrics(y,p,t):
 z=p>=t; pos=y==1; neg=~pos
 far=z[neg].mean(); frr=(~z[pos]).mean()
 return dict(FAR=far,FRR=frr,BA=1-(far+frr)/2,n_valid=int(pos.sum()),n_invalid=int(neg.sum()),false_accept=int(z[neg].sum()),false_reject=int((~z[pos]).sum()))
def rule(d,c):
 w,rad,fs,ls,fc=c
 return (usable(d)&((d.observed_distance_m+w*d.gps_accuracy_m)<=rad)&(d.face_similarity>=fs)&(d.liveness_score>=ls)&(d.frame_consistency>=fc)).to_numpy() if isinstance(usable(d),pd.Series) else (usable(d)&((d.observed_distance_m+w*d.gps_accuracy_m).to_numpy()<=rad)&(d.face_similarity.to_numpy()>=fs)&(d.liveness_score.to_numpy()>=ls)&(d.frame_consistency.to_numpy()>=fc))
rows=[]; scenario_rows=[]; paired=[]; timings=[]; selected=[]
for fit_rep in range(3):
 print('FIT',fit_rep,flush=True)
 train=pooled(12000,SEED+fit_rep*10000)
 val=pooled(6000,SEED+fit_rep*10000+100)
 cal=pooled(6000,SEED+fit_rep*10000+200)
 models={}
 specs=[('Logistic interactions','interactions',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500))) for c in [0.1,1,10]]
 specs += [('Logistic additive','base',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))),('Logistic no reliability','no_reliability',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)))]
 specs += [('GAM additive','base',make_pipeline(SplineTransformer(n_knots=k,degree=3),StandardScaler(),LogisticRegression(C=1,max_iter=700))) for k in [4,6]]
 specs += [('Random forest','base',RandomForestClassifier(n_estimators=120,min_samples_leaf=l,max_features=1.0,n_jobs=2,random_state=SEED+fit_rep)) for l in [10,40]]
 specs += [('Gradient boosting','base',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=l,l2_regularization=1,random_state=SEED+fit_rep)) for l in [15,31]]
 for name,kind,m in specs:
  m.fit(x(train.loc[usable(train)],kind),train.loc[usable(train),'truth_valid_presence'])
  pv=m.predict_proba(x(val.loc[usable(val)],kind))[:,1]
  loss=brier_score_loss(val.loc[usable(val),'truth_valid_presence'],pv)
  if name not in models or loss<models[name]['loss']: models[name]=dict(model=m,kind=kind,loss=loss)
 for name,ob in models.items():
  # Independent sigmoid calibration on usable captures; gate remains a decision restriction.
  raw=ob['model'].predict_proba(x(cal.loc[usable(cal)],ob['kind']))[:,1]
  logit=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1)
  cm=LogisticRegression(C=1e6,max_iter=300).fit(logit,cal.loc[usable(cal),'truth_valid_presence'])
  ob['calibrator']=cm
 def scores(d,ob):
  u=usable(d); p=np.zeros(len(d)); raw=ob['model'].predict_proba(x(d.loc[u],ob['kind']))[:,1]
  logit=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1)
  p[u]=ob['calibrator'].predict_proba(logit)[:,1]; return p
 targets={'BA':None,'FAR_1pct':.01,'FAR_0.1pct':.001}
 for name,ob in models.items():
  pv=scores(val,ob); ob['thresholds']={op:threshold(val.truth_valid_presence.to_numpy(),pv,t) for op,t in targets.items()}
  selected.append(dict(fit=fit_rep,method=name,validation_brier_usable=ob['loss'],thresholds=ob['thresholds']))
 # Tune a finite conjunction family using the same validation attempts and gates.
 best={op:(-1,None) for op in targets}
 for c in itertools.product([0,.5,1],[140,150,160],[.35,.4,.5,.6],[.42,.62,.72,.82,.90,.95],[.70,.78]):
  z=rule(val,c); mm=metrics(val.truth_valid_presence.to_numpy(),z,.5)
  for op,t in targets.items():
   objective=mm['BA'] if t is None else 1-mm['FRR']
   if (t is None or mm['FAR']<=t) and objective>best[op][0]: best[op]=(objective,c)
 for op in best:
  if best[op][1] is None: best[op]=(0,(1,0,1,1,1))
 selected.append(dict(fit=fit_rep,method='Tuned hard rule',settings={k:v[1] for k,v in best.items()}))
 conditions=list(regimes)+['accuracy_underreported','heavy_tail','boundary_conflict','outsider_heavy']
 for cond in conditions:
  for rep in range(5):
   seed=SEED+fit_rep*10000+1000+conditions.index(cond)*20+rep
   if cond in regimes: d=sim(20000,cond,seed)
   else:
    key='extension_temp'; cfg=dict(regimes['nominal'])
    if cond=='accuracy_underreported': cfg['gps_error_mult']=2.5
    if cond=='outsider_heavy': cfg['weights']=[.08,.08,.08,.50,.08,.06,.06,.06]
    regimes[key]=cfg; d=sim(20000,key,seed); del regimes[key]
    if cond in ['heavy_tail','boundary_conflict']:
     rng=np.random.default_rng(seed+900000)
     dt=d.true_distance_m.to_numpy().copy()
     if cond=='boundary_conflict':
      m=d.scenario=='genuine_outside'; dt[m]=150+rng.uniform(.1,10,m.sum())
      m=d.scenario=='genuine_boundary_inside'; dt[m]=150-rng.uniform(.1,10,m.sum())
     sigma=d.gps_accuracy_m.to_numpy()/2.448
     if cond=='heavy_tail': err=rng.standard_t(3,(len(d),2))/np.sqrt(3)*sigma[:,None]
     else: err=rng.normal(size=(len(d),2))*sigma[:,None]
     d['true_distance_m']=dt; d['observed_distance_m']=np.sqrt((dt+err[:,0])**2+err[:,1]**2)
     d['geo_pass']=((d.observed_distance_m+d.gps_accuracy_m<=150)&(d.gps_quality_ok==1)).astype(int)
     d['M1_geofence']=d.geo_pass; d['M3_geo_face']=d.geo_pass*d.face_pass; d['M4_current_attendpro']=d.geo_pass*d.face_pass*d.liveness_pass
   y=d.truth_valid_presence.to_numpy(); preds={}
   for name,col in [('Geofence only','M1_geofence'),('Face only','M2_face'),('Geo plus face','M3_geo_face'),('Current hard rule','M4_current_attendpro')]: preds[(name,'deployed')]=(d[col].to_numpy(),.5)
   for op,(_,c) in best.items(): preds[('Tuned hard rule',op)]=(rule(d,c).astype(float),.5)
   for name,ob in models.items():
    tick=time.perf_counter(); p=scores(d,ob); elapsed=time.perf_counter()-tick
    timings.append(dict(fit=fit_rep,condition=cond,rep=rep,method=name,seconds=elapsed,n=len(d)))
    for op,t in ob['thresholds'].items(): preds[(name,op)]=(p,t)
    if fit_rep==0 and rep==0 and cond=='nominal':
     u=usable(d); frac,mean=calibration_curve(y[u],p[u],n_bins=10,strategy='quantile'); pd.DataFrame(dict(predicted=mean,observed=frac)).to_csv(OUT/(name.replace(' ','_')+'_calibration.csv'),index=False)
   for (name,op),(p,t) in preds.items():
    mm=metrics(y,p,t); rr=dict(fit=fit_rep,condition=cond,rep=rep,method=name,operating_point=op,**mm)
    if name in models:
     u=usable(d); rr.update(AUC_usable=roc_auc_score(y[u],p[u]),Brier_usable=brier_score_loss(y[u],p[u]))
    rows.append(rr)
    for sc in np.unique(d.scenario):
     m=(d.scenario==sc).to_numpy(); scenario_rows.append(dict(fit=fit_rep,condition=cond,rep=rep,method=name,operating_point=op,scenario=sc,n=int(m.sum()),acceptance_rate=float((p[m]>=t).mean()),valid=int(y[m].sum())))
    if op in targets and name!='Tuned hard rule':
     bp,bt=preds[('Tuned hard rule',op)]; z=(p>=t).astype(int); bz=(bp>=bt).astype(int)
     for label,m in [('invalid',y==0),('valid',y==1)]:
      diff=z[m]-bz[m]; paired.append(dict(fit=fit_rep,condition=cond,rep=rep,method=name,operating_point=op,stratum=label,acceptance_difference=float(diff.mean()),conditional_mcse=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
   if fit_rep==0 and rep==0 and cond=='nominal': d.to_csv(OUT/'nominal_test_sample.csv',index=False)
  print('finished',fit_rep,cond,flush=True)
 joblib.dump(models,OUT/f'models_fit_{fit_rep}.joblib')
 pd.DataFrame(rows).to_csv(OUT/'replicate_metrics.csv',index=False)
 pd.DataFrame(scenario_rows).to_csv(OUT/'scenario_metrics.csv',index=False)
 pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
 pd.DataFrame(timings).to_csv(OUT/'inference_times.csv',index=False)
 (OUT/'selected_models_and_rules.json').write_text(json.dumps(selected,indent=2))
res=pd.DataFrame(rows)
res.groupby(['condition','method','operating_point'])[['FAR','FRR','BA','Brier_usable','AUC_usable']].agg(['mean','std']).to_csv(OUT/'summary.csv')
# Fit-level averages distinguish between-fit variation from conditional test Monte Carlo noise.
res.groupby(['fit','condition','method','operating_point'])[['FAR','FRR','BA']].mean().to_csv(OUT/'fit_level_metrics.csv')
fig,ax=plt.subplots(figsize=(7,6))
for path in OUT.glob('*_calibration.csv'):
 z=pd.read_csv(path); ax.plot(z.predicted,z.observed,marker='o',label=path.stem.replace('_calibration','').replace('_',' '))
ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Predicted probability (usable captures)',ylabel='Observed valid fraction'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'calibration.png',dpi=180); plt.close(fig)
print('COMPLETE',flush=True)
