"""Boundary training, threshold-only adaptation, and frozen-reference comparison."""
from pathlib import Path
import json,time,hashlib,joblib
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss
ROOT=Path(__file__).resolve().parent
helpers=(ROOT/'run_extension.py').read_text().split('rows=[];')[0]
exec(helpers)
OUT=ROOT/'boundary_training';OUT.mkdir(exist_ok=True)

def boundary(d,seed,fraction=.5,high_noise=False):
 d=d.copy();rng=np.random.default_rng(seed); dt=d.true_distance_m.to_numpy().copy()
 outside=(d.scenario=='genuine_outside').to_numpy();inside=(d.scenario=='genuine_boundary_inside').to_numpy()
 if fraction<1: outside &= rng.random(len(d))<fraction
 dt[outside]=150+rng.uniform(.1,10,outside.sum());dt[inside]=150-rng.uniform(.1,10,inside.sum())
 changed=outside|inside
 acc=d.gps_accuracy_m.to_numpy().copy()
 if high_noise: acc=np.clip(rng.lognormal(np.log(25),.55,len(d)),1,140);changed=np.ones(len(d),dtype=bool)
 sigma=acc[changed]/2.448;err=rng.normal(size=(changed.sum(),2))*sigma[:,None]
 obs=d.observed_distance_m.to_numpy().copy();obs[changed]=np.sqrt((dt[changed]+err[:,0])**2+err[:,1]**2)
 d['true_distance_m']=dt;d['observed_distance_m']=obs;d['gps_accuracy_m']=acc;d['effective_distance_m']=obs+acc
 d['gps_quality_ok']=(acc<=50).astype(int);d['geo_pass']=((obs+acc<=150)&(acc<=50)).astype(int)
 d['M1_geofence']=d.geo_pass;d['M3_geo_face']=d.geo_pass*d.face_pass;d['M4_current_attendpro']=d.geo_pass*d.face_pass*d.liveness_pass
 return d

def scores(d,ob):
 u=usable(d);p=np.zeros(len(d));raw=ob['model'].predict_proba(x(d.loc[u],ob['kind']))[:,1];z=np.clip(raw,1e-6,1-1e-6)
 p[u]=ob['calibrator'].predict_proba(np.log(z/(1-z)).reshape(-1,1))[:,1];return p

def specs(fit):
 v=[('Logistic interactions','interactions',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500))) for c in [.1,1,10]]
 v += [('Logistic additive','base',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))),('Logistic no reliability','no_reliability',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)))]
 v += [('GAM additive','base',make_pipeline(SplineTransformer(n_knots=k,degree=3),StandardScaler(),LogisticRegression(C=1,max_iter=700))) for k in [4,6]]
 v += [('Random forest','base',RandomForestClassifier(n_estimators=120,min_samples_leaf=l,max_features=1.,n_jobs=2,random_state=SEED+fit)) for l in [10,40]]
 v += [('Gradient boosting','base',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=l,l2_regularization=1,random_state=SEED+fit)) for l in [15,31]]
 return v

targets={'BA':None,'FAR_1pct':.01,'FAR_0.1pct':.001}
rows=[];subgroups=[];pairs=[];settings=[]
old_settings=json.loads((ROOT/'extension/selected_models_and_rules.json').read_text())
for fit in range(3):
 print('FIT',fit,flush=True)
 originals=joblib.load(ROOT/'extension'/f'models_fit_{fit}.joblib')
 train=boundary(pooled(12000,SEED+fit*10000),SEED+700000+fit*100)
 val=boundary(pooled(6000,SEED+fit*10000+100),SEED+700001+fit*100)
 cal=boundary(pooled(6000,SEED+fit*10000+200),SEED+700002+fit*100)
 learned={}
 for name,kind,m in specs(fit):
  m.fit(x(train.loc[usable(train)],kind),train.loc[usable(train),'truth_valid_presence'])
  pv=m.predict_proba(x(val.loc[usable(val)],kind))[:,1];loss=brier_score_loss(val.loc[usable(val),'truth_valid_presence'],pv)
  if name not in learned or loss<learned[name]['loss']:learned[name]=dict(model=m,kind=kind,loss=loss)
 for name,ob in learned.items():
  raw=ob['model'].predict_proba(x(cal.loc[usable(cal)],ob['kind']))[:,1];z=np.clip(raw,1e-6,1-1e-6)
  ob['calibrator']=LogisticRegression(C=1e6,max_iter=300).fit(np.log(z/(1-z)).reshape(-1,1),cal.loc[usable(cal),'truth_valid_presence'])
  p=scores(val,ob);ob['thresholds']={op:threshold(val.truth_valid_presence.to_numpy(),p,t) for op,t in targets.items()}
 adapt={name:{op:threshold(val.truth_valid_presence.to_numpy(),scores(val,ob),t) for op,t in targets.items()} for name,ob in originals.items()}
 best={op:(-1,None) for op in targets}
 for c in itertools.product([0,.5,1],[140,150,160],[.35,.4,.5,.6],[.42,.62,.72,.82,.90,.95],[.70,.78]):
  mm=metrics(val.truth_valid_presence.to_numpy(),rule(val,c),.5)
  for op,t in targets.items():
   obj=mm['BA'] if t is None else 1-mm['FRR']
   if (t is None or mm['FAR']<=t) and obj>best[op][0]:best[op]=(obj,c)
 for op in best:
  if best[op][1] is None:best[op]=(0,(1,0,1,1,1))
 oldrule=next(z['settings'] for z in old_settings if z['fit']==fit and z['method']=='Tuned hard rule')
 settings.append(dict(fit=fit,adapted_thresholds=adapt,boundary_thresholds={n:o['thresholds'] for n,o in learned.items()},boundary_rules={k:v[1] for k,v in best.items()}))
 joblib.dump(learned,OUT/f'boundary_models_fit_{fit}.joblib')
 for ci,cond in enumerate(['nominal','boundary_enriched','near_boundary','near_boundary_noisy']):
  for rep in range(5):
   seed=SEED+800000+fit*1000+ci*20+rep;d=sim(30000,'nominal',seed)
   if cond!='nominal':d=boundary(d,seed+90000,fraction=.5 if cond=='boundary_enriched' else 1,high_noise=cond=='near_boundary_noisy')
   y=d.truth_valid_presence.to_numpy();assert np.array_equal(y,(d.scenario.isin(['genuine_inside','genuine_boundary_inside','genuine_degraded_inside'])&(d.true_distance_m<=150)).astype(int))
   predictions={}
   for name in originals:
    po=scores(d,originals[name]);pn=scores(d,learned[name])
    for op in targets:
     predictions[(name,'Frozen original',op)]=(po,originals[name]['thresholds'][op])
     predictions[(name,'Threshold adapted',op)]=(po,adapt[name][op])
     predictions[(name,'Boundary trained',op)]=(pn,learned[name]['thresholds'][op])
   for op in targets:
    predictions[('Tuned hard rule','Frozen original',op)]=(rule(d,oldrule[op]).astype(float),.5)
    predictions[('Tuned hard rule','Boundary tuned',op)]=(rule(d,best[op][1]).astype(float),.5)
   predictions[('Current hard rule','Deployed','deployed')]=(d.M4_current_attendpro.to_numpy(),.5)
   groups={'genuine_outside':(d.scenario=='genuine_outside').to_numpy(),'boundary_inside':(d.scenario=='genuine_boundary_inside').to_numpy(),'other_legitimate':(y==1)&(d.scenario!='genuine_boundary_inside').to_numpy(),'impostors':d.scenario.isin(['impostor_inside','impostor_outside']).to_numpy(),'presentation_material':d.scenario.isin(['spoof_inside','spoof_boundary']).to_numpy()}
   for lo,hi in [(.1,2),(2,5),(5,10)]:
    delta=d.true_distance_m.to_numpy()-150;groups[f'outside_{lo}_{hi}m']=groups['genuine_outside']&(delta>=lo)&(delta<hi)
   for (name,arm,op),(p,t) in predictions.items():
    z=p>=t;rows.append(dict(fit=fit,condition=cond,rep=rep,method=name,arm=arm,operating_point=op,**metrics(y,p,t)))
    for g,m in groups.items():
     if m.any():subgroups.append(dict(fit=fit,condition=cond,rep=rep,method=name,arm=arm,operating_point=op,group=g,n=int(m.sum()),accepted=int(z[m].sum()),acceptance_rate=float(z[m].mean())))
    if name in learned and arm!='Frozen original':
     bp,bt=predictions[(name,'Frozen original',op)];bz=bp>=bt
     for g,m in {'valid':y==1,'invalid':y==0,**groups}.items():
      if m.sum()>1:
       diff=z[m].astype(int)-bz[m].astype(int);pairs.append(dict(fit=fit,condition=cond,rep=rep,method=name,arm=arm,operating_point=op,group=g,difference=float(diff.mean()),conditional_mcse=float(diff.std(ddof=1)/np.sqrt(m.sum()))))
  print('finished',fit,cond,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(subgroups).to_csv(OUT/'subgroup_metrics.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
r=pd.DataFrame(rows);r.groupby(['condition','method','arm','operating_point'])[['FAR','FRR','BA']].agg(['mean','std']).to_csv(OUT/'summary.csv')
r.groupby(['fit','condition','method','arm','operating_point'])[['FAR','FRR','BA']].mean().to_csv(OUT/'fit_level_summary.csv')
pd.DataFrame(subgroups).groupby(['condition','method','arm','operating_point','group']).acceptance_rate.agg(['mean','std']).to_csv(OUT/'subgroup_summary.csv')
print('COMPLETE',flush=True)
