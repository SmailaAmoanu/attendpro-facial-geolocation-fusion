from pathlib import Path
import json,hashlib,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'boundary_training'
r=pd.read_csv(OUT/'metrics.csv');g=pd.read_csv(OUT/'subgroup_metrics.csv');p=pd.read_csv(OUT/'paired_differences.csv')
assert len(r)==3660,len(r)
assert r.groupby(['fit','condition','rep']).size().eq(61).all()
assert np.allclose(r.BA,1-(r.FAR+r.FRR)/2)
assert np.allclose(r.FAR,r.false_accept/r.n_invalid)
assert np.allclose(r.FRR,r.false_reject/r.n_valid)
assert r[['FAR','FRR','BA']].notna().all().all()
mean=r.groupby(['condition','method','arm','operating_point'])[['FAR','FRR','BA']].mean().reset_index();mean.to_csv(OUT/'mean_metrics.csv',index=False)
gmean=g.groupby(['condition','method','arm','operating_point','group']).acceptance_rate.mean().reset_index()
q=gmean[(gmean.condition=='near_boundary')&(gmean.operating_point=='FAR_1pct')&(gmean.group.isin(['genuine_outside','boundary_inside']))].pivot(index=['method','arm'],columns='group',values='acceptance_rate').reset_index()
z=mean[(mean.condition=='near_boundary')&(mean.operating_point=='FAR_1pct')].merge(q,on=['method','arm'])
z['boundary_inside_rejection']=1-z.boundary_inside
z.to_csv(OUT/'near_boundary_key_results.csv',index=False)
# Wilson intervals conditional on each fitted model and test batch.
for rate,num,den in [('FAR','false_accept','n_invalid'),('FRR','false_reject','n_valid')]:
 phat=r[num]/r[den];n=r[den];zz=1.959963984540054
 c=(phat+zz*zz/(2*n))/(1+zz*zz/n);h=zz*np.sqrt(phat*(1-phat)/n+zz*zz/(4*n*n))/(1+zz*zz/n)
 r[rate+'_wilson_low']=c-h;r[rate+'_wilson_high']=c+h
r.to_csv(OUT/'metrics_with_intervals.csv',index=False)
bf=p.groupby(['fit','condition','method','arm','operating_point','group']).difference.mean().reset_index()
bf.groupby(['condition','method','arm','operating_point','group']).difference.agg(['mean','std','min','max']).to_csv(OUT/'paired_between_fit_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
names=['Logistic interactions','GAM additive','Random forest','Gradient boosting'];arms=['Frozen original','Threshold adapted','Boundary trained']
for ax,col,title in zip(axs,['genuine_outside','boundary_inside_rejection'],['Genuine outsiders accepted','Legitimate boundary users rejected']):
 for k,arm in enumerate(arms):
  zz=z[z.arm==arm].set_index('method').loc[names];ax.bar(np.arange(4)+(k-1)*.24,100*zz[col],width=.24,label=arm)
 ax.set_xticks(np.arange(4));ax.set_xticklabels(['Logistic','GAM','Random forest','Boosting']);ax.set_ylabel('Rate (%)');ax.set_title(title);ax.legend(fontsize=8)
fig.suptitle('Independent near-boundary test; validation-selected 1% aggregate FAR');fig.tight_layout();fig.savefig(OUT/'boundary_tradeoff.png',dpi=180);plt.close(fig)
def table(d,cols):
 return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(f'{100*v[c]:.2f}%' if c not in ['method','arm'] else str(v[c]) for c in cols)+' |' for _,v in d.iterrows())
report='''AttendPro Paper II — boundary-training experiment

Completed scope

Three independent fitting runs compare six fusion specifications: logistic interactions, additive logistic, logistic without reliability covariates, additive spline logistic GAM, random forest, and histogram gradient boosting. All models retain GPS usability and face-quality gates. The original fit artifacts and their original thresholds are read-only references.

Three learned-model arms:
1. Frozen original: original fitted model, calibrator, and operating thresholds.
2. Threshold adapted: same original fitted model and calibrator; only thresholds are chosen on boundary-enriched validation data.
3. Boundary trained: model selected and trained on boundary-enriched data, calibrated on its separate calibration split, and thresholded on its validation split. This arm evaluates the whole adapted learning pipeline, not training changes alone.

Training sample size and raw capture draws match each original fit: 60,000 training, 30,000 validation, and 30,000 calibration attempts pooled equally over the original five regimes. Boundary enrichment moves 50% of genuine outsider attempts to 150 + U(0.1,10) m and genuine boundary-inside attempts to 150 - U(0.1,10) m. Only the changed location measurements are regenerated. Identity, facial quality, similarity, and liveness evidence are preserved. The positive label continues to require correct represented identity, live capture, and true distance <=150 m. The protocol/session checks are outside this statistical simulation.

Four independent test populations: nominal, boundary-enriched (same type of enrichment), near-boundary (all genuine outsiders moved to 0.1–10 m outside), and near-boundary with median GPS accuracy 25 m and lognormal sigma 0.55. The last condition regenerates all location measurements; other evidence is preserved. Each population has five batches of 30,000 attempts per fit: 1,800,000 unique test attempts overall. All methods/arms receive the same attempts within a batch.

Operating points maximize validation balanced accuracy, or maximize sensitivity subject to empirical validation FAR <=1% or <=0.1%. The same 432-setting hard-rule family is tuned on enriched validation and compared with the original tuned configurations and deployed rule. Candidate model families/searches are the same as the preceding experiment. No final test labels enter tuning.

Near-boundary results at the validation-selected 1% FAR setting

These averages use 15 independent test batches conditional on three fitted model sets. "genuine_outside" is acceptance among genuine live outsiders. Boundary-inside rejection is shown separately from rejection among all legitimate attempts. Targets refer to validation and are not guarantees on shifted test data.

'''+table(z[z.method.isin(names+['Tuned hard rule'])],['method','arm','FAR','FRR','genuine_outside','boundary_inside_rejection'])+'''

Interpretation and limits

Compare Boundary trained against Threshold adapted to assess whether the adapted learning pipeline adds value beyond threshold conservatism. Compare both against Frozen original to assess the failure of transporting the original decision setting into a new boundary population. Lower outsider acceptance must be assessed alongside legitimate boundary rejection, not as an unconditional gain.

The experiment can identify sensitivity to training/validation coverage. It cannot estimate an irreducible Bayes-error bound or prove how much remaining error is unavoidable GPS ambiguity. Reported accuracy is synthetic and calibrated by assumption; additional stress checks here change its magnitude, not coordinate integrity. No location-spoofing resistance or production error rates are established.

Wilson intervals are conditional per-batch intervals. Paired differences and conditional Monte Carlo standard errors share the same test attempts. Between-fit summaries use only three independent training runs and should not be mistaken for a precise population confidence claim. Models and thresholds are frozen within testing. This study covers the original identity/liveness scenario mixture; the earlier eight-cell transfer experiment remains a separate test.

Reproduction

First run run_extension.py if original model artifacts are missing. Then run:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_boundary_training.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report_boundary_training.py

The cumulative package includes the preceding historical reruns and extension plus this follow-up. The original EXPERIMENT_REPORT.md is preserved; read this boundary report for the new results. No manuscript changes or software deployment were performed.
'''
(OUT/'Boundary_Experiment_Report.txt').write_text(report)
print(z[z.method.isin(names+['Tuned hard rule'])][['method','arm','FAR','FRR','genuine_outside','boundary_inside_rejection']].to_string(index=False))
# Check that reference fit artifacts retain the pre-existing manifest hashes.
old=json.loads((ROOT/'SHA256SUMS.json').read_text())
for k,v in old.items():
 if k.startswith('extension/models_fit_'):assert hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v
files=[a for a in ROOT.rglob('*') if a.is_file() and a.name!='SHA256SUMS.json' and a.suffix!='.zip' and '__pycache__' not in str(a)]
(ROOT/'SHA256SUMS.json').write_text(json.dumps({str(a.relative_to(ROOT)):hashlib.sha256(a.read_bytes()).hexdigest() for a in files},indent=2))
print('VERIFIED',len(r),'boundary metric rows')
