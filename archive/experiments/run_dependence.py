"""Facial-primitive copula sensitivity; preserves original location/quality draws.

Run from any directory. No original results are overwritten. Frozen models are
the archived extension fits; matched models use the same selection procedure.
"""
from pathlib import Path
import json, itertools, time, platform, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
ns={'__file__':str(ROOT/'run_extension.py')}
exec((ROOT/'run_extension.py').read_text().split('rows=[]; scenario_rows=[]')[0],ns)
sim,usable,x,rule,metrics,threshold = [ns[k] for k in ['sim','usable','x','rule','metrics','threshold']]
regimes=ns['regimes']
OUT=ROOT/'dependence'; OUT.mkdir(exist_ok=True)
SEED=20261012
CONFIGS=[('Independent','independent',0.),('Gaussian 0.3','gaussian',.3),('Gaussian 0.6','gaussian',.6),('t4 0.0','t',0.),('t4 0.3','t',.3),('t4 0.6','t',.6)]
PRIMITIVES=['similarity','consistency','area_span','nose_span','landmark_motion','light_span']

def generate(n,regime,seed,family,rho,boundary=False,diagnostics=False):
    d=sim(n,regime,seed)
    rng=np.random.default_rng(seed+4000000)
    normals=rng.standard_normal((n,6))
    omega=np.full((6,6),rho); np.fill_diagonal(omega,1.)
    z=normals@np.linalg.cholesky(omega).T
    scale=rng.chisquare(4,n)
    u=stats.t.cdf(z/np.sqrt(scale[:,None]/4),4) if family=='t' else stats.norm.cdf(z)
    u=np.clip(u,1e-12,1-1e-12)
    sc=d.scenario.to_numpy()
    imp=np.isin(sc,['impostor_inside','impostor_outside'])
    live=~np.isin(sc,['spoof_inside','spoof_boundary'])
    qo=d.face_quality_score.to_numpy()
    mu=np.where(imp,np.clip(.18+.13*qo,.16,.34),np.clip(.46+.30*qo,.42,.80))
    concentration=np.where(imp,26,30)
    s=stats.beta.ppf(u[:,0],mu*concentration,(1-mu)*concentration)
    cmu=np.where(live,.89,.94); cc=np.where(live,35,45)
    c=stats.beta.ppf(u[:,1],cmu*cc,(1-cmu)*cc)
    sm=regimes[regime]['spoof_motion_mult']
    spans=[]
    for j,(al,sl,ap,sp) in enumerate([(2.2,.009,1.7,.0024),(2.1,.014,1.7,.003),(2.2,.011,1.7,.0025),(2.,2.8,1.6,.90)]):
        spans.append(stats.gamma.ppf(u[:,j+2],np.where(live,al,ap),scale=np.where(live,sl,sp*sm)))
    a,no,land,light=spans
    motion=np.minimum(1,.25*a/.025+.35*no/.035+.30*land/.035+.10*np.minimum(light/8,1))
    ell=np.clip(.6*motion+.4*np.clip((c-.65)/.30,0,1),0,1)
    passive=(c>=.78)&(ell>=.42)
    face=d.quality_pass.to_numpy().astype(bool)&(s>=.40)
    eligible=face&(~passive)
    challenge=eligible&(rng.random(n)<np.where(live,.90,.12))
    ell[challenge]=np.maximum(ell[challenge],.62)
    lp=passive|(challenge&(c>=.78))
    d['face_similarity']=s; d['frame_consistency']=c; d['motion_score']=motion
    d['liveness_score']=ell; d['face_pass']=face.astype(int); d['liveness_pass']=lp.astype(int)
    if boundary:
        rb=np.random.default_rng(seed+900000)
        dt=d.true_distance_m.to_numpy().copy()
        m=sc=='genuine_outside'; dt[m]=150+rb.uniform(.1,10,m.sum())
        m=sc=='genuine_boundary_inside'; dt[m]=150-rb.uniform(.1,10,m.sum())
        err=rb.normal(size=(n,2))*d.gps_accuracy_m.to_numpy()[:,None]/2.448
        d['true_distance_m']=dt; d['observed_distance_m']=np.sqrt((dt+err[:,0])**2+err[:,1]**2)
        d['effective_distance_m']=d.observed_distance_m+d.gps_accuracy_m
        d['geo_pass']=((d.effective_distance_m<=150)&(d.gps_quality_ok==1)).astype(int)
    d['M2_face']=face.astype(int); d['M3_geo_face']=d.geo_pass*face
    d['M4_current_attendpro']=d.geo_pass*face*lp
    if diagnostics: return d,u,np.column_stack([s,c,*spans])
    return d

def pooled(n,seed,family,rho):
    return pd.concat([generate(n,r,seed+i,family,rho) for i,r in enumerate(regimes)],ignore_index=True)

def scores(d,ob):
    gate=usable(d); p=np.zeros(len(d))
    raw=ob['model'].predict_proba(x(d.loc[gate],ob['kind']))[:,1]
    raw=np.clip(raw,1e-6,1-1e-6)
    p[gate]=ob['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
    return p

def fit_models(train,val,cal,fit):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler,SplineTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss
    specs=[('Logistic interactions','interactions',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500))) for c in [.1,1,10]]
    specs += [('Logistic additive','base',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))),('Logistic no reliability','no_reliability',make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)))]
    specs += [('GAM additive','base',make_pipeline(SplineTransformer(n_knots=k,degree=3),StandardScaler(),LogisticRegression(C=1,max_iter=700))) for k in [4,6]]
    specs += [('Random forest','base',RandomForestClassifier(n_estimators=120,min_samples_leaf=l,max_features=1.,n_jobs=2,random_state=SEED+fit)) for l in [10,40]]
    specs += [('Gradient boosting','base',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=l,l2_regularization=1,random_state=SEED+fit)) for l in [15,31]]
    models={}
    for name,kind,m in specs:
        m.fit(x(train.loc[usable(train)],kind),train.loc[usable(train),'truth_valid_presence'])
        p=m.predict_proba(x(val.loc[usable(val)],kind))[:,1]
        loss=brier_score_loss(val.loc[usable(val),'truth_valid_presence'],p)
        if name not in models or loss<models[name]['loss']: models[name]=dict(model=m,kind=kind,loss=loss)
    for ob in models.values():
        raw=ob['model'].predict_proba(x(cal.loc[usable(cal)],ob['kind']))[:,1]
        raw=np.clip(raw,1e-6,1-1e-6)
        ob['calibrator']=LogisticRegression(C=1e6,max_iter=300).fit(np.log(raw/(1-raw)).reshape(-1,1),cal.loc[usable(cal),'truth_valid_presence'])
        ob['thresholds']={'FAR_1pct':threshold(val.truth_valid_presence.to_numpy(),scores(val,ob),.01)}
    best=(-1,None)
    for setting in itertools.product([0,.5,1],[140,150,160],[.35,.4,.5,.6],[.42,.62,.72,.82,.90,.95],[.70,.78]):
        mm=metrics(val.truth_valid_presence.to_numpy(),rule(val,setting),.5)
        if mm['FAR']<=.01 and 1-mm['FRR']>best[0]: best=(1-mm['FRR'],setting)
    return models, best[1] if best[1] is not None else (1,0,1,1,1)

def main():
    rows=[]; diags=[]; marginal=[]; selected=[]; paired=[]
    archived=json.loads((ROOT/'extension/selected_models_and_rules.json').read_text())
    start=time.time()
    for fit in range(3):
        frozen=joblib.load(ROOT/f'extension/models_fit_{fit}.joblib')
        frozen_rule=next(a['settings']['FAR_1pct'] for a in archived if a['fit']==fit and a['method']=='Tuned hard rule')
        for config,family,rho in CONFIGS:
            print('TRAIN',fit,config,flush=True)
            splitseed=SEED+fit*10000
            train=pooled(12000,splitseed,family,rho)
            val=pooled(6000,splitseed+100,family,rho)
            cal=pooled(6000,splitseed+200,family,rho)
            models,tuned=fit_models(train,val,cal,fit)
            selected.append(dict(fit=fit,config=config,tuned_rule=tuned,models={k:dict(parameters=str(o['model']),validation_brier=o['loss'],threshold=o['thresholds']['FAR_1pct']) for k,o in models.items()}))
            for population in ['nominal','boundary_conflict']:
                for batch in range(5):
                    seed=SEED+fit*10000+1000+(0 if population=='nominal' else 100)+batch
                    d,u,primitives=generate(20000,'nominal',seed,family,rho,population=='boundary_conflict',True)
                    y=d.truth_valid_presence.to_numpy(); sc=d.scenario.to_numpy()
                    if population=='nominal':
                        for j,name in enumerate(PRIMITIVES):
                            marginal.append(dict(fit=fit,batch=batch,config=config,primitive=name,pit_KS=stats.kstest(u[:,j],'uniform').statistic,mean=primitives[:,j].mean(),q05=np.quantile(primitives[:,j],.05),q50=np.quantile(primitives[:,j],.5),q95=np.quantile(primitives[:,j],.95)))
                        diags.append(dict(fit=fit,batch=batch,config=config,usable_rate=usable(d).mean(),quality_mean=d.face_quality_score.mean(),similarity_mean=d.face_similarity.mean(),consistency_mean=d.frame_consistency.mean(),liveness_mean=d.liveness_score.mean(),liveness_q05=d.liveness_score.quantile(.05),liveness_q95=d.liveness_score.quantile(.95),pit_tau_s_c=stats.kendalltau(u[:,0],u[:,1]).statistic,rank_corr_s_l=stats.spearmanr(d.face_similarity,d.liveness_score).statistic,challenge_rate=float(((d.face_pass==1)&(~((d.frame_consistency>=.78)&((.6*d.motion_score+.4*np.clip((d.frame_consistency-.65)/.30,0,1))>=.42)))).mean())))
                    preds={('reference','Current hard rule'):d.M4_current_attendpro.to_numpy().astype(bool)}
                    for protocol,ms,rr in [('frozen',frozen,frozen_rule),('matched',models,tuned)]:
                        preds[(protocol,'Tuned hard rule')]=rule(d,rr)
                        for name,ob in ms.items(): preds[(protocol,name)]=scores(d,ob)>=ob['thresholds']['FAR_1pct']
                    for (protocol,name),z in preds.items():
                        m_out=sc=='genuine_outside'; m_in=sc=='genuine_boundary_inside'; m_imp=sc=='impostor_inside'; m_spoof=np.isin(sc,['spoof_inside','spoof_boundary'])
                        rows.append(dict(fit=fit,batch=batch,config=config,family=family,rho=rho,population=population,protocol=protocol,method=name,**metrics(y,z,.5),outside_accept=z[m_out].mean(),boundary_inside_reject=(~z[m_in]).mean(),inside_impostor_accept=z[m_imp].mean(),presentation_accept=z[m_spoof].mean()))
                        if protocol in ['frozen','matched'] and name!='Tuned hard rule':
                            base=preds[(protocol,'Tuned hard rule')]
                            for stratum,mask in [('invalid',y==0),('valid',y==1)]:
                                diff=z[mask].astype(int)-base[mask].astype(int)
                                paired.append(dict(fit=fit,batch=batch,config=config,population=population,protocol=protocol,method=name,stratum=stratum,acceptance_difference=diff.mean(),conditional_mcse=diff.std(ddof=1)/np.sqrt(len(diff))))
            pd.DataFrame(rows).to_csv(OUT/'batch_metrics.csv',index=False)
            pd.DataFrame(diags).to_csv(OUT/'batch_diagnostics.csv',index=False)
            pd.DataFrame(marginal).to_csv(OUT/'primitive_marginals.csv',index=False)
            pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
            (OUT/'selected_models.json').write_text(json.dumps(selected,indent=2))
            print('DONE',fit,config,'elapsed',round(time.time()-start),flush=True)
    r=pd.DataFrame(rows)
    keys=['config','family','rho','population','protocol','method']
    cols=['FAR','FRR','BA','outside_accept','boundary_inside_reject','inside_impostor_accept','presentation_accept']
    f=r.groupby(['fit',*keys])[cols].mean().reset_index(); f.to_csv(OUT/'fit_metrics.csv',index=False)
    f.groupby(keys)[cols].agg(['mean','min','max','std']).to_csv(OUT/'summary.csv')
    f.groupby(keys)[cols].mean().reset_index().to_csv(OUT/'mean_metrics.csv',index=False)
    pd.DataFrame(diags).groupby('config').mean(numeric_only=True).to_csv(OUT/'diagnostics.csv')
    (OUT/'manifest.json').write_text(json.dumps(dict(seed=SEED,configs=CONFIGS,primitive_block=PRIMITIVES,copula_df=4,train_per_regime=12000,val_per_regime=6000,cal_per_regime=6000,fits=3,test_batches=5,test_batch_n=20000,test_populations=['nominal','boundary_conflict'],generated_test_outcomes=3600000,underlying_paired_test_cases=600000,note='Same scenario/location/quality and base normals across copulas; derived liveness is recomputed. Test nominal/boundary samples are separate. Frozen=archived original conditional-innovation fits; matched=redevelopment under each copula; both development populations pool five capture regimes.',versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,sklearn=__import__('sklearn').__version__),seconds=time.time()-start),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
