"""Structural and numerical checks for the dependence extension."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_dependence as r

def verify():
    original=r.sim(10000,'nominal',123)
    generated=[]
    for config,family,rho in r.CONFIGS:
        d,u,p=r.generate(10000,'nominal',123,family,rho,diagnostics=True)
        for col in ['scenario','truth_valid_presence','true_distance_m','gps_accuracy_m','observed_distance_m','quality_pass','face_quality_score','detector_confidence','brightness','blur_score','face_area_ratio','nose_offset_x']:
            assert np.array_equal(d[col],original[col]),(config,col)
        assert np.all((u>0)&(u<1))
        assert np.all((d.liveness_score>=0)&(d.liveness_score<=1))
        assert np.array_equal(r.rule(d,(1,150,.4,.42,.78)),d.M4_current_attendpro.astype(bool))
        assert np.array_equal(r.usable(d),r.usable(original))
        db=r.generate(10000,'nominal',123,family,rho,boundary=True)
        sc=db.scenario
        assert db.loc[sc=='genuine_outside','true_distance_m'].between(150.1,160).all()
        assert db.loc[sc=='genuine_boundary_inside','true_distance_m'].between(140,149.9).all()
        assert np.array_equal(db.truth_valid_presence,d.truth_valid_presence)
        generated.append(db)
    for d in generated[1:]:
        assert np.array_equal(d.observed_distance_m,generated[0].observed_distance_m)
    raw=pd.read_csv(r.OUT/'batch_metrics.csv')
    assert len(raw)==3*6*2*5*15,len(raw)
    assert not raw.duplicated(['fit','batch','config','population','protocol','method']).any()
    assert np.allclose(raw.FAR,raw.false_accept/raw.n_invalid)
    assert np.allclose(raw.FRR,raw.false_reject/raw.n_valid)
    assert np.allclose(raw.BA,1-(raw.FAR+raw.FRR)/2)
    assert (raw.n_valid+raw.n_invalid==20000).all()
    for _,block in raw.groupby(['fit','batch','population']):
        assert block.n_valid.nunique()==1
        assert block.n_invalid.nunique()==1
    diag=pd.read_csv(r.OUT/'batch_diagnostics.csv')
    for _,block in diag.groupby(['fit','batch']):
        assert block.usable_rate.nunique()==1
        assert block.quality_mean.nunique()==1
    fits=raw.groupby(['fit','config','family','rho','population','protocol','method'])[['FAR','FRR','BA','outside_accept','boundary_inside_reject','inside_impostor_accept','presentation_accept']].mean().reset_index()
    saved=pd.read_csv(r.OUT/'fit_metrics.csv')
    pd.testing.assert_frame_equal(fits,saved,check_exact=False,rtol=1e-12,atol=1e-12)
    marg=pd.read_csv(r.OUT/'primitive_marginals.csv')
    report=dict(structural_checks='passed',metric_count_checks='passed',generated_test_outcomes=3600000,underlying_paired_test_cases=600000,rows=len(raw),maximum_conditional_pit_KS=float(marg.pit_KS.max()),gate_and_quality_preservation='exact within paired batches',note='KS is a diagnostic distance, not a simultaneous hypothesis test. Fixed component distributions do not imply fixed derived liveness distribution.')
    (r.OUT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': verify()
