"""Replay only the archived nominal test draws; do not refit any model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / 'experiments'
source = (EXPERIMENTS / 'AttendPro_original_simulation_recovered.py').read_text()
namespace = {'__file__': str(EXPERIMENTS / 'AttendPro_original_simulation_recovered.py')}
exec(source.split('# Fit the fusion model on pooled')[0].replace('import nbformat as nbf', ''), namespace)
simulate = namespace['simulate_attendpro']
rows = []
for fit in range(3):
    for batch in range(5):
        seed = 20261001 + fit * 10000 + 1000 + batch
        data = simulate(20000, 'nominal', seed)
        valid = data.truth_valid_presence.to_numpy().astype(bool)
        usable = ((data.gps_quality_ok == 1) & (data.quality_pass == 1)).to_numpy()
        rows.append(dict(fit=fit, batch=batch, seed=seed, n=len(data),
                         n_valid=int(valid.sum()), n_invalid=int((~valid).sum()),
                         gate_rejected_valid=int((valid & ~usable).sum()),
                         gate_rejected_invalid=int((~valid & ~usable).sum()),
                         gate_frr=float((~usable[valid]).mean()),
                         usable_valid=float(usable[valid].mean()),
                         usable_invalid=float(usable[~valid].mean())))
gates = pd.DataFrame(rows)
archived = pd.read_csv(EXPERIMENTS / 'extension/replicate_metrics.csv')
archived = archived[(archived.condition == 'nominal') & (archived.method == 'Gradient boosting') & (archived.operating_point == 'FAR_1pct')]
for row in rows:
    prior = archived[(archived.fit == row['fit']) & (archived.rep == row['batch'])].iloc[0]
    assert int(prior.n_valid) == row['n_valid']
    assert int(prior.n_invalid) == row['n_invalid']
gates.to_csv(ROOT / 'results/reviewer_gate_diagnostics.csv', index=False)

fits = pd.read_csv(EXPERIMENTS / 'extension/fit_level_metrics.csv')
selected = fits[(fits.condition == 'nominal') & (fits.operating_point == 'FAR_1pct')]
boost = selected[selected.method == 'Gradient boosting'].set_index('fit')
rule = selected[selected.method == 'Tuned hard rule'].set_index('fit')
paired = (boost[['FAR', 'FRR']] - rule[['FAR', 'FRR']]) * 100
paired.columns = ['FAR_difference_pp', 'FRR_difference_pp']
paired.to_csv(ROOT / 'results/reviewer_paired_fit_differences.csv')

summary = {
    'basis': 'Exact replay of the existing nominal test seeds; original generator and classifiers unchanged.',
    'gate_rejection_valid_percent': float(gates.groupby('fit').gate_frr.mean().mean() * 100),
    'gate_rejection_valid_fit_range_percent': [float(gates.groupby('fit').gate_frr.mean().min() * 100), float(gates.groupby('fit').gate_frr.mean().max() * 100)],
    'usable_invalid_percent': float(gates.groupby('fit').usable_invalid.mean().mean() * 100),
    'paired_FAR_difference_pp_range': [float(paired.FAR_difference_pp.min()), float(paired.FAR_difference_pp.max())],
    'paired_FRR_difference_pp_range': [float(paired.FRR_difference_pp.min()), float(paired.FRR_difference_pp.max())],
    'paired_FAR_difference_pp_mean': float(paired.FAR_difference_pp.mean()),
    'paired_FRR_difference_pp_mean': float(paired.FRR_difference_pp.mean()),
}
(ROOT / 'results/reviewer_diagnostics_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
