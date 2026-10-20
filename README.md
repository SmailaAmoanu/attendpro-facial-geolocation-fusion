# AttendPro Paper II — code and notebooks

**Facial–Geolocation Fusion for Attendance Verification: Dependence Sensitivity
and Boundary Trade-offs**

This small package contains commented Python code, four notebooks, and five
selected result tables. Data and model files are generated when you run the code.
No saved model bundles, large attempt datasets, manuscript assets or plots are
required. Notebook outputs are cleared to keep the repository small.

## Install

Python 3.12 is recommended. From this folder:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -e ".[notebooks]"
```

For scripts without notebooks, use `python -m pip install -e .`.
Scientific dependencies are pinned in `requirements.txt`.

## Run a small complete example first

```bash
python scripts/run_example.py --out runs/my-first-run
```

This generates synthetic attempts, fits all six model families, selects and
calibrates them, tunes the threshold rule and tests frozen decisions on fresh
nominal and shifted data. It writes settings, metrics and attempt data into the
chosen folder. Small-run results illustrate the workflow; they are not the
paper's full-size results.

## Notebooks

```bash
python -m jupyterlab
```

Open and run in order:

1. **01_generate_and_understand**: truth, noisy GPS/facial evidence, gates and reference rule.
2. **02_train_calibrate_test**: independent splits, six models, calibration and fresh testing.
3. **03_boundary_and_dependence**: boundary redevelopment, cutoff-only adaptation and copula matching.
4. **04_read_paper_results**: inspect the few retained output tables and plot the principal trade-offs.

Run cells from top to bottom. Each notebook locates this repository from its
root or `notebooks/`. Generated outputs go under `runs/`. The first three use
small samples so teammates can understand and change the simulation quickly.

## Generate the full study from scratch

```bash
python scripts/reproduce.py --out runs/full-study
```

This executes the original stages in dependency order:

1. Generate the development splits, fit original models and produce extension results.
2. Use those newly generated models for the boundary comparison and reports.
3. Generate the facial-dependence study and its diagnostics.
4. Recompute the gate-rejection and paired-fit summaries.

It starts with **code only**, not bundled fits. New models, attempt samples,
full CSV outputs and per-script logs appear in the output workspace. The full
grid is much slower than the notebooks; time and memory depend on your machine.
For the original nominal/extension stage and descriptive diagnostics only:

```bash
python scripts/reproduce.py --nominal-only --out runs/nominal-study
```

Choose a new output folder for each run. The wrapper protects existing runs and
executes the historical scripts in a copied workspace. Do not run them directly
inside `archive/`, because they write outputs next to their source files.

## Logic of the entire workflow

**Assign truth first.** A valid attempt requires correct represented identity,
live presentation and true distance within 150 m. Eight scenarios include
genuine insiders/outsiders, impostors and presentation material. Each row is
one attempt, not an employee with repeated visits. Scenario prevalence and
score distributions are study assumptions, not estimates from production data.

**Generate imperfect measurements.** The simulator creates reported GPS
uncertainty, noisy 2-D coordinates and observed distance. Reported accuracy is
treated as an approximate 95% uncertainty radius: larger means less precise.
Facial similarity, observed quality, frame consistency and derived liveness
are generated directly. No real images or face-recognition networks are trained.

**Apply common gates.** GPS uncertainty must be ≤50 m, and capture-quality
checks must pass. All fusion methods reject failed gates. These attempts remain
in the error denominators, creating a common legitimate-rejection floor.

**Use observed predictors only.** Models receive distance, GPS uncertainty,
similarity, observed quality, liveness and frame consistency. They never receive
true distance, scenario labels or latent quality. Six families cover additive,
interaction and reduced logistic regression, spline-additive logistic, random
forest and histogram gradient boosting.

**Train, select, calibrate and tune.** The original study uses three development
fits, each with 60,000 training, 30,000 validation and 30,000 calibration attempts
pooled across five regimes. Models fit usable training attempts. Validation
Brier loss selects candidates; independent calibration adjusts probability
scores. Validation chooses a cutoff maximizing valid acceptance subject to
empirical FAR ≤1%. The rule searches 432 settings on the same validation data.
Validation is reused for candidate/cutoff selection; final test data stay separate.

**Freeze before testing.** Nominal tests use five fresh 20,000-attempt batches
per fit. Boundary development adds examples near the radius while preserving
facial evidence. Compare unchanged decisions, cutoff-only adaptation and full
redevelopment. Copula experiments instead change facial dependence while
preserving paired truth, location and quality. Derived liveness is recomputed.
Matching facial dependence does not supply missing location examples.

**Report aggregate and subgroup errors.** FAR is acceptance among invalid
attempts; FRR is rejection among valid attempts. Genuine-outside acceptance and
boundary-inside rejection use their own denominators. Rates in CSVs are fractions;
multiply by 100 for percentages. Average batches within fit, then average fits.
The 1% target is a validation constraint, not a deployment guarantee. Three-fit
ranges are descriptive, not comprehensive confidence intervals.

## Terms and abbreviations

| Term | Meaning |
|---|---|
| Frozen | Original model, calibration and cutoff unchanged |
| Tuned rule | Threshold conjunction selected on validation data |
| Cutoff adapted | Only final cutoff changes; model/calibration stay fixed |
| Redeveloped | Repeat fitting, selection, calibration and cutoff tuning |
| Matched | Redevelop under the facial-dependence condition being evaluated |
| FAR | False acceptance rate |
| FRR | False rejection rate |
| BA | Balanced accuracy: 1−(FAR+FRR)/2 |
| Calibration | Adjust model scores to predicted probabilities using a separate split |
| Copula | Joint dependence construction retaining specified primitive marginal distributions |
| ρ (rho) | Copula correlation parameter; not the final predictors' observed correlation |
| GAM additive | Historical CSV name for cubic-spline additive logistic regression |

Student-t with four degrees of freedom and ρ=0 is not independence. Dependence
outcomes share base cases: 3.6 million outcomes represent 600,000 paired cases.
Degraded capture is presently attached to a genuine-inside scenario, a limitation
that should be considered before generalizing quality effects. The experiments
do not establish real-world accuracy or resistance to fabricated coordinates.

## Files and retained outputs

| Path | Purpose |
|---|---|
| `src/attendpro_paper2/` | Readable, commented generator/model/intervention functions |
| `notebooks/` | Four notebooks; outputs generated by running them |
| `scripts/run_example.py` | Small end-to-end run |
| `scripts/reproduce.py` | Original full study, starting without model artifacts |
| `archive/experiments/*.py` | Original experiment algorithms retained for exact seed/order reproduction |
| `results/paper/` | Selected nominal, boundary, dependence, gate and paired-fit tables |
| `runs/` | Generated results; excluded from Git |

Selected tables retain original filenames and column labels. `fit` indexes
development runs; `rep`/`batch` index test batches. `method` names the model;
`arm` identifies boundary adaptation; `protocol` distinguishes frozen/matched
dependence fits. `genuine_outside` is outsider acceptance, while
`boundary_inside_rejection` is legitimate boundary rejection. `gate_frr` is
shared-gate rejection among valid attempts. `*_difference_pp` columns are
boosting-minus-rule differences in percentage points.

## Check and extend

```bash
python scripts/verify_package.py
python scripts/check_notebooks.py
```

The notebook checker uses an in-process IPython session and does not require
Jupyter network sockets. Add `--write` to save executed outputs. The code and
notebook cells were checked during preparation; the complete historical grid
has not been retrained as part of making this light package.

For new research, change a clearly named regime or comparator, preserve the
baseline and write results to a new folder. Keep truth out of predictors and
hold independent tests until decisions are frozen. `REGIMES` is explicit in
`simulation.py`; `pooled()` uses every current regime in insertion order.
Adding a regime therefore changes both the mixture and pooled sample size.
For a pure test shift, remove the added regime before creating development data.

Upload the contents of this folder to `attendpro-facial-geolocation-fusion`.
Your team can generate large outputs locally rather than committing them.
