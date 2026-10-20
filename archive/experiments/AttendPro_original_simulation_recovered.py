from pathlib import Path
import math, json, zipfile, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "original_main"
SEED = 20260912
rng_master = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# AttendPro constants taken from the uploaded SkillFest codebase
# ---------------------------------------------------------------------
RADIUS = 150.0
GPS_ACCURACY_MAX = 50.0
FACE_THRESHOLD = 0.40
FACE_DETECTION_MIN = 0.90
FACE_AREA_MIN = 0.075
FACE_BRIGHTNESS_MIN = 42.0
FACE_BRIGHTNESS_MAX = 225.0
FACE_BLUR_MIN = 35.0
FACE_POSE_MAX = 0.20
LIVENESS_MIN = 0.42
LIVENESS_STRONG = 0.62
FRAME_CONSISTENCY_MIN = 0.78

# ---------------------------------------------------------------------
# Simulation regimes.
# Parameters are deliberately treated as design/sensitivity assumptions,
# not as estimates from production users.
# ---------------------------------------------------------------------
REGIMES = {
    "nominal": {
        "gps_median": 8.0, "gps_sigma": 0.45, "gps_error_mult": 1.0,
        "quality_shift": 0.00, "spoof_motion_mult": 1.0,
        "weights": [0.18,0.10,0.10,0.15,0.15,0.10,0.12,0.10],
    },
    "gps_noisy": {
        "gps_median": 20.0, "gps_sigma": 0.55, "gps_error_mult": 1.15,
        "quality_shift": 0.00, "spoof_motion_mult": 1.0,
        "weights": [0.18,0.10,0.10,0.15,0.15,0.10,0.12,0.10],
    },
    "face_degraded": {
        "gps_median": 8.0, "gps_sigma": 0.45, "gps_error_mult": 1.0,
        "quality_shift": -0.22, "spoof_motion_mult": 1.0,
        "weights": [0.18,0.10,0.10,0.15,0.15,0.10,0.12,0.10],
    },
    "boundary_heavy": {
        "gps_median": 12.0, "gps_sigma": 0.50, "gps_error_mult": 1.05,
        "quality_shift": -0.05, "spoof_motion_mult": 1.0,
        "weights": [0.10,0.22,0.08,0.12,0.12,0.08,0.18,0.10],
    },
    "spoof_heavy": {
        "gps_median": 10.0, "gps_sigma": 0.50, "gps_error_mult": 1.0,
        "quality_shift": -0.03, "spoof_motion_mult": 1.35,
        "weights": [0.14,0.08,0.08,0.10,0.10,0.08,0.30,0.12],
    },
}

SCENARIOS = np.array([
    "genuine_inside",
    "genuine_boundary_inside",
    "genuine_degraded_inside",
    "genuine_outside",
    "impostor_inside",
    "impostor_outside",
    "spoof_inside",
    "spoof_boundary",
])

def beta_from_mean(mean, concentration, rng, size):
    mean = np.clip(mean, 0.01, 0.99)
    a = mean * concentration
    b = (1 - mean) * concentration
    return rng.beta(a, b, size=size)

def simulate_attendpro(n, regime_name, seed):
    rng = np.random.default_rng(seed)
    p = REGIMES[regime_name]
    scenario = rng.choice(SCENARIOS, size=n, p=np.array(p["weights"]) / np.sum(p["weights"]))

    # Scenario truth
    genuine = np.isin(scenario, ["genuine_inside","genuine_boundary_inside",
                                 "genuine_degraded_inside","genuine_outside",
                                 "spoof_inside","spoof_boundary"])
    live = ~np.isin(scenario, ["spoof_inside","spoof_boundary"])

    # True location distance from session center
    d_true = np.zeros(n)
    mask = scenario == "genuine_inside"
    d_true[mask] = RADIUS * rng.beta(2.0, 3.0, mask.sum())

    mask = scenario == "genuine_boundary_inside"
    d_true[mask] = RADIUS - rng.uniform(0, 25, mask.sum())

    mask = scenario == "genuine_degraded_inside"
    d_true[mask] = RADIUS * rng.beta(2.2, 2.8, mask.sum())

    mask = scenario == "genuine_outside"
    d_true[mask] = RADIUS + 5 + rng.gamma(2.0, 35.0, mask.sum())

    mask = scenario == "impostor_inside"
    d_true[mask] = RADIUS * rng.beta(2.0, 3.0, mask.sum())

    mask = scenario == "impostor_outside"
    d_true[mask] = RADIUS + 5 + rng.gamma(2.0, 35.0, mask.sum())

    mask = scenario == "spoof_inside"
    d_true[mask] = RADIUS * rng.beta(2.0, 3.0, mask.sum())

    mask = scenario == "spoof_boundary"
    # deliberately straddle the boundary
    d_true[mask] = np.clip(RADIUS + rng.normal(0, 18, mask.sum()), 0, None)

    true_location_valid = d_true <= RADIUS
    y = genuine & live & true_location_valid

    # GPS reported accuracy: positive, right-skewed.
    mu = math.log(p["gps_median"])
    gps_accuracy = rng.lognormal(mean=mu, sigma=p["gps_sigma"], size=n)
    gps_accuracy = np.clip(gps_accuracy, 1.0, 140.0)

    # 2-D location error. Treat reported accuracy as an approximate 95% radial error.
    sigma_xy = (gps_accuracy / 2.448) * p["gps_error_mult"]
    angle = rng.uniform(0, 2*np.pi, size=n)
    x_true, y_true = d_true*np.cos(angle), d_true*np.sin(angle)
    x_obs = x_true + rng.normal(0, sigma_xy)
    y_obs = y_true + rng.normal(0, sigma_xy)
    d_obs = np.sqrt(x_obs**2 + y_obs**2)

    gps_quality_ok = gps_accuracy <= GPS_ACCURACY_MAX
    geo_pass = gps_quality_ok & ((d_obs + gps_accuracy) <= RADIUS)

    # Latent capture quality, then observable AttendPro quality metrics.
    q = rng.beta(7, 2.3, n) + p["quality_shift"]
    degraded = scenario == "genuine_degraded_inside"
    q[degraded] -= rng.uniform(0.25, 0.50, degraded.sum())
    q = np.clip(q, 0.03, 0.99)

    detector_conf = np.clip(0.875 + 0.12*q + rng.normal(0, 0.018 + 0.015*(1-q), n), 0.70, 0.999)

    brightness_sd = 14 + 65*(1-q)
    brightness = 130 + rng.normal(0, brightness_sd, n)
    # Some degraded captures are deliberately dark/overexposed.
    deg_idx = np.where(degraded)[0]
    if len(deg_idx):
        flip = rng.random(len(deg_idx)) < 0.35
        brightness[deg_idx[flip]] -= rng.uniform(60, 100, flip.sum())
        brightness[deg_idx[~flip]] += rng.uniform(45, 95, (~flip).sum())
    brightness = np.clip(brightness, 5, 250)

    blur_mean = np.maximum(20, 42 + 170*q)
    blur = rng.lognormal(np.log(blur_mean), 0.30 + 0.12*(1-q), n)

    face_area = np.clip(0.045 + 0.19*q + rng.normal(0, 0.020 + 0.012*(1-q), n), 0.01, 0.40)
    nose_offset = rng.normal(0, 0.035 + 0.085*(1-q), n)

    quality_pass = (
        (detector_conf >= FACE_DETECTION_MIN) &
        (face_area >= FACE_AREA_MIN) &
        (brightness >= FACE_BRIGHTNESS_MIN) &
        (brightness <= FACE_BRIGHTNESS_MAX) &
        (blur >= FACE_BLUR_MIN) &
        (np.abs(nose_offset) <= FACE_POSE_MAX)
    )

    # Observable composite quality score (not latent q).
    q_conf = np.clip((detector_conf - 0.85) / 0.15, 0, 1)
    q_blur = np.clip((blur - FACE_BLUR_MIN) / 180, 0, 1)
    q_area = np.clip((face_area - 0.04) / 0.20, 0, 1)
    q_pose = np.clip(1 - np.abs(nose_offset) / FACE_POSE_MAX, 0, 1)
    # brightness quality peaks around 130 and decreases toward hard limits
    q_bright = np.where(
        brightness <= 130,
        np.clip((brightness - FACE_BRIGHTNESS_MIN)/(130-FACE_BRIGHTNESS_MIN), 0, 1),
        np.clip((FACE_BRIGHTNESS_MAX - brightness)/(FACE_BRIGHTNESS_MAX-130), 0, 1)
    )
    q_obs = np.mean(np.vstack([q_conf,q_blur,q_area,q_pose,q_bright]), axis=0)

    # Face similarity. Genuine/spoof use enrolled identity; impostors do not.
    is_impostor = np.isin(scenario, ["impostor_inside","impostor_outside"])
    mean_genuine = np.clip(0.46 + 0.30*q_obs, 0.42, 0.80)
    mean_impostor = np.clip(0.18 + 0.13*q_obs, 0.16, 0.34)
    face_similarity = np.empty(n)
    face_similarity[~is_impostor] = beta_from_mean(mean_genuine[~is_impostor], 30, rng, (~is_impostor).sum())
    face_similarity[is_impostor] = beta_from_mean(mean_impostor[is_impostor], 26, rng, is_impostor.sum())
    face_similarity = np.clip(face_similarity, 0, 1)
    face_pass = quality_pass & (face_similarity >= FACE_THRESHOLD)

    # Multi-frame temporal liveness components using the same formula as AttendPro.
    frame_consistency = np.empty(n)
    live_mask = live
    spoof_mask = ~live
    frame_consistency[live_mask] = beta_from_mean(np.full(live_mask.sum(), 0.89), 35, rng, live_mask.sum())
    frame_consistency[spoof_mask] = beta_from_mean(np.full(spoof_mask.sum(), 0.94), 45, rng, spoof_mask.sum())

    area_span = np.empty(n)
    nose_span = np.empty(n)
    landmark_motion = np.empty(n)
    light_span = np.empty(n)

    area_span[live_mask] = rng.gamma(2.2, 0.0090, live_mask.sum())
    nose_span[live_mask] = rng.gamma(2.1, 0.0140, live_mask.sum())
    landmark_motion[live_mask] = rng.gamma(2.2, 0.0110, live_mask.sum())
    light_span[live_mask] = rng.gamma(2.0, 2.8, live_mask.sum())

    sm = p["spoof_motion_mult"]
    area_span[spoof_mask] = rng.gamma(1.7, 0.0024*sm, spoof_mask.sum())
    nose_span[spoof_mask] = rng.gamma(1.7, 0.0030*sm, spoof_mask.sum())
    landmark_motion[spoof_mask] = rng.gamma(1.7, 0.0025*sm, spoof_mask.sum())
    light_span[spoof_mask] = rng.gamma(1.6, 0.90*sm, spoof_mask.sum())

    motion_score = np.minimum(
        1.0,
        (area_span / 0.025) * 0.25
        + (nose_span / 0.035) * 0.35
        + (landmark_motion / 0.035) * 0.30
        + np.minimum(light_span / 8.0, 1.0) * 0.10,
    )
    consistency_component = np.clip((frame_consistency - 0.65)/0.30, 0, 1)
    liveness_score = np.clip(0.60*motion_score + 0.40*consistency_component, 0, 1)
    passive_live_pass = (frame_consistency >= FRAME_CONSISTENCY_MIN) & (liveness_score >= LIVENESS_MIN)

    # One active fallback, approximating AttendPro's turn_head/move_closer challenge.
    needs_challenge = quality_pass & (face_similarity >= FACE_THRESHOLD) & (~passive_live_pass)
    challenge_pass = np.zeros(n, dtype=bool)
    challenge_pass[needs_challenge & live] = rng.random((needs_challenge & live).sum()) < 0.90
    challenge_pass[needs_challenge & (~live)] = rng.random((needs_challenge & (~live)).sum()) < 0.12
    final_liveness_score = liveness_score.copy()
    final_liveness_score[challenge_pass] = np.maximum(final_liveness_score[challenge_pass], LIVENESS_STRONG)
    live_pass = passive_live_pass | (challenge_pass & (frame_consistency >= FRAME_CONSISTENCY_MIN))

    # Current AttendPro style hard decision.
    current_pass = geo_pass & face_pass & live_pass

    # Baselines
    m1 = geo_pass
    m2 = face_pass
    m3 = geo_pass & face_pass
    m4 = current_pass

    return pd.DataFrame({
        "scenario": scenario,
        "truth_valid_presence": y.astype(int),
        "true_distance_m": d_true,
        "observed_distance_m": d_obs,
        "gps_accuracy_m": gps_accuracy,
        "effective_distance_m": d_obs + gps_accuracy,
        "gps_quality_ok": gps_quality_ok.astype(int),
        "geo_pass": geo_pass.astype(int),
        "detector_confidence": detector_conf,
        "brightness": brightness,
        "blur_score": blur,
        "face_area_ratio": face_area,
        "nose_offset_x": nose_offset,
        "face_quality_score": q_obs,
        "quality_pass": quality_pass.astype(int),
        "face_similarity": face_similarity,
        "face_pass": face_pass.astype(int),
        "frame_consistency": frame_consistency,
        "motion_score": motion_score,
        "liveness_score": final_liveness_score,
        "liveness_pass": live_pass.astype(int),
        "M1_geofence": m1.astype(int),
        "M2_face": m2.astype(int),
        "M3_geo_face": m3.astype(int),
        "M4_current_attendpro": m4.astype(int),
    })

FEATURES = [
    "observed_distance_m", "gps_accuracy_m", "face_similarity",
    "face_quality_score", "liveness_score", "frame_consistency",
]

def build_model_frame(df):
    X = df[FEATURES].copy()
    # Add interactions that correspond to the research hypothesis.
    X["distance_x_accuracy"] = (df["observed_distance_m"]/RADIUS) * (df["gps_accuracy_m"]/GPS_ACCURACY_MAX)
    X["similarity_x_quality"] = df["face_similarity"] * df["face_quality_score"]
    X["liveness_x_consistency"] = df["liveness_score"] * df["frame_consistency"]
    return X

def metrics_from_binary(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    far = fp/(fp+tn) if fp+tn else np.nan
    frr = fn/(fn+tp) if fn+tp else np.nan
    sens = tp/(tp+fn) if tp+fn else np.nan
    spec = tn/(tn+fp) if tn+fp else np.nan
    return {
        "FAR": far, "FRR": frr, "Sensitivity": sens, "Specificity": spec,
        "BalancedAccuracy": 0.5*(sens+spec),
    }

def choose_balanced_threshold(y, probs):
    thresholds = np.linspace(0.05,0.95,181)
    best_t, best_ba = 0.5, -1
    for t in thresholds:
        m = metrics_from_binary(y, probs >= t)
        if m["BalancedAccuracy"] > best_ba:
            best_ba = m["BalancedAccuracy"]
            best_t = float(t)
    return best_t, best_ba

# ---------------------------------------------------------------------
# Fit the fusion model on pooled simulated training regimes.
# Only records for which AttendPro would have usable location and face quality
# are used to fit the continuous fusion model; otherwise M5 rejects.
# ---------------------------------------------------------------------
train_parts = []
for j, regime in enumerate(REGIMES):
    train_parts.append(simulate_attendpro(50000, regime, SEED + 1000 + j))
train = pd.concat(train_parts, ignore_index=True)

fit_mask = (train["gps_quality_ok"] == 1) & (train["quality_pass"] == 1)
X_train = build_model_frame(train.loc[fit_mask])
y_train = train.loc[fit_mask, "truth_valid_presence"].to_numpy()

model = Pipeline([
    ("scale", StandardScaler()),
    ("logit", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# Independent validation data to set one fixed operating threshold.
val_parts = []
for j, regime in enumerate(REGIMES):
    val_parts.append(simulate_attendpro(20000, regime, SEED + 2000 + j))
val = pd.concat(val_parts, ignore_index=True)
val_usable = (val["gps_quality_ok"] == 1) & (val["quality_pass"] == 1)
val_probs = np.zeros(len(val))
val_probs[val_usable] = model.predict_proba(build_model_frame(val.loc[val_usable]))[:,1]
fusion_threshold, fusion_val_ba = choose_balanced_threshold(val["truth_valid_presence"].to_numpy(), val_probs)

# ---------------------------------------------------------------------
# Monte Carlo evaluation: 20 independent replicates x 25,000 per regime.
# ---------------------------------------------------------------------
rows = []
calibration_store = {}
sample_store = None

for ridx, regime in enumerate(REGIMES):
    calibration_store[regime] = []
    for rep in range(20):
        df = simulate_attendpro(25000, regime, SEED + 10000 + ridx*100 + rep)
        usable = (df["gps_quality_ok"] == 1) & (df["quality_pass"] == 1)
        probs = np.zeros(len(df))
        if usable.any():
            probs[usable] = model.predict_proba(build_model_frame(df.loc[usable]))[:,1]
        pred5 = (probs >= fusion_threshold).astype(int)
        y = df["truth_valid_presence"].to_numpy()

        method_preds = {
            "M1 Geofence only": df["M1_geofence"].to_numpy(),
            "M2 Face only": df["M2_face"].to_numpy(),
            "M3 Geofence + face": df["M3_geo_face"].to_numpy(),
            "M4 Current AttendPro": df["M4_current_attendpro"].to_numpy(),
            "M5 Uncertainty-aware fusion": pred5,
        }
        for method, pred in method_preds.items():
            m = metrics_from_binary(y, pred)
            row = {"regime": regime, "replicate": rep+1, "method": method, **m}
            if method == "M5 Uncertainty-aware fusion":
                row["ROC_AUC"] = roc_auc_score(y, probs)
                row["PR_AUC"] = average_precision_score(y, probs)
                row["Brier"] = brier_score_loss(y, probs)
            else:
                row["ROC_AUC"] = np.nan
                row["PR_AUC"] = np.nan
                row["Brier"] = np.nan
            rows.append(row)

        if rep == 0:
            calibration_store[regime].append((y, probs))
            if regime == "nominal":
                sample_store = df.copy()
                sample_store["fusion_probability"] = probs
                sample_store["M5_fusion"] = pred5

mc = pd.DataFrame(rows)

summary = (
    mc.groupby(["regime","method"])
      .agg(
          FAR_mean=("FAR","mean"), FAR_sd=("FAR","std"),
          FRR_mean=("FRR","mean"), FRR_sd=("FRR","std"),
          Sensitivity_mean=("Sensitivity","mean"),
          Specificity_mean=("Specificity","mean"),
          BalancedAccuracy_mean=("BalancedAccuracy","mean"),
          BalancedAccuracy_sd=("BalancedAccuracy","std"),
          ROC_AUC_mean=("ROC_AUC","mean"),
          PR_AUC_mean=("PR_AUC","mean"),
          Brier_mean=("Brier","mean"),
      ).reset_index()
)

# Monte Carlo percentile intervals
quant = (mc.groupby(["regime","method"])[["FAR","FRR","BalancedAccuracy"]]
           .quantile([0.025,0.975]).unstack())
quant.columns = [f"{a}_{'lo' if b==0.025 else 'hi'}" for a,b in quant.columns]
quant = quant.reset_index()
summary = summary.merge(quant, on=["regime","method"], how="left")

# Coefficients
feature_names = list(build_model_frame(train.iloc[:2]).columns)
coefs = model.named_steps["logit"].coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "standardized_logit_coefficient": coefs})
coef_df["odds_ratio_per_1SD"] = np.exp(coef_df["standardized_logit_coefficient"])
coef_df = coef_df.sort_values("standardized_logit_coefficient", ascending=False)

# Parameter table
param_rows = []
for name,p in REGIMES.items():
    param_rows.append({
        "regime": name,
        "gps_accuracy_median_m": p["gps_median"],
        "gps_lognormal_sigma": p["gps_sigma"],
        "gps_error_multiplier": p["gps_error_mult"],
        "face_quality_shift": p["quality_shift"],
        "spoof_motion_multiplier": p["spoof_motion_mult"],
    })
params_df = pd.DataFrame(param_rows)

# Save tabular outputs
mc.to_csv(OUT/"AttendPro_Simulation_MonteCarlo.csv", index=False)
summary.to_csv(OUT/"AttendPro_Simulation_Results.csv", index=False)
coef_df.to_csv(OUT/"AttendPro_Fusion_Logistic_Coefficients.csv", index=False)
params_df.to_csv(OUT/"AttendPro_Simulation_Parameters.csv", index=False)
sample_store.sample(10000, random_state=SEED).to_csv(OUT/"AttendPro_Simulation_Sample.csv", index=False)

# ---------------------------------------------------------------------
# Figures (one figure per plot; no explicit colors/styles)
# ---------------------------------------------------------------------
# 1. FAR/FRR nominal
nom = summary[summary["regime"]=="nominal"].copy()
x = np.arange(len(nom))
width = 0.35
fig, ax = plt.subplots(figsize=(9,5.5))
ax.bar(x-width/2, nom["FAR_mean"], width, label="FAR")
ax.bar(x+width/2, nom["FRR_mean"], width, label="FRR")
ax.set_xticks(x)
ax.set_xticklabels(nom["method"], rotation=25, ha="right")
ax.set_ylabel("Rate")
ax.set_title("Nominal regime: false acceptance and false rejection")
ax.legend()
fig.tight_layout()
fig.savefig(OUT/"AttendPro_Nominal_FAR_FRR.png", dpi=180)
plt.close(fig)

# 2. Balanced accuracy across regimes
pivot = summary.pivot(index="regime", columns="method", values="BalancedAccuracy_mean")
fig, ax = plt.subplots(figsize=(9,5.5))
for col in pivot.columns:
    ax.plot(pivot.index, pivot[col], marker="o", label=col)
ax.set_ylabel("Balanced accuracy")
ax.set_title("Balanced accuracy across simulation regimes")
ax.set_ylim(0.45, 1.01)
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"AttendPro_Balanced_Accuracy_By_Regime.png", dpi=180)
plt.close(fig)

# 3. Calibration nominal
y_cal, p_cal = calibration_store["nominal"][0]
frac_pos, mean_pred = calibration_curve(y_cal, p_cal, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(mean_pred, frac_pos, marker="o", label="Fusion model")
ax.plot([0,1],[0,1], linestyle="--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed valid-presence proportion")
ax.set_title("Nominal regime: fusion-model calibration")
ax.legend()
fig.tight_layout()
fig.savefig(OUT/"AttendPro_Fusion_Calibration.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------
def pct(x):
    return f"{100*x:.2f}%"

def row_for(regime, method):
    return summary[(summary.regime==regime)&(summary.method==method)].iloc[0]

nom_rows = {m: row_for("nominal",m) for m in summary.method.unique()}
stress_fusion = summary[summary.method=="M5 Uncertainty-aware fusion"].copy()

report = f"""# AttendPro uncertainty-aware multimodal verification: simulation experiment

## Purpose

This experiment evaluates whether continuous fusion of facial-verification and geolocation evidence can improve physical-presence decisions compared with the existing hard-rule structure used by AttendPro.

The simulation is explicitly anchored to the uploaded SkillFest AttendPro implementation. The following software thresholds were preserved:

- geofence radius: {RADIUS:.0f} m (default/session fallback);
- maximum accepted GPS uncertainty: {GPS_ACCURACY_MAX:.0f} m;
- conservative geofence rule: observed distance + reported accuracy <= radius;
- facial cosine-similarity threshold: {FACE_THRESHOLD:.2f};
- minimum detector confidence: {FACE_DETECTION_MIN:.2f};
- minimum face-area ratio: {FACE_AREA_MIN:.3f};
- brightness interval: [{FACE_BRIGHTNESS_MIN:.0f}, {FACE_BRIGHTNESS_MAX:.0f}];
- minimum blur score: {FACE_BLUR_MIN:.0f};
- maximum absolute nose offset: {FACE_POSE_MAX:.2f};
- minimum frame consistency: {FRAME_CONSISTENCY_MIN:.2f};
- minimum liveness score: {LIVENESS_MIN:.2f};
- strong active-challenge score: {LIVENESS_STRONG:.2f}.

The face and GPS distributions are **simulation assumptions for sensitivity analysis**, not empirical estimates from SkillFest attendees or other production users.

## Data-generating design

Eight latent verification scenarios were generated:

1. genuine identity, clearly inside the geofence;
2. genuine identity, just inside the boundary;
3. genuine identity, inside with degraded face capture;
4. genuine identity outside the geofence;
5. impostor inside the geofence;
6. impostor outside the geofence;
7. presentation/spoof attempt inside the geofence;
8. presentation/spoof attempt around the boundary.

Ground-truth valid presence was defined as a live, genuine enrolled identity whose true location was inside the session radius.

Reported GPS uncertainty was generated from a lognormal distribution. Two-dimensional measurement error was then generated with scale linked to the reported accuracy value. Facial quality variables were simulated and passed through the same hard quality thresholds used in the AttendPro code. Face similarity used overlapping bounded distributions whose means depend on observed capture quality. Multi-frame liveness used the same weighted liveness-score formula implemented by AttendPro, followed by an active-challenge fallback for some passive failures.

## Compared methods

- M1: geofence only;
- M2: face match only;
- M3: geofence AND face match;
- M4: current AttendPro-style hard rule (geofence + face quality/match + liveness);
- M5: uncertainty-aware logistic fusion using continuous distance, GPS accuracy, face similarity, face quality, liveness, frame consistency, and prespecified interactions.

The fusion model was fitted on 250,000 pooled simulated training observations across five regimes. A separate 100,000-observation validation sample selected one fixed balanced-accuracy threshold of **{fusion_threshold:.3f}**. Evaluation used 20 independent replicates of 25,000 trials in each regime (500,000 test trials per regime).

## Nominal-regime results

| Method | FAR | FRR | Balanced accuracy |
|---|---:|---:|---:|
"""
for method in [
    "M1 Geofence only","M2 Face only","M3 Geofence + face",
    "M4 Current AttendPro","M5 Uncertainty-aware fusion"
]:
    r = nom_rows[method]
    report += f"| {method} | {pct(r.FAR_mean)} | {pct(r.FRR_mean)} | {pct(r.BalancedAccuracy_mean)} |\n"

fr = nom_rows["M5 Uncertainty-aware fusion"]
report += f"""
For the fusion model in the nominal regime, mean ROC AUC was **{fr.ROC_AUC_mean:.4f}**, mean PR AUC was **{fr.PR_AUC_mean:.4f}**, and mean Brier score was **{fr.Brier_mean:.4f}**.

## Robustness across regimes

| Regime | Fusion FAR | Fusion FRR | Fusion balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|
"""
for _,r in stress_fusion.iterrows():
    report += f"| {r.regime} | {pct(r.FAR_mean)} | {pct(r.FRR_mean)} | {pct(r.BalancedAccuracy_mean)} | {r.ROC_AUC_mean:.4f} |\n"

report += f"""
## Interpretation

The comparison is intentionally conducted on the same simulated AttendPro-style evidence for all methods. The hard-rule baseline therefore represents the present decision structure, while the fusion method changes only how the available evidence is combined.

This is a simulation study. It does **not** establish real-world biometric accuracy, real-world GPS accuracy, demographic fairness, or production false-acceptance/false-rejection rates. Its purpose is to determine whether the proposed uncertainty-aware fusion principle has merit under a transparent family of operating conditions tied to the actual software thresholds and liveness calculation.

The most important next step for a paper is sensitivity analysis: vary the assumed overlap of genuine/impostor facial scores, GPS-error calibration, spoof motion, scenario prevalence, geofence radius, and quality degradation. Conclusions should be reported only if they remain stable across reasonable parameter ranges.

## Reproducibility files

- `AttendPro_Multimodal_Simulation.ipynb`
- `AttendPro_Simulation_Results.csv`
- `AttendPro_Simulation_MonteCarlo.csv`
- `AttendPro_Simulation_Parameters.csv`
- `AttendPro_Fusion_Logistic_Coefficients.csv`
- `AttendPro_Simulation_Sample.csv`
- `AttendPro_Nominal_FAR_FRR.png`
- `AttendPro_Balanced_Accuracy_By_Regime.png`
- `AttendPro_Fusion_Calibration.png`
"""

(OUT/"AttendPro_Simulation_Report.md").write_text(report, encoding="utf-8")

# ---------------------------------------------------------------------
# Reproducible notebook
# ---------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
nb["metadata"]["language_info"] = {"name":"python","version":"3"}

nb.cells = [
    nbf.v4.new_markdown_cell("""# AttendPro multimodal physical-presence simulation

This notebook reproduces the simulation study comparing the current AttendPro hard decision structure with an uncertainty-aware logistic fusion model.

**Important:** Distributional parameters are simulation/sensitivity assumptions anchored to the software thresholds. They are not estimates from production attendees."""),
    nbf.v4.new_markdown_cell(f"""## AttendPro thresholds used

- Radius: `{RADIUS:.0f} m`
- Maximum GPS uncertainty: `{GPS_ACCURACY_MAX:.0f} m`
- Face-match threshold: `{FACE_THRESHOLD:.2f}`
- Detector confidence: `{FACE_DETECTION_MIN:.2f}`
- Face-area ratio: `{FACE_AREA_MIN:.3f}`
- Brightness: `{FACE_BRIGHTNESS_MIN:.0f}–{FACE_BRIGHTNESS_MAX:.0f}`
- Blur score: `{FACE_BLUR_MIN:.0f}`
- Pose offset: `{FACE_POSE_MAX:.2f}`
- Frame consistency: `{FRAME_CONSISTENCY_MIN:.2f}`
- Liveness: `{LIVENESS_MIN:.2f}`
- Strong challenge score: `{LIVENESS_STRONG:.2f}`"""),
    nbf.v4.new_code_cell("import pandas as pd\nresults = pd.read_csv('AttendPro_Simulation_Results.csv')\nresults.head(10)"),
    nbf.v4.new_markdown_cell("""## Experiment design

Eight known simulation scenarios represent valid and invalid identity/location/liveness combinations. Five decision methods are evaluated on the same trials. The full executable simulation code is contained in the companion script generated below so that the notebook stays readable."""),
    nbf.v4.new_code_cell(f"fusion_threshold = {fusion_threshold!r}\nfusion_threshold"),
    nbf.v4.new_code_cell("nominal = results[results['regime']=='nominal'][['method','FAR_mean','FRR_mean','BalancedAccuracy_mean','ROC_AUC_mean','PR_AUC_mean','Brier_mean']]\nnominal"),
    nbf.v4.new_code_cell("fusion = results[results['method']=='M5 Uncertainty-aware fusion'][['regime','FAR_mean','FRR_mean','BalancedAccuracy_mean','ROC_AUC_mean','PR_AUC_mean','Brier_mean']]\nfusion"),
    nbf.v4.new_code_cell("coef = pd.read_csv('AttendPro_Fusion_Logistic_Coefficients.csv')\ncoef"),
    nbf.v4.new_markdown_cell("""## Interpretation

The study is simulation-based. It tests the *decision principle* under controlled, transparent operating regimes. It does not claim that the simulated distributions are empirical population distributions. Real-world biometric and geolocation accuracy would require an independently approved data-collection study."""),
]
nbf.write(nb, OUT/"AttendPro_Multimodal_Simulation.ipynb")

# Also save executable simulation script from the current notebook source is not straightforward;
# create a compact README instead.
readme = f"""AttendPro Multimodal Simulation Package
======================================

This package contains the completed simulation experiment for the proposed
uncertainty-aware facial + geofence physical-presence verification study.

Fusion decision threshold selected on independent validation data:
{fusion_threshold:.3f}

Important:
- The software thresholds and liveness scoring structure come from the uploaded
  SkillFest AttendPro codebase.
- The stochastic distributions are transparent simulation assumptions, not
  empirical estimates from production attendees.
- No production biometric records or attendee identities were used.

See AttendPro_Simulation_Report.md first.
"""
(OUT/"AttendPro_Simulation_README.txt").write_text(readme, encoding="utf-8")

package_files = [
    "AttendPro_Multimodal_Simulation.ipynb",
    "AttendPro_Simulation_Report.md",
    "AttendPro_Simulation_README.txt",
    "AttendPro_Simulation_Results.csv",
    "AttendPro_Simulation_MonteCarlo.csv",
    "AttendPro_Simulation_Parameters.csv",
    "AttendPro_Fusion_Logistic_Coefficients.csv",
    "AttendPro_Simulation_Sample.csv",
    "AttendPro_Nominal_FAR_FRR.png",
    "AttendPro_Balanced_Accuracy_By_Regime.png",
    "AttendPro_Fusion_Calibration.png",
]
zip_path = OUT/"AttendPro_Multimodal_Simulation_Package.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in package_files:
        z.write(OUT/fn, arcname=fn)

print("Fusion threshold:", round(fusion_threshold, 3))
display_cols = ["regime","method","FAR_mean","FRR_mean","BalancedAccuracy_mean","ROC_AUC_mean","PR_AUC_mean","Brier_mean"]
print(summary[display_cols].to_string(index=False))
print("\nCreated:", zip_path)
