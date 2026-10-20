"""Synthetic verification measurements used by AttendPro Paper II.

The generator body is copied from the archived source without changing random
number order. A row is an attendance attempt, not a person or a captured image.
Constants and regime settings below are study assumptions/application settings.
Importing this module never fits a model or writes files.
"""

import math
import numpy as np
import pandas as pd

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
        "gps_median": 8.0,
        "gps_sigma": 0.45,
        "gps_error_mult": 1.0,
        "quality_shift": 0.00,
        "spoof_motion_mult": 1.0,
        "weights": [0.18, 0.10, 0.10, 0.15, 0.15, 0.10, 0.12, 0.10],
    },
    "gps_noisy": {
        "gps_median": 20.0,
        "gps_sigma": 0.55,
        "gps_error_mult": 1.15,
        "quality_shift": 0.00,
        "spoof_motion_mult": 1.0,
        "weights": [0.18, 0.10, 0.10, 0.15, 0.15, 0.10, 0.12, 0.10],
    },
    "face_degraded": {
        "gps_median": 8.0,
        "gps_sigma": 0.45,
        "gps_error_mult": 1.0,
        "quality_shift": -0.22,
        "spoof_motion_mult": 1.0,
        "weights": [0.18, 0.10, 0.10, 0.15, 0.15, 0.10, 0.12, 0.10],
    },
    "boundary_heavy": {
        "gps_median": 12.0,
        "gps_sigma": 0.50,
        "gps_error_mult": 1.05,
        "quality_shift": -0.05,
        "spoof_motion_mult": 1.0,
        "weights": [0.10, 0.22, 0.08, 0.12, 0.12, 0.08, 0.18, 0.10],
    },
    "spoof_heavy": {
        "gps_median": 10.0,
        "gps_sigma": 0.50,
        "gps_error_mult": 1.0,
        "quality_shift": -0.03,
        "spoof_motion_mult": 1.35,
        "weights": [0.14, 0.08, 0.08, 0.10, 0.10, 0.08, 0.30, 0.12],
    },
}

SCENARIOS = np.array(
    [
        "genuine_inside",
        "genuine_boundary_inside",
        "genuine_degraded_inside",
        "genuine_outside",
        "impostor_inside",
        "impostor_outside",
        "spoof_inside",
        "spoof_boundary",
    ]
)


def beta_from_mean(mean, concentration, rng, size):
    mean = np.clip(mean, 0.01, 0.99)
    a = mean * concentration
    b = (1 - mean) * concentration
    return rng.beta(a, b, size=size)


def simulate_attendpro(n, regime_name, seed):
    rng = np.random.default_rng(seed)
    p = REGIMES[regime_name]
    scenario = rng.choice(
        SCENARIOS, size=n, p=np.array(p["weights"]) / np.sum(p["weights"])
    )

    # Scenario truth
    genuine = np.isin(
        scenario,
        [
            "genuine_inside",
            "genuine_boundary_inside",
            "genuine_degraded_inside",
            "genuine_outside",
            "spoof_inside",
            "spoof_boundary",
        ],
    )
    live = ~np.isin(scenario, ["spoof_inside", "spoof_boundary"])

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
    angle = rng.uniform(0, 2 * np.pi, size=n)
    x_true, y_true = d_true * np.cos(angle), d_true * np.sin(angle)
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

    detector_conf = np.clip(
        0.875 + 0.12 * q + rng.normal(0, 0.018 + 0.015 * (1 - q), n), 0.70, 0.999
    )

    brightness_sd = 14 + 65 * (1 - q)
    brightness = 130 + rng.normal(0, brightness_sd, n)
    # Some degraded captures are deliberately dark/overexposed.
    deg_idx = np.where(degraded)[0]
    if len(deg_idx):
        flip = rng.random(len(deg_idx)) < 0.35
        brightness[deg_idx[flip]] -= rng.uniform(60, 100, flip.sum())
        brightness[deg_idx[~flip]] += rng.uniform(45, 95, (~flip).sum())
    brightness = np.clip(brightness, 5, 250)

    blur_mean = np.maximum(20, 42 + 170 * q)
    blur = rng.lognormal(np.log(blur_mean), 0.30 + 0.12 * (1 - q), n)

    face_area = np.clip(
        0.045 + 0.19 * q + rng.normal(0, 0.020 + 0.012 * (1 - q), n), 0.01, 0.40
    )
    nose_offset = rng.normal(0, 0.035 + 0.085 * (1 - q), n)

    quality_pass = (
        (detector_conf >= FACE_DETECTION_MIN)
        & (face_area >= FACE_AREA_MIN)
        & (brightness >= FACE_BRIGHTNESS_MIN)
        & (brightness <= FACE_BRIGHTNESS_MAX)
        & (blur >= FACE_BLUR_MIN)
        & (np.abs(nose_offset) <= FACE_POSE_MAX)
    )

    # Observable composite quality score (not latent q).
    q_conf = np.clip((detector_conf - 0.85) / 0.15, 0, 1)
    q_blur = np.clip((blur - FACE_BLUR_MIN) / 180, 0, 1)
    q_area = np.clip((face_area - 0.04) / 0.20, 0, 1)
    q_pose = np.clip(1 - np.abs(nose_offset) / FACE_POSE_MAX, 0, 1)
    # brightness quality peaks around 130 and decreases toward hard limits
    q_bright = np.where(
        brightness <= 130,
        np.clip((brightness - FACE_BRIGHTNESS_MIN) / (130 - FACE_BRIGHTNESS_MIN), 0, 1),
        np.clip((FACE_BRIGHTNESS_MAX - brightness) / (FACE_BRIGHTNESS_MAX - 130), 0, 1),
    )
    q_obs = np.mean(np.vstack([q_conf, q_blur, q_area, q_pose, q_bright]), axis=0)

    # Face similarity. Genuine/spoof use enrolled identity; impostors do not.
    is_impostor = np.isin(scenario, ["impostor_inside", "impostor_outside"])
    mean_genuine = np.clip(0.46 + 0.30 * q_obs, 0.42, 0.80)
    mean_impostor = np.clip(0.18 + 0.13 * q_obs, 0.16, 0.34)
    face_similarity = np.empty(n)
    face_similarity[~is_impostor] = beta_from_mean(
        mean_genuine[~is_impostor], 30, rng, (~is_impostor).sum()
    )
    face_similarity[is_impostor] = beta_from_mean(
        mean_impostor[is_impostor], 26, rng, is_impostor.sum()
    )
    face_similarity = np.clip(face_similarity, 0, 1)
    face_pass = quality_pass & (face_similarity >= FACE_THRESHOLD)

    # Multi-frame temporal liveness components using the same formula as AttendPro.
    frame_consistency = np.empty(n)
    live_mask = live
    spoof_mask = ~live
    frame_consistency[live_mask] = beta_from_mean(
        np.full(live_mask.sum(), 0.89), 35, rng, live_mask.sum()
    )
    frame_consistency[spoof_mask] = beta_from_mean(
        np.full(spoof_mask.sum(), 0.94), 45, rng, spoof_mask.sum()
    )

    area_span = np.empty(n)
    nose_span = np.empty(n)
    landmark_motion = np.empty(n)
    light_span = np.empty(n)

    area_span[live_mask] = rng.gamma(2.2, 0.0090, live_mask.sum())
    nose_span[live_mask] = rng.gamma(2.1, 0.0140, live_mask.sum())
    landmark_motion[live_mask] = rng.gamma(2.2, 0.0110, live_mask.sum())
    light_span[live_mask] = rng.gamma(2.0, 2.8, live_mask.sum())

    sm = p["spoof_motion_mult"]
    area_span[spoof_mask] = rng.gamma(1.7, 0.0024 * sm, spoof_mask.sum())
    nose_span[spoof_mask] = rng.gamma(1.7, 0.0030 * sm, spoof_mask.sum())
    landmark_motion[spoof_mask] = rng.gamma(1.7, 0.0025 * sm, spoof_mask.sum())
    light_span[spoof_mask] = rng.gamma(1.6, 0.90 * sm, spoof_mask.sum())

    motion_score = np.minimum(
        1.0,
        (area_span / 0.025) * 0.25
        + (nose_span / 0.035) * 0.35
        + (landmark_motion / 0.035) * 0.30
        + np.minimum(light_span / 8.0, 1.0) * 0.10,
    )
    consistency_component = np.clip((frame_consistency - 0.65) / 0.30, 0, 1)
    liveness_score = np.clip(0.60 * motion_score + 0.40 * consistency_component, 0, 1)
    passive_live_pass = (frame_consistency >= FRAME_CONSISTENCY_MIN) & (
        liveness_score >= LIVENESS_MIN
    )

    # One active fallback, approximating AttendPro's turn_head/move_closer challenge.
    needs_challenge = (
        quality_pass & (face_similarity >= FACE_THRESHOLD) & (~passive_live_pass)
    )
    challenge_pass = np.zeros(n, dtype=bool)
    challenge_pass[needs_challenge & live] = (
        rng.random((needs_challenge & live).sum()) < 0.90
    )
    challenge_pass[needs_challenge & (~live)] = (
        rng.random((needs_challenge & (~live)).sum()) < 0.12
    )
    final_liveness_score = liveness_score.copy()
    final_liveness_score[challenge_pass] = np.maximum(
        final_liveness_score[challenge_pass], LIVENESS_STRONG
    )
    live_pass = passive_live_pass | (
        challenge_pass & (frame_consistency >= FRAME_CONSISTENCY_MIN)
    )

    # Current AttendPro style hard decision.
    current_pass = geo_pass & face_pass & live_pass

    # Baselines
    m1 = geo_pass
    m2 = face_pass
    m3 = geo_pass & face_pass
    m4 = current_pass

    return pd.DataFrame(
        {
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
        }
    )


FEATURES = [
    "observed_distance_m",
    "gps_accuracy_m",
    "face_similarity",
    "face_quality_score",
    "liveness_score",
    "frame_consistency",
]


def build_model_frame(df):
    X = df[FEATURES].copy()
    # Add interactions that correspond to the research hypothesis.
    X["distance_x_accuracy"] = (df["observed_distance_m"] / RADIUS) * (
        df["gps_accuracy_m"] / GPS_ACCURACY_MAX
    )
    X["similarity_x_quality"] = df["face_similarity"] * df["face_quality_score"]
    X["liveness_x_consistency"] = df["liveness_score"] * df["frame_consistency"]
    return X
