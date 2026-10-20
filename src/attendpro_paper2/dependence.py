"""Gaussian and Student-t facial-primitive sensitivity experiment."""

import numpy as np
from scipy import stats
from .simulation import simulate_attendpro as sim, REGIMES as regimes

CONFIGS = [
    ("Independent", "independent", 0.0),
    ("Gaussian 0.3", "gaussian", 0.3),
    ("Gaussian 0.6", "gaussian", 0.6),
    ("t4 0.0", "t", 0.0),
    ("t4 0.3", "t", 0.3),
    ("t4 0.6", "t", 0.6),
]
PRIMITIVES = [
    "similarity",
    "consistency",
    "area_span",
    "nose_span",
    "landmark_motion",
    "light_span",
]


def generate(n, regime, seed, family, rho, boundary=False, diagnostics=False):
    """Generate one paired facial-dependence condition.

    With equal seed, sample size and regime, quality/location/truth remain
    identical across copulas. The six primitives change jointly; liveness is
    recomputed and need not retain its marginal distribution. diagnostics=True
    returns (attempts, copula_uniforms, primitive_measurements).
    """
    if family not in {"independent", "gaussian", "t"}:
        raise ValueError("family must be 'independent', 'gaussian', or 't'")
    if family == "independent" and rho != 0:
        raise ValueError("Primitive independence requires rho=0")
    # Start from the same scenario, GPS and quality draws for every copula.
    d = sim(n, regime, seed)
    rng = np.random.default_rng(seed + 4000000)
    normals = rng.standard_normal((n, 6))
    omega = np.full((6, 6), rho)
    np.fill_diagonal(omega, 1.0)
    z = normals @ np.linalg.cholesky(omega).T
    # A shared chi-square scale creates t-tail dependence even when rho=0.
    # Keep this draw for every family to preserve the original RNG schedule.
    scale = rng.chisquare(4, n)
    u = (
        stats.t.cdf(z / np.sqrt(scale[:, None] / 4), 4)
        if family == "t"
        else stats.norm.cdf(z)
    )
    u = np.clip(u, 1e-12, 1 - 1e-12)
    sc = d.scenario.to_numpy()
    imp = np.isin(sc, ["impostor_inside", "impostor_outside"])
    live = ~np.isin(sc, ["spoof_inside", "spoof_boundary"])
    qo = d.face_quality_score.to_numpy()
    mu = np.where(
        imp,
        np.clip(0.18 + 0.13 * qo, 0.16, 0.34),
        np.clip(0.46 + 0.30 * qo, 0.42, 0.80),
    )
    concentration = np.where(imp, 26, 30)
    # Inverse CDFs keep the specified conditional primitive distributions.
    # s is similarity; c is consistency; ell below is final liveness evidence.
    s = stats.beta.ppf(u[:, 0], mu * concentration, (1 - mu) * concentration)
    cmu = np.where(live, 0.89, 0.94)
    cc = np.where(live, 35, 45)
    c = stats.beta.ppf(u[:, 1], cmu * cc, (1 - cmu) * cc)
    sm = regimes[regime]["spoof_motion_mult"]
    spans = []
    # Each tuple is (live shape, live scale, presentation shape, presentation scale).
    # The four components are area, nose, landmark and lighting motion spans.
    for j, (al, sl, ap, sp) in enumerate(
        [
            (2.2, 0.009, 1.7, 0.0024),
            (2.1, 0.014, 1.7, 0.003),
            (2.2, 0.011, 1.7, 0.0025),
            (2.0, 2.8, 1.6, 0.90),
        ]
    ):
        spans.append(
            stats.gamma.ppf(
                u[:, j + 2], np.where(live, al, ap), scale=np.where(live, sl, sp * sm)
            )
        )
    a, no, land, light = spans
    motion = np.minimum(
        1,
        0.25 * a / 0.025
        + 0.35 * no / 0.035
        + 0.30 * land / 0.035
        + 0.10 * np.minimum(light / 8, 1),
    )
    ell = np.clip(0.6 * motion + 0.4 * np.clip((c - 0.65) / 0.30, 0, 1), 0, 1)
    passive = (c >= 0.78) & (ell >= 0.42)
    face = d.quality_pass.to_numpy().astype(bool) & (s >= 0.40)
    eligible = face & (~passive)
    # This simulates challenge outcomes; it does not process a real video.
    challenge = eligible & (rng.random(n) < np.where(live, 0.90, 0.12))
    ell[challenge] = np.maximum(ell[challenge], 0.62)
    lp = passive | (challenge & (c >= 0.78))
    d["face_similarity"] = s
    d["frame_consistency"] = c
    d["motion_score"] = motion
    d["liveness_score"] = ell
    d["face_pass"] = face.astype(int)
    d["liveness_pass"] = lp.astype(int)
    if boundary:
        # Preserve authorization sides and use the same location stream across copulas.
        rb = np.random.default_rng(seed + 900000)
        dt = d.true_distance_m.to_numpy().copy()
        m = sc == "genuine_outside"
        dt[m] = 150 + rb.uniform(0.1, 10, m.sum())
        m = sc == "genuine_boundary_inside"
        dt[m] = 150 - rb.uniform(0.1, 10, m.sum())
        err = rb.normal(size=(n, 2)) * d.gps_accuracy_m.to_numpy()[:, None] / 2.448
        d["true_distance_m"] = dt
        d["observed_distance_m"] = np.sqrt((dt + err[:, 0]) ** 2 + err[:, 1] ** 2)
        d["effective_distance_m"] = d.observed_distance_m + d.gps_accuracy_m
        d["geo_pass"] = (
            (d.effective_distance_m <= 150) & (d.gps_quality_ok == 1)
        ).astype(int)
    d["M2_face"] = face.astype(int)
    d["M3_geo_face"] = d.geo_pass * face
    d["M4_current_attendpro"] = d.geo_pass * face * lp
    if diagnostics:
        return d, u, np.column_stack([s, c, *spans])
    return d
