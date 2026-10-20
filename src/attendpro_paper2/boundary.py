"""Location-only intervention from the original boundary experiment."""

import numpy as np


def boundary(d, seed, fraction=0.5, high_noise=False):
    """Move selected outsiders and all boundary insiders; preserve facial evidence.

    fraction=.5 reproduces development enrichment; fraction=1 creates the
    near-boundary test. No observation crosses its true authorization side.
    high_noise regenerates all GPS readings using a median accuracy of 25 m.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie between 0 and 1")
    d = d.copy()
    rng = np.random.default_rng(seed)
    dt = d.true_distance_m.to_numpy().copy()
    outside = (d.scenario == "genuine_outside").to_numpy()
    inside = (d.scenario == "genuine_boundary_inside").to_numpy()
    if fraction < 1:
        outside &= rng.random(len(d)) < fraction
    dt[outside] = 150 + rng.uniform(0.1, 10, outside.sum())
    dt[inside] = 150 - rng.uniform(0.1, 10, inside.sum())
    changed = outside | inside
    # Facial columns and truth remain untouched; only changed GPS observations
    # are regenerated, except in the explicit high-noise sensitivity condition.
    acc = d.gps_accuracy_m.to_numpy().copy()
    if high_noise:
        acc = np.clip(rng.lognormal(np.log(25), 0.55, len(d)), 1, 140)
        changed = np.ones(len(d), dtype=bool)
    sigma = acc[changed] / 2.448
    err = rng.normal(size=(changed.sum(), 2)) * sigma[:, None]
    obs = d.observed_distance_m.to_numpy().copy()
    obs[changed] = np.sqrt((dt[changed] + err[:, 0]) ** 2 + err[:, 1] ** 2)
    d["true_distance_m"] = dt
    d["observed_distance_m"] = obs
    d["gps_accuracy_m"] = acc
    d["effective_distance_m"] = obs + acc
    d["gps_quality_ok"] = (acc <= 50).astype(int)
    d["geo_pass"] = ((obs + acc <= 150) & (acc <= 50)).astype(int)
    d["M1_geofence"] = d.geo_pass
    d["M3_geo_face"] = d.geo_pass * d.face_pass
    d["M4_current_attendpro"] = d.geo_pass * d.face_pass * d.liveness_pass
    return d
