"""Fit, calibrate and evaluate the paper's six fusion specifications.

Truth is used only as a target or for evaluation. Scenario, true distance and
latent capture quality never enter the feature matrix. Callers must generate
training, validation, calibration and test data using separate seeds.
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .simulation import FEATURES, REGIMES, build_model_frame, simulate_attendpro

METHODS = [
    "Logistic interactions",
    "Logistic additive",
    "Logistic no reliability",
    "GAM additive",
    "Random forest",
    "Gradient boosting",
]


def usable(data):
    """Both quality gates must pass, regardless of the classifier's score."""
    return ((data.gps_quality_ok == 1) & (data.quality_pass == 1)).to_numpy()


def features(data, kind="base"):
    """Select observed evidence only; fit scalers inside training pipelines."""
    if kind == "no_reliability":
        return data[
            [
                "observed_distance_m",
                "face_similarity",
                "liveness_score",
                "frame_consistency",
            ]
        ]
    if kind == "interactions":
        return build_model_frame(data)
    return data[FEATURES]


def pooled(n_per_regime, seed, generator=simulate_attendpro):
    """Pool equal numbers from the five regimes, with a seed offset per regime."""
    return pd.concat(
        [generator(n_per_regime, regime, seed + i) for i, regime in enumerate(REGIMES)],
        ignore_index=True,
    )


def candidates(seed):
    """The original small search, including model-specific regularization."""
    for c in [0.1, 1, 10]:
        yield "Logistic interactions", "interactions", make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=500)
        )
    yield "Logistic additive", "base", make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500)
    )
    yield "Logistic no reliability", "no_reliability", make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500)
    )
    for knots in [4, 6]:
        yield "GAM additive", "base", make_pipeline(
            SplineTransformer(n_knots=knots, degree=3),
            StandardScaler(),
            LogisticRegression(C=1, max_iter=700),
        )
    for leaf in [10, 40]:
        yield "Random forest", "base", RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=leaf,
            max_features=1.0,
            n_jobs=2,
            random_state=seed,
        )
    for leaves in [15, 31]:
        yield "Gradient boosting", "base", HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=leaves, l2_regularization=1, random_state=seed
        )


def select_threshold(y, probabilities, target=0.01):
    """Maximize valid acceptance subject to empirical validation FAR.

    target=None maximizes balanced accuracy instead. ROC includes the infinite
    reject-all threshold. A validation constraint is not a test-set guarantee.
    """
    false_positive, true_positive, thresholds = roc_curve(
        y, probabilities, drop_intermediate=False
    )
    if target is None:
        return float(thresholds[np.argmax(true_positive - false_positive)])
    allowed = np.flatnonzero(false_positive <= target)
    best = allowed[np.argmax(true_positive[allowed])]
    return float(thresholds[best])


def _log_odds(probabilities):
    # Clipping prevents infinite inputs to the independent sigmoid calibrator.
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def scores(data, fitted):
    """Calibrated scores; unusable attempts receive zero and are rejected."""
    gate = usable(data)
    probabilities = np.zeros(len(data))
    if gate.any():
        raw = fitted["model"].predict_proba(features(data.loc[gate], fitted["kind"]))[
            :, 1
        ]
        probabilities[gate] = fitted["calibrator"].predict_proba(_log_odds(raw))[:, 1]
    return probabilities


def predict(data, fitted, operating_point="FAR_1pct"):
    """Return the final acceptance decisions, including the common gates."""
    return usable(data) & (
        scores(data, fitted) >= fitted["thresholds"][operating_point]
    )


def fit_models(train, validation, calibration, seed=20261001, methods=None):
    """Fit on training, select on validation, calibrate on separate calibration.

    Validation is reused for candidate selection and acceptance-cutoff selection,
    exactly as in the paper. No final test data are accepted by this function.
    methods=None fits all six families; a list supports smaller teaching runs.
    """
    if methods is not None and not set(methods) <= set(METHODS):
        raise ValueError(f"Unknown method; choose from {METHODS}")
    models = {}
    train_gate, val_gate, cal_gate = map(usable, (train, validation, calibration))
    for name, kind, estimator in candidates(seed):
        if methods is not None and name not in methods:
            continue
        estimator.fit(
            features(train.loc[train_gate], kind),
            train.loc[train_gate, "truth_valid_presence"],
        )
        raw = estimator.predict_proba(features(validation.loc[val_gate], kind))[:, 1]
        loss = brier_score_loss(validation.loc[val_gate, "truth_valid_presence"], raw)
        if name not in models or loss < models[name]["loss"]:
            models[name] = dict(model=estimator, kind=kind, loss=loss)
    for fitted in models.values():
        raw = fitted["model"].predict_proba(
            features(calibration.loc[cal_gate], fitted["kind"])
        )[:, 1]
        fitted["calibrator"] = LogisticRegression(C=1e6, max_iter=300).fit(
            _log_odds(raw), calibration.loc[cal_gate, "truth_valid_presence"]
        )
        probabilities = scores(validation, fitted)
        fitted["thresholds"] = {
            name: select_threshold(
                validation.truth_valid_presence.to_numpy(), probabilities, target
            )
            for name, target in {
                "BA": None,
                "FAR_1pct": 0.01,
                "FAR_0.1pct": 0.001,
            }.items()
        }
    return models


def rule(data, setting=(1, 150, 0.4, 0.42, 0.78)):
    """Conjunction: setting=(accuracy_weight, radius, similarity, liveness, consistency)."""
    weight, radius, similarity, liveness, consistency = setting
    return (
        usable(data)
        & (
            (data.observed_distance_m + weight * data.gps_accuracy_m).to_numpy()
            <= radius
        )
        & (data.face_similarity.to_numpy() >= similarity)
        & (data.liveness_score.to_numpy() >= liveness)
        & (data.frame_consistency.to_numpy() >= consistency)
    )


def evaluate(y, accepted):
    """Errors over all attempts, including rejected unusable attempts.

    Rates are fractions, not percentages. Missing class/subgroup rates are NaN.
    """
    y, accepted = np.asarray(y).astype(bool), np.asarray(accepted).astype(bool)
    valid, invalid = y, ~y
    far = float(accepted[invalid].mean()) if invalid.any() else float("nan")
    frr = float((~accepted[valid]).mean()) if valid.any() else float("nan")
    return dict(
        FAR=far,
        FRR=frr,
        BA=1 - (far + frr) / 2,
        n_valid=int(valid.sum()),
        n_invalid=int(invalid.sum()),
        false_accept=int(accepted[invalid].sum()),
        false_reject=int((~accepted[valid]).sum()),
    )


def tune_rule(validation, target=0.01):
    """Choose among the original 432 conjunction settings on validation only."""
    best_acceptance, best_setting = -1, (1, 0, 1, 1, 1)
    settings = product(
        [0, 0.5, 1],
        [140, 150, 160],
        [0.35, 0.4, 0.5, 0.6],
        [0.42, 0.62, 0.72, 0.82, 0.90, 0.95],
        [0.70, 0.78],
    )
    for setting in settings:
        result = evaluate(validation.truth_valid_presence, rule(validation, setting))
        objective = result["BA"] if target is None else 1 - result["FRR"]
        if (target is None or result["FAR"] <= target) and objective > best_acceptance:
            best_acceptance, best_setting = objective, setting
    return best_setting
