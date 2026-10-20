"""Fast checks for the light package; no saved model files are required."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from attendpro_paper2 import FEATURES, rule, simulate_attendpro, usable
from attendpro_paper2.boundary import boundary
from attendpro_paper2.dependence import CONFIGS, generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = simulate_attendpro(2000, "nominal", 12345)
    truth = original.scenario.isin(
        ["genuine_inside", "genuine_boundary_inside", "genuine_degraded_inside"]
    )
    truth &= original.true_distance_m <= 150
    assert np.array_equal(truth, original.truth_valid_presence.astype(bool))
    assert not {"scenario", "true_distance_m", "truth_valid_presence"} & set(FEATURES)
    assert np.array_equal(rule(original), original.M4_current_attendpro.astype(bool))
    moved = boundary(original, 23456, fraction=1)
    for col in [
        "face_similarity",
        "face_quality_score",
        "liveness_score",
        "truth_valid_presence",
    ]:
        assert np.array_equal(moved[col], original[col])
    preserved = [
        "scenario",
        "truth_valid_presence",
        "observed_distance_m",
        "gps_accuracy_m",
        "quality_pass",
        "face_quality_score",
    ]
    for _, family, rho in CONFIGS:
        paired = generate(2000, "nominal", 12345, family, rho)
        pd.testing.assert_frame_equal(
            paired[preserved], original[preserved], check_exact=True
        )
        assert np.array_equal(usable(paired), usable(original))
    nominal = pd.read_csv(ROOT / "results/paper/extension_mean_metrics.csv")
    row = nominal[
        (nominal.method == "Gradient boosting")
        & (nominal.operating_point == "FAR_1pct")
    ].iloc[0]
    assert round(100 * row.FAR, 2) == 0.55
    assert round(100 * row.FRR, 2) == 20.95
    print(
        json.dumps(
            {
                "status": "passed",
                "paired_copula_conditions": 6,
                "truth_gates_and_boundary_invariants": "passed",
                "saved_model_dependency": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
