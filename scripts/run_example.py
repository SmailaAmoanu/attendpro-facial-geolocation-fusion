"""A small, readable end-to-end run. This is not a reproduction of paper values."""

import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from attendpro_paper2 import (
    evaluate,
    fit_models,
    pooled,
    predict,
    rule,
    simulate_attendpro,
    tune_rule,
)
from attendpro_paper2.boundary import boundary
from attendpro_paper2.dependence import CONFIGS, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/example"))
    parser.add_argument("--train-per-regime", type=int, default=600)
    parser.add_argument("--test-n", type=int, default=3000)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output already exists. Choose a new --out directory.")
    args.out.mkdir(parents=True)

    # These are independent draws, not a random split of one previously tuned set.
    seed = 20261001
    train = pooled(args.train_per_regime, seed)
    validation = pooled(max(300, args.train_per_regime // 2), seed + 100)
    calibration = pooled(max(300, args.train_per_regime // 2), seed + 200)
    models = fit_models(train, validation, calibration)
    setting = tune_rule(validation)

    # Freeze every model and cutoff before generating these test populations.
    nominal = simulate_attendpro(args.test_n, "nominal", seed + 1000)
    near_boundary = boundary(nominal, seed + 91000, fraction=1)
    rows = []
    populations = [("nominal", nominal), ("near_boundary_frozen", near_boundary)]
    for label, family, rho in CONFIGS:
        populations.append(
            (
                f"dependence_frozen: {label}",
                generate(args.test_n, "nominal", seed + 1000, family, rho),
            )
        )
    for population, data in populations:
        decisions = {
            "Reference rule": rule(data),
            "Tuned hard rule": rule(data, setting),
        }
        decisions.update(
            {name: predict(data, fitted) for name, fitted in models.items()}
        )
        for method, accepted in decisions.items():
            rows.append(
                dict(
                    population=population,
                    method=method,
                    **evaluate(data.truth_valid_presence, accepted),
                )
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.out / "metrics.csv", index=False)
    nominal.to_csv(args.out / "nominal_attempts.csv", index=False)
    (args.out / "settings.json").write_text(
        json.dumps(
            {
                "purpose": "Small learning example, not paper replication",
                "seed": seed,
                "train_per_regime": args.train_per_regime,
                "test_n": args.test_n,
                "tuned_rule": setting,
                "note": "All shifted examples use frozen models. Use Notebook 03 for redevelopment.",
            },
            indent=2,
        )
        + "\n"
    )
    print(results[results.population == "nominal"].to_string(index=False))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
