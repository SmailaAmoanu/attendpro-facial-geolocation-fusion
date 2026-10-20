"""Generate the original study from scratch; no saved models or large data needed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/full-study")
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Generate nominal/extension results and models; omit later stages.",
    )
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit("Choose a new --out directory to protect earlier runs.")
    output.mkdir(parents=True)
    work = output / "workspace"
    experiments = work / "experiments"
    shutil.copytree(ROOT / "archive/experiments", experiments)
    shutil.copy2(ROOT / "archive/review_diagnostics.py", work / "review_diagnostics.py")
    (work / "results").mkdir()
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    plan = [experiments / "run_extension.py"]
    if not args.nominal_only:
        plan += [
            experiments / "run_boundary_training.py",
            experiments / "report_boundary_training.py",
            experiments / "run_dependence.py",
            experiments / "verify_dependence.py",
        ]
    plan.append(work / "review_diagnostics.py")
    manifest = {
        "python": sys.version,
        "started_unix": time.time(),
        "source": "Original scripts; no packaged model prerequisites",
        "completed_scripts": [],
    }
    for script in plan:
        print(
            f"Running {script.name}; see {output / (script.stem + '.log')}", flush=True
        )
        with (output / f"{script.stem}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if script.name == "run_extension.py":
            # This convenience table was not written by the historical runner.
            frame = pd.read_csv(experiments / "extension/fit_level_metrics.csv")
            frame.groupby(["condition", "method", "operating_point"])[
                ["FAR", "FRR", "BA"]
            ].mean().reset_index().to_csv(
                experiments / "extension/mean_metrics.csv", index=False
            )
            # The boundary reporter checks these newly generated reference fits
            # remain unchanged during boundary adaptation. No prior archive is needed.
            hashes = {
                p.relative_to(experiments)
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in experiments.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
            (experiments / "SHA256SUMS.json").write_text(
                json.dumps(hashes, indent=2) + "\n"
            )
        manifest["completed_scripts"].append(script.relative_to(work).as_posix())
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished. Generated models, data and results are in {work}.")


if __name__ == "__main__":
    main()
