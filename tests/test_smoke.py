"""End-to-end smoke test on a tiny simulated channel (about a minute on a CPU).

Everything runs in a temporary project root (LEGENDRE_MSD_ROOT), so no hit pool,
library or result in the repository is read or written. The test simulates its
own Brownian hit pools, runs the paper sweeps and one revision sweep with all
five detectors, and draws the paper figures.

    python -m pytest tests          # or: python tests/test_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DETECTORS = {"fixed_threshold", "viterbi", "legendre", "legendre_viterbi", "legendre_viterbi_full"}
# Tiny but complete: 1 s of ISI (5 taps at T_s = 0.2 s), 200 bits.
SMALL = ["--n-bits", "200", "--channel-memory", "1.0", "--n-emit-distribution", "20000", "--full-device", "cpu"]


def run(root: Path, *args: str) -> None:
    env = dict(os.environ, LEGENDRE_MSD_ROOT=str(root), MPLBACKEND="Agg")
    subprocess.run([sys.executable, *args], cwd=REPO, env=env, check=True)


def summary(root: Path, prefix: str) -> pd.DataFrame:
    (path,) = (root / "experiments").glob(f"{prefix}__*/summary.csv")
    return pd.read_csv(path)


def check_operating_points(df: pd.DataFrame, column: str, values: set[float]) -> None:
    points = df[df["point_kind"] == "operating_point"]
    assert set(points[column].round(6)) == values
    for _, group in points.groupby(column):
        assert set(group["detector"]) == DETECTORS
    assert points["BER"].between(0.0, 0.6).all()
    assert (points["lambda_env"] == 0.0).all()  # no environmental noise unless injected
    assert {"N_Tx", "ts", "detector_memory_taps", "channel_memory_taps"} <= set(points.columns)


def test_pipeline_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        run(root, "experiments.py", "ntx", "--values", "50", "100", *SMALL)
        ntx = summary(root, "paper_ntx")
        check_operating_points(ntx, "N_Tx", {50.0, 100.0})
        curves = ntx[ntx["point_kind"] == "decision_boundary"]
        assert set(curves["detector"]) == {"legendre", "fixed_threshold"}
        assert curves.groupby(["N_Tx", "detector"]).size().max() <= 401  # thinned

        run(root, "experiments.py", "ts", "--values", "0.2", "0.5", *SMALL)
        check_operating_points(summary(root, "paper_ts"), "ts", {0.2, 0.5})

        run(root, "experiments.py", "noise", "--values", "0", "1", *SMALL)
        noise = summary(root, "reviewer_noise")
        assert set(noise["lambda_env"].round(6)) == {0.0, round(float(noise["lambda_0"].iloc[0]), 6)}

        # Resume: a rerun skips finished points and adds no rows.
        before = len(summary(root, "paper_ntx"))
        run(root, "experiments.py", "ntx", "--values", "50", "100", *SMALL)
        assert len(summary(root, "paper_ntx")) == before

        run(root, "make_figures.py")
        for name in ("figure_1_ber_vs_NTx.pdf", "figure_2_pfa_vs_pmiss_NTx100_rec.pdf",
                     "figure_3_ber_vs_TS.pdf", "figure_4_ber_vs_noise.pdf"):
            assert (root / "plots" / name).stat().st_size > 0


if __name__ == "__main__":
    test_pipeline_end_to_end()
    print("smoke test passed")
