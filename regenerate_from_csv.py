"""Regenerate paper figures from existing experiment summary CSVs.

Does NOT re-run any detector. It loads the latest baseline (Fig. 1/2/4) and
TS-sweep (Fig. 3) summary CSVs and calls the same plotting functions used by
generate_figures_v2_new.py, so styling and detector set match a full run.

Usage:
    python regenerate_from_csv.py [BASELINE_SUMMARY.csv] [TS_SWEEP_SUMMARY.csv]
If paths are omitted, the newest matching summaries under experiments/ are used.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

import generate_figures as g

HERE = Path(__file__).resolve().parent


def newest(pattern: str) -> str:
    matches = glob.glob(str(HERE / pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No summary CSV matches: {pattern}")
    return max(matches, key=os.path.getmtime)


def main() -> None:
    if len(sys.argv) >= 3:
        baseline_csv, ts_csv = sys.argv[1], sys.argv[2]
    else:
        baseline_csv = newest("experiments/figures__*/**summary.csv")
        ts_csv = newest("experiments/figure_3_ts_sweep__*/**summary.csv")

    print("baseline summary:", baseline_csv)
    print("ts-sweep summary:", ts_csv)

    baseline_df = pd.read_csv(baseline_csv)
    ts_sweep_df = pd.read_csv(ts_csv)

    plots_dir = g.maybe_make_out_dir(HERE / "plots")
    fig1 = plots_dir / "figure_1_ber_vs_NTx.pdf"
    fig2 = plots_dir / "figure_2_pfa_vs_pmiss_NTx100.pdf"
    fig3 = plots_dir / "figure_3_ber_vs_TS.pdf"
    fig4 = plots_dir / "figure_4_runtime_comparison.pdf"

    g.save_figure_1_ber_vs_ntx(baseline_df, fig1)
    g.save_figure_2_false_alarm_vs_missed_detection(baseline_df, fig2, n_tx=g.FIGURE_2_N_TX)
    g.save_figure_3_ber_vs_ts(ts_sweep_df, fig3)
    g.save_figure_4_runtime(baseline_df, fig4)
    csvs = g.save_plot_csvs(baseline_df, ts_sweep_df, plots_dir)

    print("\nSaved figures:")
    for f in (fig1, fig2, fig3, fig4):
        print("  ", f)
    print("Saved plot CSVs:")
    for f in csvs.values():
        print("  ", f)


if __name__ == "__main__":
    main()
