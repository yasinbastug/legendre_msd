"""Draw the paper figures from experiment results (no detector is re-run).

Reads the summary CSVs written by experiments.py and saves, under plots/:

    figure_1_ber_vs_NTx.pdf               BER vs. N_Tx               (ntx sweep)
    figure_2_pfa_vs_pmiss_NTx100_rec.pdf  P01 vs. P10 at N_Tx = 100  (ntx sweep)
    figure_3_ber_vs_TS.pdf                BER vs. T_s                (ts sweep)
    figure_4_ber_vs_noise.pdf             BER vs. lambda_env/lambda_0 (noise sweep)
    figure_runtime_comparison.pdf         runtime per detector       (ntx sweep)

plus the CSVs behind each figure (file names match the LaTeX source), and one
plot per experiment folder (plots/<experiment>.pdf), including the chtarget
comparison plots.

By default the newest matching experiment folder is used for each sweep; pass
--match to pick a variant, e.g. --match cht0p9__tailbg for the runs with a 90%
channel and the ISI tail modelled as background. A variant's paper figures are
saved under plots/<match>/.

Usage:
    python make_figures.py
    python make_figures.py --match cht0p9__tailbg
    python make_figures.py --match B10__gamma1     # runs with --B 10 --gamma 1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

import pipeline as gf
from experiments import SWEEPS, parse_cli, plot_chtarget_comparison, plot_sweep

FIGURE_2_N_TX = 100


# Folder parts written by experiments.py --L/--gamma/--B (operating-point variants).
OPERATING_POINT_PART = re.compile(r"^(L\d+|gamma\S+|B\d+)$")


def newest_summary(sweep_prefix: str, match: str) -> Path | None:
    """Newest experiments/<sweep_prefix>__*<match>*/summary.csv, or None.

    Without --match, operating-point variants (__B10, __gamma1, ...) are skipped.
    """
    candidates = [
        p for p in (gf.PROJECT_ROOT / "experiments").glob(f"{sweep_prefix}__*/summary.csv")
        if match in p.parent.name
        and (match or not any(OPERATING_POINT_PART.match(part) for part in p.parent.name.split("__")[1:]))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", default="", help="substring the experiment folder names must contain")
    opts = parser.parse_args()

    # Paper figures of a variant go to plots/<match>/ so the default ones are kept.
    plots_dir = gf.maybe_make_out_dir(gf.PROJECT_ROOT / "plots" / opts.match if opts.match else gf.PROJECT_ROOT / "plots")
    saved = []

    ntx_csv = newest_summary("paper_ntx", opts.match)
    ts_csv = newest_summary("paper_ts", opts.match)
    noise_csv = newest_summary("reviewer_noise", "")

    if ntx_csv is not None:
        print("N_Tx sweep:", ntx_csv.parent.name)
        ntx_df = pd.read_csv(ntx_csv)
        out = plots_dir / "figure_1_ber_vs_NTx.pdf"
        gf.save_figure_1_ber_vs_ntx(ntx_df, out)
        saved.append(out)
        out = plots_dir / "figure_2_pfa_vs_pmiss_NTx100_rec.pdf"
        gf.save_figure_2_false_alarm_vs_missed_detection(ntx_df, out, n_tx=FIGURE_2_N_TX)
        saved.append(out)
        out = plots_dir / "figure_runtime_comparison.pdf"
        gf.save_figure_4_runtime(ntx_df, out)
        saved.append(out)
    else:
        print("No paper_ntx results found: run `python experiments.py ntx` first.")

    if ts_csv is not None:
        print("T_s sweep: ", ts_csv.parent.name)
        ts_df = pd.read_csv(ts_csv)
        out = plots_dir / "figure_3_ber_vs_TS.pdf"
        gf.save_figure_3_ber_vs_ts(ts_df, out)
        saved.append(out)
    else:
        print("No paper_ts results found: run `python experiments.py ts` first.")

    if ntx_csv is not None and ts_csv is not None:
        saved.extend(gf.save_plot_csvs(ntx_df, ts_df, plots_dir).values())

    if noise_csv is not None:
        print("Noise sweep:", noise_csv.parent.name)
        reviewer_plot = plot_sweep(noise_csv, "noise")
        out = plots_dir / "figure_4_ber_vs_noise.pdf"
        shutil.copyfile(reviewer_plot, out)
        saved.append(out)

    # One plot per experiment folder (<prefix>_<sweep>__<options>).
    for summary_csv in sorted((gf.PROJECT_ROOT / "experiments").glob("*/summary.csv")):
        head = summary_csv.parent.name.split("__")[0]
        sweep = head.split("_", 1)[1] if "_" in head else ""
        if sweep in SWEEPS or sweep == "fit":
            saved.append(plot_sweep(summary_csv, sweep))

    # Tail ignored vs. modelled as background, per T_s with chtarget results.
    for ts, taps in ((0.2, 18), (0.5, 7)):
        cli = parse_cli(["plot", "chtarget", "--ts", str(ts), "--detector-taps", str(taps)])
        try:
            saved.append(plot_chtarget_comparison(cli))
        except SystemExit:
            pass

    print("\nSaved:")
    for path in saved:
        print("  ", path)


if __name__ == "__main__":
    main()
