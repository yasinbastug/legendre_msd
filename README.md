# Legendre-MSD: Receiver-Surface Hit Patterns via Legendre Approximation for Molecular Signal Detection

Code for the paper *"Receiver-Surface Hit Patterns via Legendre Approximation
for Molecular Signal Detection"* (Bastug, Ozbey, Yilmaz).

The receiver is a perfectly absorbing spherical Rx in an unbounded 3-D diffusive
medium. Instead of using only the molecule **count**, the detectors model the
**angular distribution** of absorption locations on the Rx surface via a finite
Legendre-polynomial expansion, and combine it with a Poisson count model.

## Detectors

| Module | Detector | Description |
|---|---|---|
| `fixed_threshold.py` | Fixed threshold | Count-only, single-symbol threshold. |
| `viterbi.py` | Count Viterbi | Count-only Poisson sequence detector. |
| `legendre.py` | Legendre | Memoryless angular likelihood-ratio detector; also builds the Legendre library. |
| `legendre_viterbi.py` | Hybrid Legendre-Viterbi | State-conditioned count recursion + one precomputed Legendre shape term per symbol. |
| `legendre_viterbi_full.py` | Full Legendre-Viterbi | State-dependent angular-time intensity recomputed per branch (best BER). |

## Pipeline

- `main.py` — Brownian-motion simulator (generates the hit pools) and single-run driver; builds the Legendre library.
- `generate_figures.py` — runs all detectors over the N_Tx and T_s sweeps and produces the paper figures.
- `regenerate_from_csv.py` — rebuilds the figures from existing experiment summary CSVs, without re-running any detector.
- `generate_figure_angular_pdf_v2.py` — produces the angular surface-density figure (Fig. 1).

## Data layout

- `simulations/` — Brownian first-hit pools (`.pt`).
- `legendre_files/` — learned Legendre libraries (`.npz`).
- `experiments/` — per-run summary CSVs + `manifest.json`.
- `plots/` — figures (`.pdf`/`.png`) and their source CSVs.

## Parameters (as used in the paper)

`L = 10`, `B = 100`, `r = 5 µm`, `D = 79.4 µm²/s`, `d = 12.5 µm`,
simulation step `dt = 1e-4 s`. Memory length is the smallest number of symbols
for which `F_hit` reaches 75% of its asymptotic value. No environmental noise
(`lambda_env = 0`), so the noise component is ISI only.

## Usage

```bash
pip install -r requirements.txt

# full experiment sweeps + figures (re-runs detectors)
python generate_figures.py

# rebuild figures from existing summary CSVs (no detectors re-run)
python regenerate_from_csv.py
```
