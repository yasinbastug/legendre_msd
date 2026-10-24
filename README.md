# Legendre-MSD

Code and results for *"Receiver-Surface Hit Patterns via Legendre Approximation
for Molecular Signal Detection"* (Y. Bastug, E. Ozbey, H. B. Yilmaz; submitted
to IEEE Communications Letters).

A point transmitter (Tx) releases molecules into unbounded 3-D diffusion, and a
perfectly absorbing spherical receiver (Rx) records where and when each molecule
is absorbed. Besides the molecule **count**, the detectors use the **angular
distribution** of absorption points, modelled per arrival-time bin and per ISI
tap with a finite Legendre expansion in `z = cos θ` (θ = angle between the Tx
direction and the absorption point).

## Detectors

| Key | Detector | Module |
|---|---|---|
| `fixed_threshold` | Count threshold (equal-prior Poisson crossing) | `fixed_threshold.py` |
| `viterbi` | Count-only Poisson Viterbi | `viterbi.py` |
| `legendre` | Memoryless Legendre likelihood-ratio test | `legendre.py` |
| `legendre_viterbi` | Hybrid: count trellis + one Legendre shape term per symbol | `legendre_viterbi.py` |
| `legendre_viterbi_full` | Full Legendre-Viterbi: angular-time intensity per branch | `legendre_viterbi_full.py` |

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

PyTorch runs the Brownian simulator and the full Legendre-Viterbi detector. Both
use CUDA when available; the simulator also uses Apple-silicon GPUs (MPS), and
everything runs on CPU, more slowly. For CUDA, install the matching PyTorch
build from <https://pytorch.org>.

Check the installation with the end-to-end smoke test, which simulates its own
small channel in a temporary folder (about a minute):

```bash
python -m pytest tests
```

## Reproducing the results

```bash
./run_all.sh paper        # Figs. 2-4 of the paper
./run_all.sh reviewer     # all revision experiments
python make_figures.py    # redraw the paper figures from the results
```

`./run_all.sh` lists all groups. Each experiment writes
`experiments/<paper|reviewer>_<sweep>__<options>/summary.csv` plus a
`manifest.json`, and `plots/<...>.pdf`. Finished points are skipped on a rerun,
so an interrupted run resumes where it stopped. Missing hit pools are simulated
automatically and cached in `simulations/` (not in git: about 200 MB per channel
at 100,000 bits).

All runs use 100,000 BCSK bits (seed 2026), `r = 5 µm`, `d = 12.5 µm`,
`D = 79.4 µm²/s`, simulation step `1e-4 s`, Legendre order `L = 10`, ridge
penalty `γ = 1e-6`, numerical floor `ε = 1e-12`, `B = 100` arrival-time bins
and, unless a sweep changes it, `N_Tx = 100`, `T_s = 0.2 s` and no
environmental noise.

### Channel memory

The detectors model `K` ISI taps, the smallest number of symbols for which
`F_hit` reaches 75% of its asymptote `r/d` (`K = 18` at `T_s = 0.2 s`, `K = 7` at
`T_s = 0.5 s`), capped by the trellis size `2^(K-1)`. By default the simulated
channel ends at the same point, as in the submitted paper. A longer channel is
selected with `--channel-memory SECONDS` or `--channel-target FRACTION`, and
`--tail-background` models the ISI beyond `K` as uniform background, adding its
expected count to the environmental-noise term:

```bash
python experiments.py ntx --channel-target 0.9 --tail-background
python make_figures.py --match cht0p9__tailbg
```

## Experiments

`python experiments.py <sweep>` runs one sweep (`--help` lists all options).

| Sweep | Varies | Default values |
|---|---|---|
| `ntx` | molecules per bit `N_Tx` (paper Figs. 2, 4) | 25 … 250 |
| `ts` | symbol duration `T_s` [s] (paper Fig. 3) | 0.2, 0.3, 0.4, 0.5 |
| `noise` | environmental noise `λ_env/λ_0`, uniform on the Rx surface and in time | 0 … 4 |
| `L` | Legendre order | 2 … 20 |
| `gamma` | ridge penalty | 0 … 100 |
| `floor` | numerical floor | 1e-12 … 1e-1 |
| `B` | arrival-time bins | 10 … 200 |
| `fit` | held-out accuracy of the Legendre density vs. `L` (no detection) | 0 … 30 |
| `memory` | detector memory as `F_hit` fraction | 0.5 … 0.75 |
| `K` | detector memory in taps | 1 … 20 |
| `tail` | ISI beyond `K` ignored (0) or modelled as background (1) | 0, 1 |
| `chmem` | channel memory [s], detector memory fixed by `--detector-taps` | 7.5 … 30 |
| `chtarget` | channel memory as `F_hit` fraction, detector memory fixed | 0.6 … 0.9 |
| `direction` | error in the assumed Tx direction [deg] | 0 … 90 |
| `distance` | Tx-Rx distance `d` [µm], own channel and library per value | 10 … 17.5 |
| `radius` | Rx radius `r` [µm], own channel and library per value | 3 … 7 |
| `diffusion` | diffusion coefficient `D` [µm²/s], own channel and library per value | 40 … 160 |

`python experiments.py plot <sweep> [same options]` redraws one sweep from its CSV.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Brownian simulator (hit pools), Legendre-library construction, received-signal generation, background noise, single-run driver |
| `legendre.py` | Legendre library fit and the memoryless Legendre detector |
| `viterbi.py`, `legendre_viterbi.py`, `legendre_viterbi_full.py`, `fixed_threshold.py` | the other detectors |
| `pipeline.py` | runs all detectors on one scenario; channel-memory helpers; figure styles |
| `experiments.py` | all sweeps |
| `make_figures.py` | paper figures from the results |
| `generate_figure_angular.py` | Fig. 1 (angular surface density) |
| `run_all.sh` | reproduces the experiments |
| `tests/` | end-to-end smoke test |
| `experiments/`, `plots/` | results and figures; `legacy_submission/` holds the submitted version's results |
| `legendre_files/` | cached Legendre libraries |

## Results of the submitted version

`experiments/legacy_submission/` and `plots/legacy_submission/` contain the
results behind the initially submitted manuscript. They differ from the current
code in two ways: the memoryless Legendre detector, the hybrid's shape term and
the fixed threshold assumed `λ_env = 1` although no environmental noise was
simulated, and the full detector shifted arrival-time bins by one when a symbol
had no arrival in the first bin. At `N_Tx = 100`, `T_s = 0.2 s` the corrected
BERs are within 0.006 of the submitted ones.

## Citation and license

See `CITATION.cff` for how to cite. The code is released under the MIT License (`LICENSE`).
