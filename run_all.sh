#!/usr/bin/env bash
# Reproduce the experiments of the paper and its revision.
#
#   ./run_all.sh paper          Figs. 2-4 (channel ends at the detector memory, as submitted)
#   ./run_all.sh paper-tail     Figs. 2-4 with a 90% F_hit channel, ISI tail as background
#   ./run_all.sh reviewer       all revision experiments
#   ./run_all.sh all            paper + paper-tail + reviewer
#   ./run_all.sh <name> ...     single steps, e.g. ./run_all.sh noise L
#
# Finished points are skipped, so the script can simply be rerun after an
# interruption. Logs go to logs/<step>.log. Figures: python make_figures.py
#
# Environment: PYTHON (default python3), N_BITS (default 100000).
set -uo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
N_BITS=${N_BITS:-100000}
EXP=("$PYTHON" -u experiments.py --n-bits "$N_BITS")

# experiments.py arguments for each step (a function, so it works with bash 3.2)
step_args() {
  case "$1" in
    ntx)               echo "ntx" ;;
    ts)                echo "ts" ;;
    ntx-tail)          echo "ntx --channel-target 0.9 --tail-background" ;;
    ts-tail)           echo "ts --channel-target 0.9 --tail-background" ;;
    noise|fit|L|gamma|floor|B|memory|direction|distance|radius|diffusion) echo "$1" ;;
    tail)              echo "tail --channel-memory 10" ;;
    K)                 echo "K --ts 0.5 --channel-memory 10 --values 1 2 3 4 5 6 7 8 10 12 14 16 18" ;;
    chtarget-0p5)      echo "chtarget --ts 0.5 --detector-taps 7" ;;
    chtarget-0p5-tail) echo "chtarget --ts 0.5 --detector-taps 7 --tail-background" ;;
    chtarget-0p2)      echo "chtarget --ts 0.2 --detector-taps 18 --values 0.6 0.65 0.7 0.75 0.8 0.85 0.8999" ;;
    chtarget-0p2-tail) echo "chtarget --ts 0.2 --detector-taps 18 --tail-background --values 0.6 0.65 0.7 0.75 0.8 0.85 0.8999" ;;
    *)                 return 1 ;;
  esac
}
PAPER=(ntx ts)
PAPER_TAIL=(ntx-tail ts-tail)
REVIEWER=(noise fit L gamma floor B memory direction tail K
          chtarget-0p5 chtarget-0p5-tail chtarget-0p2 chtarget-0p2-tail
          distance radius diffusion)

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  echo "Steps: ${PAPER[*]} ${PAPER_TAIL[*]} ${REVIEWER[*]}"
  exit 1
}

[[ $# -gt 0 ]] || usage
steps=()
for arg in "$@"; do
  case "$arg" in
    paper)      steps+=("${PAPER[@]}") ;;
    paper-tail) steps+=("${PAPER_TAIL[@]}") ;;
    reviewer)   steps+=("${REVIEWER[@]}") ;;
    all)        steps+=("${PAPER[@]}" "${PAPER_TAIL[@]}" "${REVIEWER[@]}") ;;
    *)          step_args "$arg" > /dev/null || { echo "Unknown step: $arg"; usage; }
                steps+=("$arg") ;;
  esac
done

mkdir -p logs
echo "$($PYTHON --version); $N_BITS bits; steps: ${steps[*]}"
failed=()
for step in "${steps[@]}"; do
  echo "=== $step  ($(date))  -> logs/$step.log"
  # shellcheck disable=SC2086
  if ! "${EXP[@]}" $(step_args "$step") >> "logs/$step.log" 2>&1; then
    echo "    FAILED, see logs/$step.log"
    failed+=("$step")
  fi
done

echo "Done ($(date))."
[[ ${#failed[@]} -eq 0 ]] || { echo "Failed steps: ${failed[*]}"; exit 1; }
