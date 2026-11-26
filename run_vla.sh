#!/bin/bash
# Run one VLA on the whole benchmark and print its results as in Table 1 of the paper.
# Start the policy server first in its own environment (see README).
# usage: bash run_vla.sh <pi05_libero|pi0_libero|pi05_libero_safety|openvla_oft|openvla> [port=8000] [trials=3] [scene ...]
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true
export MUJOCO_GL=${MUJOCO_GL:-egl}
M=$1; PORT=${2:-8000}; T=${3:-3}; shift $(( $# < 3 ? $# : 3 ))
SCENES=${*:-$("$SIMPY" -c "import json;print(' '.join(json.load(open('instructions.json'))))")}
for s in $SCENES; do
  for v in $("$SIMPY" -c "import json;print(' '.join(json.load(open('instructions.json'))['$s']))"); do
    for t in $(seq 0 $((T - 1))); do
      "$SIMPY" vla/run_vla.py "$M" "$s" --instr "$v" --trial "$t" --port "$PORT"
    done
  done
done
"$SIMPY" risk_scenes/summarize.py results/vla
