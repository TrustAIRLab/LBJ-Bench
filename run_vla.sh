#!/bin/bash
# Minimal VLA run: one policy on the hazardous and control scenes with every instruction variant.
# Start the policy server first in its own environment, e.g.
#   python vla/serve_openpi.py pi05_libero 8000        (openpi env; checkpoint in vla/checkpoints/pi05_libero)
#   OPENVLA_OFT_DIR=... python vla/serve_openvla.py openvla_oft 8000
# usage: bash run_vla.sh <pi05_libero|pi0_libero|pi05_libero_safety|openvla_oft|openvla> [port=8000] [trials=3] [scene ...]
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true
export MUJOCO_GL=${MUJOCO_GL:-egl}
M=$1; PORT=${2:-8000}; T=${3:-3}; shift $(( $# < 3 ? $# : 3 ))
SCENES=${*:-$(python3 -c "import json;print(' '.join(json.load(open('instructions.json'))))")}
for s in $SCENES; do
  for v in $(python3 -c "import json;print(' '.join(json.load(open('instructions.json'))['$s']))"); do
    for t in $(seq 0 $((T - 1))); do
      "$SIMPY" vla/run_vla.py "$M" "$s" --instr "$v" --trial "$t" --port "$PORT" --states
    done
  done
done
