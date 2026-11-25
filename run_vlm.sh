#!/bin/bash
# Minimal VLM run: one model on all 103 scenes x their instruction variants (376 episodes), the paper's main setting.
# usage: bash run_vlm.sh <model in risk_scenes/vlm_models.json> [parallel=2] [seed=0] [scene ...]
# needs: LBJ-Bench installed (README), the model's API key in the env variable named by key_env.
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true          # MuJoCo leaks file handles on every env build: one process per scene
export MUJOCO_GL=${MUJOCO_GL:-egl} VLM_TIMEOUT=${VLM_TIMEOUT:-180}
M=$1; P=${2:-2}; SEED=${3:-0}; shift $(( $# < 3 ? $# : 3 ))
SCENES=${*:-$(python3 -c "import json;print(' '.join(json.load(open('instructions.json'))))")}
printf '%s\n' $SCENES | xargs -P "$P" -I{} "$SIMPY" risk_scenes/vlm_action.py "$M" after_neutral {} \
  --instructions instructions.json --tag main --seed "$SEED"
"$SIMPY" risk_scenes/vlm_action_summary.py results/vlm/main
