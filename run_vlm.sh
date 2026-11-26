#!/bin/bash
# Run one VLM agent on the whole benchmark (376 episodes) and print its results as in Table 1 of the paper.
# usage: bash run_vlm.sh <model in risk_scenes/vlm_models.json> [parallel=2] [scene ...]
# needs: the model's API key, and OPENAI_API_KEY for the refusal judge (gpt-6-luna).
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true
export MUJOCO_GL=${MUJOCO_GL:-egl} VLM_TIMEOUT=${VLM_TIMEOUT:-180}
M=$1; P=${2:-2}; shift $(( $# < 2 ? $# : 2 ))
SCENES=${*:-$("$SIMPY" -c "import json;print(' '.join(json.load(open('instructions.json'))))")}
printf '%s\n' $SCENES | xargs -P "$P" -I{} "$SIMPY" risk_scenes/vlm_action.py "$M" after_neutral {} --tag main
"$SIMPY" risk_scenes/refusal_judge.py results/vlm/main
"$SIMPY" risk_scenes/summarize.py results/vlm/main
