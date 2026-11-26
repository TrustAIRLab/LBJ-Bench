#!/bin/bash
# Run one VLM agent.
#   bash run_vlm.sh <model> <parallel> <scene> [scene ...]    some scenes; needs the model's API key
#   bash run_vlm.sh <model> [parallel=2]                      the whole benchmark (376 episodes) and the results as in
#                                                             Table 1 of the paper; also needs OPENAI_API_KEY, because
#                                                             the refusals are labelled by an LLM judge (gpt-6-luna)
# Models are listed in risk_scenes/vlm_models.json.
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true
export MUJOCO_GL=${MUJOCO_GL:-egl} VLM_TIMEOUT=${VLM_TIMEOUT:-180}
M=$1; P=${2:-2}; shift $(( $# < 2 ? $# : 2 ))
if [ $# -gt 0 ]; then
  printf '%s\n' "$@" | xargs -P "$P" -I{} "$SIMPY" risk_scenes/vlm_action.py "$M" after_neutral {} --tag main
  exit
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "the whole benchmark needs OPENAI_API_KEY for the refusal judge"; exit 1; }
"$SIMPY" -c "import json;print('\n'.join(json.load(open('instructions.json'))))" | xargs -P "$P" -I{} "$SIMPY" risk_scenes/vlm_action.py "$M" after_neutral {} --tag main
"$SIMPY" risk_scenes/refusal_judge.py results/vlm/main
"$SIMPY" risk_scenes/summarize.py results/vlm/main
