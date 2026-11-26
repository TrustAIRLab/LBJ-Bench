#!/bin/bash
# Run one VLM agent on the whole benchmark (376 episodes), or on the scenes given.
# usage: bash run_vlm.sh <model in risk_scenes/vlm_models.json> [parallel=2] [scene ...]
# needs: the model's API key. With OPENAI_API_KEY set, the refusals are also labelled by the LLM judge (gpt-6-luna)
# and the results are printed as in Table 1 of the paper.
set -e
cd "$(dirname "$0")"
source LIBERO-Safety/sim_env.sh
ulimit -n 65536 2>/dev/null || true
export MUJOCO_GL=${MUJOCO_GL:-egl} VLM_TIMEOUT=${VLM_TIMEOUT:-180}
M=$1; P=${2:-2}; shift $(( $# < 2 ? $# : 2 ))
SCENES=${*:-$("$SIMPY" -c "import json;print(' '.join(json.load(open('instructions.json'))))")}
printf '%s\n' $SCENES | xargs -P "$P" -I{} "$SIMPY" risk_scenes/vlm_action.py "$M" after_neutral {} --tag main
if [ -n "${OPENAI_API_KEY:-}" ]; then
  "$SIMPY" risk_scenes/refusal_judge.py results/vlm/main
  "$SIMPY" risk_scenes/summarize.py results/vlm/main
else
  echo "Episodes are in results/vlm/main. For the results table, set OPENAI_API_KEY and run:"
  echo "  \$SIMPY risk_scenes/refusal_judge.py results/vlm/main && \$SIMPY risk_scenes/summarize.py results/vlm/main"
fi
