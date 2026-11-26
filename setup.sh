#!/bin/bash
# One-step setup of LBJ-Bench. Run from the repo root:
#   bash setup.sh                           everything except the baby objects
#   bash setup.sh --accept-smil-license     also downloads SMIL and generates the baby objects
# It clones LIBERO-Safety, downloads its assets (about 11 GB), creates the Python environment, downloads the
# LBJ-Bench data and installs it. Every step is skipped when its result is already there, so it can be rerun.
# The baby objects come from the SMIL body model, which may only be used for non-commercial research. Read
#   https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil_license.txt
# and pass --accept-smil-license only if you agree. Without the babies, 23 of the 103 scenes will not load.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SMIL=0; [ "${1:-}" = "--accept-smil-license" ] && SMIL=1
for tool in git curl unzip uv; do
  command -v "$tool" >/dev/null || { echo "missing: $tool$([ "$tool" = uv ] && echo ' (install: curl -LsSf https://astral.sh/uv/install.sh | sh)')"; exit 1; }
done
LS=LIBERO-Safety; A=$LS/libero/libero/assets

echo "[1/5] LIBERO-Safety"
[ -d $LS/.git ] || git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git $LS
git -C $LS checkout -q "$(cat libero_overlay/UPSTREAM_COMMIT)"

echo "[2/5] LIBERO-Safety assets"
if [ ! -d $A/scenes ]; then
  curl -L -C - -o $LS/assets.zip https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/resolve/main/assets.zip
  unzip -q -o $LS/assets.zip -d $LS/libero/libero/ && rm $LS/assets.zip
fi

echo "[3/5] Python environment"
PY=$LS/.venv-sim/bin/python
[ -x $PY ] || uv venv $LS/.venv-sim --python 3.10
uv pip install -q --python $PY -r requirements-sim.txt
uv pip install -q --python $PY -e $LS/third_party/robosuite-1.4
uv pip install -q --python $PY "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
uv pip install -q --python $PY -e $LS

echo "[4/5] LBJ-Bench data and objects"
bash libero_overlay/install.sh $LS

echo "[5/5] Baby objects"
if [ -f $A/risk_objects/baby_supine/baby_supine.xml ]; then
  echo "already there"
elif [ $SMIL = 1 ]; then
  mkdir -p $LS/.smil
  [ -f $LS/.smil/smil/smil_web.pkl ] || { curl -L -o $LS/.smil/smil.zip https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil.zip; unzip -q -o $LS/.smil/smil.zip -d $LS/.smil/; }
  $PY risk_scenes/make_baby.py $LS/.smil/smil/smil_web.pkl
else
  echo "skipped: rerun with --accept-smil-license to generate them (23 scenes need them)"
fi

echo
echo "Done. In every new shell:  source $LS/sim_env.sh"
echo "Check the installation:    MUJOCO_GL=egl \$SIMPY risk_scenes/check_judge.py noop"
