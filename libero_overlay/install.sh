#!/bin/bash
# Install LBJ-Bench into a LIBERO-Safety checkout: get the benchmark data, copy it together with the object
# registration, and write a repo-local LIBERO config.
# usage (from the repo root): bash libero_overlay/install.sh [path/to/LIBERO-Safety]   (default ./LIBERO-Safety)
# The data (scenes, initial states, objects, instructions) is downloaded from Hugging Face into ./data.
# Set LBJ_DATA=<folder> to use a copy you already have.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DST="$(cd "${1:-$ROOT/LIBERO-Safety}" && pwd)"
want=$(cat "$HERE/UPSTREAM_COMMIT"); have=$(git -C "$DST" rev-parse HEAD)
[ "$want" = "$have" ] || echo "warning: LIBERO-Safety is at $have, the benchmark was built on $want"
DATA="${LBJ_DATA:-$ROOT/data}"
if [ ! -f "$DATA/instructions.json" ]; then
  REPO="$(cat "$HERE/HF_REPO")"
  PY="$DST/.venv-sim/bin/python"; [ -x "$PY" ] || PY=python3
  echo "downloading $REPO into $DATA"
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO', repo_type='dataset', local_dir='$DATA')"
fi
cp -R "$HERE/libero" "$HERE/.sim_shims" "$HERE/sim_env.sh" "$DST/"
cp -R "$DATA/libero" "$DST/"
cp "$DATA/instructions.json" "$ROOT/instructions.json"
grep -q risk_objects "$DST/libero/libero/envs/objects/__init__.py" || git -C "$DST" apply "$HERE/objects_init.patch"
L="$DST/libero/libero"; mkdir -p "$DST/.libero_config"
cat > "$DST/.libero_config/config.yaml" <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $L/../datasets
assets: $L/assets
YAML
echo "LBJ-Bench installed into $DST"
