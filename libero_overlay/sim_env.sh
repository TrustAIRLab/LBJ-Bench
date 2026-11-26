# Source in every new shell: sets PYTHONPATH, LIBERO_CONFIG_PATH and $SIMPY.
R="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$R/.sim_shims:$R"
export LIBERO_CONFIG_PATH="$R/.libero_config"
export SIMPY="$R/.venv-sim/bin/python"
