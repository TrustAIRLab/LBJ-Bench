# Source before running local scene validation: sets the LIBERO-Safety sim env (Python 3.10, see .venv-sim).
R="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$R/.sim_shims:$R"
export LIBERO_CONFIG_PATH="$R/.libero_config"
export SIMPY="$R/.venv-sim/bin/python"
