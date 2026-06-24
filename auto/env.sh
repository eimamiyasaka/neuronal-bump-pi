# Source before running AUTO drivers:  source auto/env.sh
# Sets up auto-07p built at ~/auto-07p (see auto/build_auto07p.sh).
export AUTO_DIR="${HOME}/auto-07p"
export PATH="${AUTO_DIR}/cmds:${AUTO_DIR}/bin:${PATH}"
export PYTHONPATH="${AUTO_DIR}/python:${PYTHONPATH:-}"
