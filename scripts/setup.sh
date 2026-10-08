#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pinn_venv="${PINN_VENV:-$project_root/.venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$project_root/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m venv "$pinn_venv"
"$pinn_venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
"$pinn_venv/bin/python" -m pip install --index-url https://pypi.org/simple -r "$project_root/requirements-cpu.lock"
"$pinn_venv/bin/python" -m pip check
"$pinn_venv/bin/python" -c 'import torch, numpy, matplotlib, nbclient, nbconvert; print("Ready:", torch.__version__, "CUDA:", torch.cuda.is_available())'
