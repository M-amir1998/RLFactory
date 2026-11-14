#!/usr/bin/env bash
# verl lives in its OWN environment (it pins torch/vLLM/transformers differently from TRL).
# Preferred: verl's published container images. Fallback below: an isolated venv next to the checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv .venv-verl --python 3.12
uv pip install --python .venv-verl/bin/python "verl==0.9.1"
uv pip install --python .venv-verl/bin/python --no-deps -e .   # RLFactory for the reward shim
.venv-verl/bin/python -c "import verl, importlib.metadata as m; print('verl', m.version('verl'))"
echo "Dry-run a prepared run first: (cd <run_dir> && .venv-verl/bin/python -m verl.trainer.main_ppo ... --cfg job)"
