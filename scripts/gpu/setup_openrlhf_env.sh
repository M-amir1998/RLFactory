#!/usr/bin/env bash
# OpenRLHF lives in its OWN environment (Ray + DeepSpeed + pinned vLLM).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv .venv-openrlhf --python 3.12
uv pip install --python .venv-openrlhf/bin/python "openrlhf[vllm]==0.11.2"
uv pip install --python .venv-openrlhf/bin/python --no-deps -e .   # RLFactory for the reward shim
.venv-openrlhf/bin/python -m openrlhf.cli.train_ppo_ray --help | head -n 20
echo "Start Ray first (ray start --head), then run the prepared launch.sh."
