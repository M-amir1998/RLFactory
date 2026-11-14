#!/usr/bin/env bash
# Run on the GPU host inside the RLFactory checkout. Creates .venv with the TRL stack (+ bitsandbytes on Linux).
# Never run in CI; never needed on a CPU-only laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --frozen --extra trl --extra gpu
uv run python - <<'PY'
import importlib.metadata as md
import torch
for pkg in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
    print(f"{pkg}=={md.version(pkg)}")
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY
