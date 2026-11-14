#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible server for rollouts or candidate evaluation (GPU host).
# Usage: scripts/gpu/serve_vllm.sh <model> <revision-sha> [lora-name=lora-path]
# Binds to 127.0.0.1 only; reach it through an SSH tunnel (ssh -L 8000:127.0.0.1:8000 <host>).
# --generation-config vllm stops the server from silently applying the model's generation_config.json;
# RLFactory sends explicit sampling parameters with every request.
set -euo pipefail
MODEL="${1:?model id}"
REVISION="${2:?pin a commit SHA}"
LORA="${3:-}"
ARGS=(--host 127.0.0.1 --port 8000 --revision "$REVISION" --generation-config vllm --gpu-memory-utilization 0.85)
if [[ -n "$LORA" ]]; then
  ARGS+=(--enable-lora --lora-modules "$LORA")
fi
exec vllm serve "$MODEL" "${ARGS[@]}"
