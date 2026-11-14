#!/usr/bin/env bash
# Inspect GPUs before any run: name, total/used memory, driver. Read-only.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
