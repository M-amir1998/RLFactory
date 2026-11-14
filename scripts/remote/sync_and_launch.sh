#!/usr/bin/env bash
# Manual, transparent equivalent of `rlfactory remote launch` (same steps, no smoke test).
# Usage: scripts/remote/sync_and_launch.sh <ssh-host-alias> <local-run-dir> <remote-workspace> <remote-code-dir>
# Uses your SSH agent/config; never stores credentials. Does not provision or pay for anything.
set -euo pipefail
HOST="${1:?ssh host alias}"; RUN_DIR="${2:?prepared run dir}"; WORKSPACE="${3:?remote workspace}"; CODE="${4:?remote code dir}"
RUN_ID="$(basename "$RUN_DIR")"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$WORKSPACE/$RUN_ID' '$CODE'"
rsync -az --delete --exclude .git --exclude '.venv*' --exclude results ./ "$HOST:$CODE/"
rsync -az "$RUN_DIR/" "$HOST:$WORKSPACE/$RUN_ID/"
ssh -o BatchMode=yes "$HOST" "cd '$CODE' && source .venv/bin/activate && cd '$WORKSPACE/$RUN_ID' && nohup bash launch.sh > launch.out 2>&1 & echo \$!"
echo "Monitor: ssh $HOST 'cat $WORKSPACE/$RUN_ID/status.json; tail -n 20 $WORKSPACE/$RUN_ID/train.log'"
