# Remote execution

Training runs on a GPU machine the user already controls. RLFactory **never** provisions, starts,
resizes or pays for infrastructure, and it stores no credentials.

```
Laptop                                                   GPU host (SSH)
  rlfactory train <cfg> --dry-run   -> run directory
  rlfactory remote check            ---- nvidia-smi ------------------------------>
  rlfactory remote launch           ---- preflight (GPU, pinned revision, memory)
                                    ---- rsync/scp code ------------------------->  code_dir
                                    ---- setup commands (uv sync --extra trl ...) >  code_dir
                                    ---- rsync/scp run directory --------------->  workspace/<run-id>
                                    ---- smoke step (TRL, 2 steps) ------------->
                                    ---- nohup bash launch.sh & ---------------->  training
  rlfactory remote status           ---- status.json / exit_status.json / log --->
  rlfactory remote fetch            <--- status, metrics, summary, adapter config  (no weights)
```

## Host configuration

`configs/remote/<name>.yaml` (only `*.example.yaml` is committed; real host files are git-ignored):

| Field | Meaning |
|---|---|
| `name` | label recorded in handles |
| `host` | hostname or, preferably, an alias from `~/.ssh/config` |
| `user`, `port` | optional; leave `user` empty when the SSH alias defines it |
| `workspace` | absolute directory for run directories |
| `code_dir` | absolute path of the RLFactory checkout on the host |
| `gpu` | expected GPU profile (`name`, `count`, `memory_gb`), informational |
| `environment.setup_commands` | run in `code_dir`, e.g. `uv sync --extra trl --extra gpu` |
| `environment.activate` | activates the training environment; runs from `code_dir` before the command changes into its run directory |
| `environment.env_vars` | non-secret variables only; secret-looking names are rejected |
| `connect_timeout_s` | SSH connect timeout |

SSH runs with `BatchMode=yes`, so it never prompts for passwords; keys come from the user's agent.

## Commands

| Command | Effect |
|---|---|
| `rlfactory remote check HOST [--dry-run]` | read-only GPU query |
| `rlfactory remote launch WORKDIR --host HOST [--no-sync-code] [--no-prepare-env] [--skip-smoke] [--yes] [--dry-run]` | confirmation, preflight, sync, environment setup, smoke, launch; writes `remote_handle.json` (never with `--dry-run`, so an earlier real handle is not overwritten) |
| `rlfactory remote status WORKDIR --host HOST` | reads `status.json`, then `exit_status.json` |
| `rlfactory remote fetch WORKDIR --host HOST` | downloads `status.json`, `metrics.jsonl`, `training_summary.json`, `exit_status.json`, `launch.out`, `outputs/final/rlfactory_checkpoint.json`, `outputs/final/adapter_config.json` |

`WORKDIR` is a prepared run directory containing `prepared.json` (written by `rlfactory train` and
by the experiment `train` stage). `remote launch` syncs code from the current directory, so run it
from the repository root. `--dry-run` uses a recording command runner and prints every `ssh`,
`rsync`/`scp` command instead of executing it.

The activation chain runs in the foreground and only the job itself is backgrounded
(`{ nohup bash launch.sh > launch.out 2>&1 < /dev/null & echo $!; }`), so no process keeps the
SSH channel open: the session ends as soon as the job's PID is printed, and the job survives the
logout. Backgrounding the whole `cd && activate && ...` chain instead keeps the channel open under
bash until training ends. Local `ssh`, `scp` and `rsync` processes always get `/dev/null` as
stdin.

## Preflight

`RemoteTrainingRunner.preflight()` fails when no GPU is visible or the model revision is not pinned,
and compares the memory estimate (when available) with 90% of the free memory of the smallest GPU.
The smoke step copies `run_config.json` with `max_steps=2`, runs the TRL entrypoint synchronously on
the GPU and aborts the launch if it fails.

## Experiments with remote training

When an experiment's `training.remote_host` is set (or the backend is verl/OpenRLHF), the `train`
stage prepares the run directory and stops with status `blocked` and instructions. After
`rlfactory remote launch`, `status` and `fetch`, `rlfactory experiment resume <run-id>` finds the
fetched `training_summary.json` with state `succeeded` and continues with evaluation, gating and
reporting. Paid compute is therefore never started as a side effect of `experiment run`.

## Known limitations

- Weights are not fetched; evaluate a remote checkpoint by serving it on the GPU host (vLLM with
  `--enable-lora`) and pointing `evaluation.candidate_policy` at that endpoint.
- There is no job scheduler integration (Slurm, Kubernetes) in v0.1.
