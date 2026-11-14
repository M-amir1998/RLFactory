# ADR-008: Remote training architecture

- Status: Accepted
- Date: 2026-11-14

## Context

Training needs GPUs that the laptop does not have. GPU capacity may come from different providers
or personal machines. Spending money must never happen as a side effect, credentials must not end
up in the repository, and runs must be recoverable when connections drop.

## Decision

- Backends prepare **self-contained run directories** locally (config, data, `launch.sh`,
  `prepared.json`).
- `rlfactory remote` executes against an **existing** SSH host described by a credential-free YAML
  (`host` alias, workspace, code directory, setup commands, non-secret environment variables). SSH
  keys stay in the user's agent; `BatchMode=yes` prevents password prompts.
- The workflow is explicit and staged: confirm -> preflight (GPU visible, pinned revision, memory
  estimate) -> sync code -> prepare environment -> sync run -> smoke step -> background launch
  (`nohup`) -> status polling -> fetch of small result files. Weights stay on the host.
- Every command goes through a `CommandRunner`, so `--dry-run` prints the exact commands and tests
  never touch a network.
- Experiments never launch remote training implicitly: the `train` stage stops as `blocked` with
  instructions and continues on `experiment resume` once results were fetched.
- No cloud provisioning API is integrated.

## Consequences

- Provider-neutral; works with any machine reachable over SSH.
- No surprise spend; launches are deliberate and confirmable.
- Status depends on files written by the run (`status.json`, `exit_status.json`), which survive SSH
  disconnects.
- Manual steps remain for serving checkpoints for evaluation (vLLM with LoRA) and for scheduler
  integration (Slurm, Kubernetes), which are future work.

## Alternatives considered

- **Provider SDKs (start/stop instances)**: risks unintended spend and couples the project to one
  provider; rejected.
- **Ray cluster as the only execution layer**: required by OpenRLHF, but too heavy as the default
  for single-GPU TRL runs.
- **Copying weights back by default**: large, slow and risky on a 33 GB-free laptop disk; fetch
  manually when needed.
