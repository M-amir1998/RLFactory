# Security policy

RLFactory executes code written by language models and handles model checkpoints, datasets and
remote machines. This document states what the project does about each risk and, just as
importantly, what it does not guarantee.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/M-amir1998/RLFactory/security/advisories/new)
for this repository. Do not open a public issue for an unfixed vulnerability. Include the affected
version or commit, a reproduction, and the impact you observed.

## Threat model summary

| Asset / input | Risk | Mitigation in RLFactory | Residual risk |
|---|---|---|---|
| Model-generated code | arbitrary code execution | never executed in-process; sandboxed (see below) | subprocess sandbox is not a boundary |
| Model outputs (all domains) | parser abuse, prompt injection into tools | strict parsers, no shell/network/file tools, typed tool arguments | tools see attacker-chosen arguments |
| Sandbox escape | host compromise | Docker: no network, non-root, read-only rootfs, dropped caps, cgroup limits | shared kernel; container runtime bugs |
| Datasets / task files | malicious content, unsafe deserialisation | JSON/JSONL and Parquet only, pydantic validation, `yaml.safe_load` | content is still untrusted text |
| Checkpoints | pickle code execution, remote code | safetensors required, `trust_remote_code=False`, pinned revisions | a trusted repo can still be compromised |
| Secrets | leakage via configs, logs, artefacts | env-var *names* only in configs, log redaction, secret scan, ignore rules | secrets already in your environment |
| Remote GPU hosts | credential exposure, unintended spend | no stored credentials, SSH agent/config, explicit confirmation, no provisioning | whatever access your SSH key grants |

## Generated code and sandbox escape

Generated code only runs through a `Sandbox` implementation, never in the RLFactory process.

- **`SubprocessSandbox` is not a security boundary.** It isolates *accidents* (infinite loops,
  memory blow-ups, output floods) by running a separate interpreter (`-S -P -B -X utf8`) in a
  temporary directory with a minimal, secret-free environment, plus time, memory, output and
  file-size limits. The program still runs as your user, can read your files and can use the
  network.
- **`DockerSandbox` is stronger but limited.** Each execution gets a fresh container with
  `--network none`, a numeric non-root user, a read-only root filesystem, a `noexec` tmpfs,
  `--cap-drop ALL`, `no-new-privileges`, memory/CPU/PID/file-size limits and a read-only bind mount
  of the work directory. The Docker socket is never mounted. Containers share the host kernel.
- For untrusted or adversarially fine-tuned models, use the Docker sandbox on a disposable machine.

Details, measured behaviours and residual risks: [docs/security/sandbox.md](docs/security/sandbox.md).

## Untrusted model outputs

Model outputs are treated as untrusted data everywhere:

- answers are extracted with bounded regular expressions and parsed with `json.loads` or the
  restricted arithmetic evaluator in `rlfactory.domain.numeric` (an AST whitelist over `Fraction`,
  bounded exponents, magnitudes and expression size; no `eval`);
- tool calls are JSON objects validated against a small registry (`calculator`, `kv_lookup`,
  `json_transform`); there is no shell, network or filesystem tool;
- the Python unit-test harness never exposes expected outputs or its report nonce to the process
  that runs generated code (see the sandbox document).

## Untrusted datasets and task files

Task suites, datasets and trajectories are read as JSON/JSONL or Parquet and validated by strict
pydantic models (`extra="forbid"`). YAML configuration is parsed with `yaml.safe_load` only. Nothing
in RLFactory unpickles data. Task files from third parties can still contain hostile prompts;
review them like any other untrusted input.

## Secrets

- Configs reference secrets by environment-variable **name** (for example `api_key_env:
  VLLM_API_KEY`); values are read at runtime and never written to disk by RLFactory.
- Remote host configs reject environment variables whose names look like secrets (`TOKEN`,
  `SECRET`, `PASSWORD`, `API_KEY`, `ACCESS_KEY`, `PRIVATE`).
- Logs pass through a redaction filter for Hugging Face, OpenAI-style, AWS, GitHub and Slack tokens,
  bearer headers and private-key blocks. Redaction is a safety net, not a licence to log secrets.
- `.gitignore` excludes `.env*`, keys, certificates, credentials, weights and `results/`;
  `scripts/secret_scan.py` runs in CI and before every push.

## Remote GPU credentials and spend

RLFactory never creates, starts or pays for cloud resources. Remote execution uses `ssh`, `scp` and
`rsync` with `BatchMode=yes` against a host alias from your own `~/.ssh/config`; private keys stay in
your SSH agent and are never configured in RLFactory. `rlfactory remote launch` asks for explicit
confirmation (unless `--yes`), and `--dry-run` prints every command without executing it. Only
small result files are fetched back; weights stay on the GPU host.

## Checkpoint trust

- `trust_remote_code` defaults to `False` everywhere (`ModelReference`, TRL `model_init_kwargs`,
  Transformers loading). Enabling it is an explicit, recorded choice.
- `TransformersPolicy` loads weights with `use_safetensors=True` and `local_files_only=True`, so it
  neither unpickles weights nor downloads anything implicitly.
- Remote training refuses to start until the model revision is pinned to a commit SHA
  (preflight check), so a mutable branch cannot change underneath a run.
- `CheckpointGuard` refuses unplanned downloads: it reports the model, expected size, free disk and
  cache path, and allows at most one uncached checkpoint at a time, only with explicit permission.

## Unsafe serialisation

No `pickle`, `marshal`, `torch.load` of untrusted files or `yaml.load` with arbitrary tags is used in
the RLFactory code base. Trajectories and datasets are JSON-based; checkpoints are expected in
safetensors format.

## Supported versions

Only the latest commit on `main` is supported during the `0.1.0.dev` phase.
