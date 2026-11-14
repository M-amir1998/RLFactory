# ADR-003: TRL is the initial training backend

- Status: Accepted
- Date: 2026-11-14

## Context

The project needs SFT, DPO and GRPO (plus optionally reward modelling) on small open models
(about 0.5B-3B parameters) on a single rented or borrowed GPU. Candidates were Hugging Face TRL,
verl and OpenRLHF. Their current versions (verified 2026-11-14) pin mutually incompatible stacks:
verl 0.9.1 pins transformers 5.9.0 and supports Python < 3.13; OpenRLHF 0.11.2 installs a specific
vLLM; both depend on flash-attn, which has no PyPI wheels and breaks universal locking.

## Decision

Use TRL (1.18) as the primary backend, installed through the `trl` extra. Keep RLFactory's core
independent of any framework behind the `TrainingBackend` protocol. Prepare verl and OpenRLHF
backends that write correct run directories, but install those frameworks only in isolated GPU-host
environments; they are not RLFactory extras.

## Consequences

- One mature, well-documented stack covers SFT, DPO, GRPO and reward models, LoRA/QLoRA and vLLM
  generation, with a small-model workflow.
- The adapter must follow TRL's fast-moving API; field names are checked against the installed
  dataclasses by a test whenever the `trl` extra is present, and removed keys are rejected.
- Large-scale or highly optimised RL (multi-node, Megatron, advanced rollout scheduling) is deferred
  to verl/OpenRLHF on GPU hosts.

## Alternatives considered

- **verl as primary**: strongest for scale, but heavy, Ray/Hydra-centric and Linux/GPU-bound;
  impossible to exercise locally.
- **OpenRLHF as primary**: similar trade-offs; its CLI changed to dotted flags without aliases in
  0.11, illustrating the maintenance risk.
- **All three in one environment**: rejected because of conflicting pins.
