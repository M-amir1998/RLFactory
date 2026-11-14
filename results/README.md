# results/

Run outputs (`results/<run-id>/`) are written here and are **git-ignored**: trajectories, datasets,
checkpoints and evaluation stores can contain large or private data. Only this README is tracked.

```
results/<run-id>/
  manifest.json          run identity, code version, suites, components, hardware
  config.yaml            the exact experiment configuration
  stages/<stage>.json    per-stage status, fingerprint, outputs, resource usage
  tasks/                 train / validation / evaluation splits, contamination filter
  rollouts/              Parquet trajectory shards (DuckDB-queryable)
  rollout_summary.json   throughput, pass rate, failures by kind
  metrics.prom           Prometheus text exposition of rollout metrics
  datasets/<kind>/       records.jsonl (canonical), train.jsonl (framework format), dataset_card.json
  training/              prepared run directories and training_summary.json
  evaluation/            base.json, candidate.json, per-label failures and stores
  evaluation.json        baseline + candidate + gate + promotion decision
  failures.parquet       observed failure categories (+ separately labelled hypotheses)
  reward_analysis.json   reward calibration and reward-hacking flags
  flywheel.json          failure-driven flywheel iterations (when run)
  report.md, plots/      human-readable report and SVG figures
```

A small simulated example report is committed under `reports/cpu-e2e/`.
