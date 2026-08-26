# Aggregate confirmatory results

Generated automatically from the completed scenario result folders.

Completed scenarios: MNIST+ReLU, Fashion-MNIST+SiLU, CIFAR-10+GELU.
SVHN+Softplus is intentionally excluded until its confirmatory benchmark is available.

## Files

- `all_scenarios_compact_summary.csv`: descriptive statistics by scenario/method.
- `v4_vs_classical_primary_evidence.csv`: paired primary-metric evidence for V4 vs He/Xavier/LeCun.
- `average_rank_across_scenarios.csv`: within-scenario ranks aggregated across scenarios.
- `loss_aulc_evidence_counts.csv`: counts of positive, high-win-rate, CI-positive and significant comparisons.
- `figures/`: cross-scenario visual summaries.

Important: cross-scenario ranks summarize relative position only; raw loss values are not averaged across different datasets.
