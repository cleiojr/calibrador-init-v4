# Calibrador Init V4

Research code for **Calibrador Init V4**, a tailored neural-network weight-initialization method that combines robust empirical search, symbolic regression and independent confirmatory evaluation against classical initializers.

The central objective is to discover a scenario-specific layerwise variance law

\[
\mathrm{Var}(W_l)=f(\mathrm{fan\_in},\mathrm{fan\_out},\mathrm{depth},\ldots)
\]

for a fixed combination of dataset, architecture, activation and training protocol.

## Method overview

The V4 pipeline uses:

1. robust search over layerwise variance profiles;
2. minimax comparison against He, Xavier and LeCun;
3. elite re-evaluation on independent selection seeds;
4. symbolic regression with PySR;
5. empirical selection of symbolic candidates;
6. independent final-seed evaluation with paired statistics.

The test set is excluded from discovery, elite selection, symbolic regression and formula selection. It is used only in the final evaluation stage.

> V4 provides an empirical validation protocol, not a universal mathematical guarantee of superiority.

## Repository layout

```text
calibrador-init-v4/
├── src/
│   └── tailored_symbolic_init_v4.py
├── benchmarks/
│   ├── benchmark_mnist_relu.py
│   ├── benchmark_fashion_mnist_silu.py
│   ├── benchmark_cifar10_gelu.py
│   └── benchmark_svhn_softplus.py
├── results/
│   ├── mnist_relu/
│   ├── fashion_mnist_silu/
│   ├── cifar10_gelu/
│   └── svhn_softplus/
├── docs/
│   └── REPRODUCIBILITY.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10+ and a CUDA-capable PyTorch environment are recommended.

```bash
pip install -r requirements.txt
```

PySR also requires a working Julia/PySR installation; follow the PySR installation instructions for the target environment.

## Running the V4 discovery pipeline

Example for MNIST + ReLU:

```bash
python src/tailored_symbolic_init_v4.py \
  --dataset mnist \
  --layers 32 64 128 32 10 \
  --activation relu
```

Example for Fashion-MNIST + SiLU:

```bash
python src/tailored_symbolic_init_v4.py \
  --dataset fashion_mnist \
  --layers 256 64 512 32 256 64 10 \
  --activation silu
```

### Important implementation note

The frozen `src/tailored_symbolic_init_v4.py` preserved in this repository is the research implementation used during development and currently exposes MNIST/Fashion-MNIST in its built-in dataset table. The independent CIFAR-10 and SVHN confirmatory scripts are kept under `benchmarks/`. This repository intentionally does not silently modify the frozen V4 implementation.

## Independent confirmatory benchmarks

### MNIST + ReLU

```bash
python benchmarks/benchmark_mnist_relu.py
```

Architecture:

```text
784 -> 32 -> 64 -> 128 -> 32 -> 10
```

The benchmark compares He, Xavier, LeCun, **V4-Symbolic** and **V4-Refined**.

Confirmatory mean validation Loss AULC:

| Method | Loss AULC ↓ |
|---|---:|
| V4-Symbolic | **0.142586** |
| Xavier | 0.141705 |
| LeCun | 0.144152 |
| V4-Refined | 0.146639 |
| He | 0.161479 |

V4-Symbolic beat He in 10/10 paired seeds for Loss AULC, with statistical significance after Holm correction. Against Xavier and LeCun, the symbolic variant was statistically competitive rather than consistently superior.

### Fashion-MNIST + SiLU

```bash
python benchmarks/benchmark_fashion_mnist_silu.py
```

Architecture:

```text
784 -> 256 -> 64 -> 512 -> 32 -> 256 -> 64 -> 10
```

Confirmatory mean validation Loss AULC:

| Method | Loss AULC ↓ |
|---|---:|
| Xavier | **0.365873** |
| V4-Refined | 0.367627 |
| V4-Symbolic | 0.367802 |
| LeCun | 0.367916 |
| He | 0.381363 |

Both V4 variants beat He in all 10 paired seeds for Loss AULC and Accuracy AULC, with statistically significant differences after Holm correction. Xavier and LeCun remained competitive in final performance.

### CIFAR-10 + GELU

```bash
python benchmarks/benchmark_cifar10_gelu.py
```

Architecture:

```text
3072 -> 1024 -> 256 -> 512 -> 128 -> 256 -> 64 -> 10
```

For this scenario the final V4 origin was `pysr`, so the final solution is **V4-Symbolic**; there is no separate refined-vector ablation.

Mean validation Loss AULC among completed runs:

| Method | Completed | Loss AULC ↓ |
|---|---:|---:|
| V4-Symbolic | 10/10 | **1.746053** |
| LeCun | 9/10 | 1.748814 |
| He | 10/10 | 1.756767 |
| Xavier | 9/10 | 1.786650 |

The V4 signal is strongest in convergence, particularly relative to Xavier, but the confirmatory paired tests do not establish general statistical superiority. Numerical failures are reported separately from completed-run performance.

### SVHN + Softplus

```bash
python benchmarks/benchmark_svhn_softplus.py
```

Architecture:

```text
3072 -> 512 -> 128 -> 512 -> 64 -> 256 -> 10
```

The independent confirmatory run is still pending; no result is claimed here yet.

## Interpreting V4-Symbolic and V4-Refined

- **V4-Symbolic**: the variance profile produced directly by the symbolic expression discovered with PySR.
- **V4-Refined**: a final layerwise variance vector obtained after numerical refinement when the original V4 run ended with `origin = formula_mutation`.

When `origin = pysr`, the symbolic formula itself is the final V4 solution.

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the confirmatory evaluation protocol and treatment of numerical failures.

Raw per-seed runs, histories, summaries, paired statistics and plots are included in `results/` for completed scenarios.

## Citation

A `CITATION.cff` file is provided. Replace `SEU_USUARIO` with the GitHub account name after creating the repository and update the citation with the dissertation/article DOI when available.

## License

MIT License.

## Derived confirmatory analyses

The `results/` tree includes the raw outputs from the completed confirmatory benchmarks and automatically generated derived analyses. Rebuild the derived tables and figures with:

```bash
python analysis/analyze_results.py
```

Per-scenario `derived/` folders contain compact descriptive summaries, metric rankings and V4-vs-classical primary evidence. `results/aggregate/` contains cross-scenario summaries based on within-scenario ranks and paired evidence. Raw loss values from different datasets are not averaged together.
