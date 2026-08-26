# Reproducibility protocol

The confirmatory benchmarks use independent seeds and do not tune the initializer during evaluation.

## Common confirmatory protocol

- 10 paired seeds: 1001–1010
- 30 epochs
- batch size: 256
- SGD, learning rate 0.05, momentum 0.9, weight decay 0
- fixed train/validation split
- test set evaluated only after training
- primary metrics: validation Loss AULC and final validation loss
- secondary metrics: validation Accuracy AULC, final validation accuracy, test loss and test accuracy
- paired statistics include bootstrap confidence intervals, paired t-test, Wilcoxon test and Holm correction where available

## Numerical failures

A run that terminates early because of a non-finite loss is not treated as a complete 30-epoch performance observation. Such failures are reported separately as stability outcomes. Completed runs that collapse to chance-level accuracy remain valid empirical observations.

## Scenario status

- MNIST + ReLU: confirmatory results included.
- Fashion-MNIST + SiLU: confirmatory results included.
- CIFAR-10 + GELU: confirmatory results included; the final accounting separates numerical failure from completed-run performance.
- SVHN + Softplus: benchmark script included; confirmatory result directory is intentionally empty until the independent run is completed.
