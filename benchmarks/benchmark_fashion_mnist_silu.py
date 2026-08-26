#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark confirmatório — Calibrador Init V4

Cenário: Fashion-MNIST + SiLU
Arquitetura: 784 -> 256 -> 64 -> 512 -> 32 -> 256 -> 64 -> 10
Ativação: SiLU

Métodos:
  - He
  - Xavier
  - LeCun
  - Calibrador Init V4-Symbolic
  - Calibrador Init V4-Refined

Protocolo:
  - 10 seeds pareadas
  - 30 épocas
  - SGD, lr=0.05, momentum=0.9, weight_decay=0
  - mesmo split treino/validação para todos
  - mesmo embaralhamento por seed para todos
  - test set usado apenas ao final
  - fórmula e vetor congelados; nenhuma busca/ajuste neste benchmark

Métricas:
  final_val_loss, final_val_acc,
  val_loss_aulc, val_acc_aulc,
  test_loss, test_acc,
  grad_norm_mean, training_time_sec.

Estatística:
  média, desvio-padrão, mediana,
  melhoria média/mediana,
  wins/losses/ties, win-rate,
  bootstrap IC95%,
  t pareado, Wilcoxon,
  Cohen dz, rank-biserial,
  correção de Holm.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from scipy import stats


SCENARIO_NAME = 'Fashion-MNIST + SiLU'
INPUT_DIM = 784
LAYERS = (256, 64, 512, 32, 256, 64, 10)
ACTIVATION_NAME = 'silu'

# Exatamente 10 seeds novas, distintas das usadas na V4.
DEFAULT_SEEDS = tuple(range(1001, 1011))
DEFAULT_EPOCHS = 30

V4_REFINED_VECTOR = np.asarray([8.4e-05, 0.00075296, 0.00873736, 0.0091705, 0.0529918, 0.0567118, 0.0350022], dtype=np.float64)
FORMULA_TEXT = 'square(log_fan_out * (square(depth) * -0.08186779))'


def v4_symbolic_variance_vector():
    dims = [INPUT_DIM] + list(LAYERS)
    L = len(LAYERS)
    out = []

    for l in range(L):
        fan_in = float(dims[l])
        fan_out = float(dims[l + 1])
        depth = float(l + 1) / float(L)
        log_fan_in = math.log(fan_in)
        log_fan_out = math.log(fan_out)

        var = (log_fan_out * ((depth ** 2) * -0.08186779)) ** 2

        if not math.isfinite(var) or var <= 0:
            raise ValueError(
                f"Fórmula gerou variância inválida na camada {l}: {var}"
            )
        out.append(float(var))

    return np.asarray(out, dtype=np.float64)


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def make_activation():
    if ACTIVATION_NAME == "silu":
        return nn.SiLU()
    if ACTIVATION_NAME == "gelu":
        return nn.GELU()
    raise ValueError(ACTIVATION_NAME)


class FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [INPUT_DIM] + list(LAYERS)
        modules = []

        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(make_activation())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(torch.flatten(x, 1))

    def linears(self):
        return [m for m in self.modules() if isinstance(m, nn.Linear)]


@torch.no_grad()
def init_variance_vector(model: FixedMLP, variances):
    layers = model.linears()
    variances = np.asarray(variances, dtype=float)

    if len(layers) != len(variances):
        raise ValueError(
            f"Modelo possui {len(layers)} camadas, vetor possui {len(variances)}."
        )

    for idx, (layer, var) in enumerate(zip(layers, variances)):
        var = float(var)

        if not math.isfinite(var) or var <= 0:
            raise ValueError(f"Variância inválida na camada {idx}: {var}")

        nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(var))

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


@torch.no_grad()
def init_classic(model: FixedMLP, method: str):
    for layer in model.linears():
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)

        if method == "lecun":
            var = 1.0 / fan_in
        elif method == "he":
            var = 2.0 / fan_in
        elif method == "xavier":
            var = 2.0 / (fan_in + fan_out)
        else:
            raise ValueError(method)

        nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(var))

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def load_data(data_dir: str, val_fraction: float):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.2860406,),
            (0.35302424,),
        ),
    ])

    full_train = datasets.FashionMNIST(
        data_dir,
        train=True,
        download=True,
        transform=tfm,
    )

    test_ds = datasets.FashionMNIST(
        data_dir,
        train=False,
        download=True,
        transform=tfm,
    )

    n_val = int(round(len(full_train) * val_fraction))
    n_train = len(full_train) - n_val

    # Split confirmatório novo e fixo, independente da busca da V4.
    split_generator = torch.Generator().manual_seed(20260823)

    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=split_generator,
    )

    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, seed, device):
    g = torch.Generator().manual_seed(seed)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=g,
        **kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


def grad_norm(model):
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())

    return math.sqrt(max(total, 0.0))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total = 0
    loss_sum = 0.0
    correct = 0
    grad_sum = 0.0
    grad_steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError("Loss não finita.")

        loss.backward()

        gn = grad_norm(model)
        if not math.isfinite(gn):
            raise FloatingPointError("Norma do gradiente não finita.")

        optimizer.step()

        bs = y.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(1) == y).sum().item())

        grad_sum += gn
        grad_steps += 1

    return {
        "train_loss": loss_sum / total,
        "train_acc": correct / total,
        "grad_norm": grad_sum / max(grad_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total = 0
    loss_sum = 0.0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs
        correct += int((logits.argmax(1) == y).sum().item())

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
    }


def aulc(values):
    y = np.asarray(values, dtype=float)

    if len(y) == 1:
        return float(y[0])

    x = np.linspace(0.0, 1.0, len(y))
    return float(np.trapz(y, x))


def initialize_method(model, method):
    if method in ("he", "xavier", "lecun"):
        init_classic(model, method)
        return

    if method == "calibrador_v4_symbolic":
        init_variance_vector(model, v4_symbolic_variance_vector())
        return

    if method == "calibrador_v4_refined":
        init_variance_vector(model, V4_REFINED_VECTOR)
        return

    raise ValueError(method)


def run_one(method, seed, epochs, train_ds, val_ds, test_ds, args, device):
    seed_everything(seed, args.deterministic)

    train_loader, val_loader, test_loader = make_loaders(
        train_ds,
        val_ds,
        test_ds,
        args.batch_size,
        args.num_workers,
        seed,
        device,
    )

    model = FixedMLP().to(device)
    initialize_method(model, method)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    val_losses = []
    val_accs = []
    grad_norms = []
    histories = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)

        val_losses.append(va["loss"])
        val_accs.append(va["acc"])
        grad_norms.append(tr["grad_norm"])

        histories.append({
            "scenario": SCENARIO_NAME,
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "train_loss": tr["train_loss"],
            "train_acc": tr["train_acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "grad_norm": tr["grad_norm"],
        })

    te = evaluate(model, test_loader, criterion, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    result = {
        "scenario": SCENARIO_NAME,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "final_val_loss": val_losses[-1],
        "final_val_acc": val_accs[-1],
        "val_loss_aulc": aulc(val_losses),
        "val_acc_aulc": aulc(val_accs),
        "test_loss": te["loss"],
        "test_acc": te["acc"],
        "grad_norm_mean": float(np.mean(grad_norms)),
        "training_time_sec": float(elapsed),
    }

    del model, optimizer, criterion
    del train_loader, val_loader, test_loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, histories


LOWER_IS_BETTER = {
    "final_val_loss": True,
    "final_val_acc": False,
    "val_loss_aulc": True,
    "val_acc_aulc": False,
    "test_loss": True,
    "test_acc": False,
    "grad_norm_mean": None,
    "training_time_sec": True,
}

PERFORMANCE_METRICS = [
    "final_val_loss",
    "final_val_acc",
    "val_loss_aulc",
    "val_acc_aulc",
    "test_loss",
    "test_acc",
    "grad_norm_mean",
    "training_time_sec",
]

PRIMARY_METRICS = [
    "val_loss_aulc",
    "final_val_loss",
]

BASELINES = [
    "he",
    "xavier",
    "lecun",
]

V4_METHODS = [
    "calibrador_v4_symbolic",
    "calibrador_v4_refined",
]


def bootstrap_mean_ci(x, n_boot=10000, seed=20260823):
    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        means[i] = np.mean(
            rng.choice(x, size=len(x), replace=True)
        )

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()

    denom = r_plus + r_minus
    return float((r_plus - r_minus) / denom) if denom else 0.0


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running = 0.0

    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        adjusted_sorted[rank] = min(running, 1.0)

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted


def build_summary(df):
    rows = []

    for method, g in df.groupby("method"):
        row = {
            "method": method,
            "n": len(g),
        }

        for metric in PERFORMANCE_METRICS:
            vals = pd.to_numeric(
                g[metric],
                errors="coerce",
            ).to_numpy(dtype=float)

            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1))
            row[f"{metric}_median"] = float(np.nanmedian(vals))

        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_statistics(df, n_boot):
    rows = []

    comparisons = []

    for v4_method in V4_METHODS:
        for baseline in BASELINES:
            comparisons.append(
                (v4_method, baseline)
            )

    # Ablação direta: vetor refinado vs fórmula simbólica.
    comparisons.append(
        (
            "calibrador_v4_refined",
            "calibrador_v4_symbolic",
        )
    )

    for focal_method, baseline_method in comparisons:
        focal = (
            df[df.method == focal_method]
            .set_index("seed")
            .sort_index()
        )

        base = (
            df[df.method == baseline_method]
            .set_index("seed")
            .sort_index()
        )

        common = focal.index.intersection(base.index)

        for metric in PERFORMANCE_METRICS:
            x = focal.loc[common, metric].to_numpy(dtype=float)
            y = base.loc[common, metric].to_numpy(dtype=float)

            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]

            if len(x) < 2:
                continue

            lower = LOWER_IS_BETTER[metric]

            if lower is True:
                # positivo = focal melhor
                improvement = y - x

            elif lower is False:
                # positivo = focal melhor
                improvement = x - y

            else:
                # métrica apenas diagnóstica
                improvement = x - y

            mean_imp = float(np.mean(improvement))
            median_imp = float(np.median(improvement))

            ci_lo, ci_hi = bootstrap_mean_ci(
                improvement,
                n_boot=n_boot,
                seed=20260823
                + abs(hash((focal_method, baseline_method, metric))) % 100000,
            )

            if lower is None:
                wins = np.nan
                losses = np.nan
                ties = np.nan
                win_rate = np.nan
            else:
                eps = 1e-12

                wins = int(
                    np.sum(improvement > eps)
                )

                losses = int(
                    np.sum(improvement < -eps)
                )

                ties = int(
                    len(improvement)
                    - wins
                    - losses
                )

                win_rate = float(
                    wins / len(improvement)
                )

            try:
                t_res = stats.ttest_rel(
                    x,
                    y,
                    nan_policy="omit",
                )
                t_stat = float(t_res.statistic)
                t_p = float(t_res.pvalue)

            except Exception:
                t_stat = np.nan
                t_p = np.nan

            try:
                if np.allclose(x, y):
                    w_stat = 0.0
                    w_p = 1.0
                else:
                    w_res = stats.wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    w_stat = float(w_res.statistic)
                    w_p = float(w_res.pvalue)

            except Exception:
                w_stat = np.nan
                w_p = np.nan

            d = x - y
            sd = float(np.std(d, ddof=1))

            cohen_dz = (
                float(np.mean(d) / sd)
                if sd > 0
                else 0.0
            )

            rb = rank_biserial(improvement)

            rows.append({
                "focal_method": focal_method,
                "baseline_method": baseline_method,
                "comparison":
                    f"{focal_method}_vs_{baseline_method}",
                "metric": metric,
                "is_primary_metric":
                    metric in PRIMARY_METRICS,
                "n_pairs": len(x),
                "focal_mean": float(np.mean(x)),
                "baseline_mean": float(np.mean(y)),
                "mean_improvement": mean_imp,
                "median_improvement": median_imp,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": win_rate,
                "paired_t_stat": t_stat,
                "paired_t_p_raw": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p_raw": w_p,
                "cohen_dz_raw_x_minus_y": cohen_dz,
                "rank_biserial_improvement": rb,
            })

    stats_df = pd.DataFrame(rows)

    stats_df["paired_t_p_holm"] = np.nan
    stats_df["wilcoxon_p_holm"] = np.nan

    # Correção de Holm por família:
    # mesma métrica + mesma variante V4 contra He/Xavier/LeCun.
    for v4_method in V4_METHODS:
        for metric in PERFORMANCE_METRICS:

            mask = (
                (stats_df["focal_method"] == v4_method)
                & (stats_df["baseline_method"].isin(BASELINES))
                & (stats_df["metric"] == metric)
            )

            idx = stats_df.index[mask].tolist()

            if not idx:
                continue

            p_t = (
                stats_df.loc[idx, "paired_t_p_raw"]
                .to_numpy(dtype=float)
            )

            p_w = (
                stats_df.loc[idx, "wilcoxon_p_raw"]
                .to_numpy(dtype=float)
            )

            finite_t = np.isfinite(p_t)
            finite_w = np.isfinite(p_w)

            if finite_t.any():
                adj_t = holm_adjust(
                    p_t[finite_t]
                )

                stats_df.loc[
                    np.asarray(idx)[finite_t],
                    "paired_t_p_holm",
                ] = adj_t

            if finite_w.any():
                adj_w = holm_adjust(
                    p_w[finite_w]
                )

                stats_df.loc[
                    np.asarray(idx)[finite_w],
                    "wilcoxon_p_holm",
                ] = adj_w

    return stats_df


DISPLAY_NAMES = {
    "he": "He",
    "xavier": "Xavier",
    "lecun": "LeCun",
    "calibrador_v4_symbolic":
        "Calibrador Init V4-Symbolic",
    "calibrador_v4_refined":
        "Calibrador Init V4-Refined",
}


def save_curves(hist_df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, g in hist_df.groupby("method"):
        curve = (
            g.groupby("epoch")["val_loss"]
            .mean()
        )

        ax.plot(
            curve.index,
            curve.values,
            label=DISPLAY_NAMES.get(method, method),
        )

    ax.set_xlabel("Época")
    ax.set_ylabel("Validation loss")
    ax.set_title(
        f"{SCENARIO_NAME} — Validation Loss"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        out_dir / "loss_curves.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for method, g in hist_df.groupby("method"):
        curve = (
            g.groupby("epoch")["val_acc"]
            .mean()
        )

        ax.plot(
            curve.index,
            curve.values,
            label=DISPLAY_NAMES.get(method, method),
        )

    ax.set_xlabel("Época")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(
        f"{SCENARIO_NAME} — Validation Accuracy"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        out_dir / "accuracy_curves.png",
        dpi=160,
    )
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="./data",
    )

    p.add_argument(
        "--output-dir",
        default='benchmark_calibrador_v4_fashion_silu_results',
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.epochs != 30:
        print(
            "ATENÇÃO: protocolo planejado usa 30 épocas; "
            f"recebido --epochs={args.epochs}."
        )

    seeds = DEFAULT_SEEDS

    if len(seeds) != 10:
        raise RuntimeError(
            "Este benchmark foi fixado para exatamente 10 seeds."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbolic_vector = (
        v4_symbolic_variance_vector()
    )

    print("=" * 100)
    print(
        "BENCHMARK CONFIRMATÓRIO — CALIBRADOR INIT V4"
    )
    print("=" * 100)
    print("Cenário:", SCENARIO_NAME)
    print("Device:", device)
    print(
        "Arquitetura:",
        [INPUT_DIM] + list(LAYERS),
    )
    print("Ativação:", ACTIVATION_NAME)
    print("Épocas:", args.epochs)
    print("Seeds:", seeds)
    print("Fórmula:", FORMULA_TEXT)
    print(
        "Vetor Symbolic:",
        symbolic_vector.tolist(),
    )
    print(
        "Vetor Refined:",
        V4_REFINED_VECTOR.tolist(),
    )
    print("=" * 100)

    train_ds, val_ds, test_ds = load_data(
        args.data_dir,
        args.val_fraction,
    )

    methods = [
        "he",
        "xavier",
        "lecun",
        "calibrador_v4_symbolic",
        "calibrador_v4_refined",
    ]

    all_runs = []
    all_histories = []

    for seed in seeds:
        for method in methods:

            print(
                f"\n[seed={seed}] "
                f"{DISPLAY_NAMES.get(method, method)}"
            )

            result, hist = run_one(
                method=method,
                seed=seed,
                epochs=args.epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                args=args,
                device=device,
            )

            all_runs.append(result)
            all_histories.extend(hist)

            print(
                f"  val_loss={result['final_val_loss']:.6f} | "
                f"val_acc={result['final_val_acc']:.4f} | "
                f"AULC_loss={result['val_loss_aulc']:.6f} | "
                f"test_loss={result['test_loss']:.6f} | "
                f"test_acc={result['test_acc']:.4f}"
            )

            pd.DataFrame(
                all_runs
            ).to_csv(
                out_dir / "runs.csv",
                index=False,
            )

            pd.DataFrame(
                all_histories
            ).to_csv(
                out_dir / "histories.csv",
                index=False,
            )

    runs_df = pd.DataFrame(all_runs)
    hist_df = pd.DataFrame(all_histories)

    summary_df = build_summary(runs_df)

    stats_df = pairwise_statistics(
        runs_df,
        args.bootstrap_samples,
    )

    summary_df.to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    stats_df.to_csv(
        out_dir / "pairwise_statistics.csv",
        index=False,
    )

    holm_df = stats_df[
        stats_df["is_primary_metric"]
        & stats_df["baseline_method"].isin(BASELINES)
    ].copy()

    holm_df.to_csv(
        out_dir / "holm_statistics.csv",
        index=False,
    )

    definitions = {
        "scenario": SCENARIO_NAME,
        "architecture":
            [INPUT_DIM] + list(LAYERS),
        "activation": ACTIVATION_NAME,
        "seeds": list(seeds),
        "epochs": args.epochs,
        "primary_metrics": PRIMARY_METRICS,
        "secondary_metrics": [
            "final_val_acc",
            "val_acc_aulc",
            "test_loss",
            "test_acc",
        ],
        "diagnostic_metrics": [
            "grad_norm_mean",
            "training_time_sec",
        ],
        "formula_text": FORMULA_TEXT,
        "symbolic_variance_vector":
            symbolic_vector.tolist(),
        "refined_variance_vector":
            V4_REFINED_VECTOR.tolist(),
        "methods": methods,
    }

    with open(
        out_dir / "method_definitions.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            definitions,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        out_dir / "metadata.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "BENCHMARK CONFIRMATÓRIO — CALIBRADOR INIT V4\n"
        )
        f.write(
            f"Cenário: {SCENARIO_NAME}\n"
        )
        f.write(
            f"Device: {device}\n"
        )
        f.write(
            f"Arquitetura: {[INPUT_DIM] + list(LAYERS)}\n"
        )
        f.write(
            f"Ativação: {ACTIVATION_NAME}\n"
        )
        f.write(
            f"Seeds: {seeds}\n"
        )
        f.write(
            f"Épocas: {args.epochs}\n"
        )
        f.write(
            f"Batch size: {args.batch_size}\n"
        )
        f.write(
            f"SGD lr={args.lr}, momentum={args.momentum}, "
            f"weight_decay={args.weight_decay}\n"
        )
        f.write(
            f"Val fraction: {args.val_fraction}\n"
        )
        f.write(
            f"Formula: {FORMULA_TEXT}\n"
        )
        f.write(
            f"Symbolic vector: {symbolic_vector.tolist()}\n"
        )
        f.write(
            f"Refined vector: {V4_REFINED_VECTOR.tolist()}\n"
        )
        f.write(
            "Métricas primárias: "
            "val_loss_aulc, final_val_loss\n"
        )
        f.write(
            "Holm: por métrica e por variante V4 "
            "contra He/Xavier/LeCun.\n"
        )

    save_curves(
        hist_df,
        out_dir,
    )

    print("\n" + "=" * 100)
    print("RESUMO")
    print("=" * 100)

    cols = [
        "method",
        "val_loss_aulc_mean",
        "final_val_loss_mean",
        "test_loss_mean",
        "test_acc_mean",
    ]

    print(
        summary_df[cols]
        .to_string(index=False)
    )

    print(
        "\nResultados salvos em:",
        out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
