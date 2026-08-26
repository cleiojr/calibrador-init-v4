
"""
tailored_symbolic_init_v4.py
============================

V4 COMPETITIVA / QUASE-CERTIFICADA

Objetivo:
Descobrir uma fórmula de inicialização sob medida para um cenário fixo
(dataset + topologia + ativação + protocolo de treino) e maximizar a chance
de superar He, Xavier e LeCun de forma robusta.

Exemplo:
    python tailored_symbolic_init_v4.py \
        --dataset mnist \
        --layers 32 64 128 32 10 \
        --activation relu

PRINCIPAIS DIFERENÇAS VS V3
---------------------------
1) Fitness minimax contra TODAS as baselines.
2) 10 selection seeds por padrão.
3) 20 final-test seeds por padrão.
4) Filtros duros:
       win_rate >= 0.90
       LCB95%(improvement) > min_practical_margin
5) Reavaliação forte dos elites.
6) PySR com muitos exemplos elite x seed x camada.
7) Formula selection independente.
8) NOVO: evolução/seleção DIRETA das fórmulas após PySR:
       fórmula -> variâncias -> treino -> fitness competitivo.
9) Certificado empírico final:
       W/T/L
       bootstrap LCB95%
       mean/median improvement
       paired t
       Wilcoxon
       Cohen dz
       rank-biserial

IMPORTANTE
----------
Não há garantia universal matemática de vencer baselines em todos os futuros
treinamentos. A V4 fornece um "certificado empírico" forte sobre seeds
independentes e aceita a fórmula apenas se os critérios pré-definidos forem
satisfeitos.

TEST SET
--------
O test set oficial não participa:
- da busca genética
- da seleção de elites
- do PySR
- da seleção/evolução das fórmulas

Ele só é usado no FINAL TEST.

Requisitos:
    torch torchvision numpy pandas scipy pysr
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pysr import PySRRegressor

try:
    from scipy import stats
    from scipy.stats import rankdata
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# GLOBAL
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_INFO = {
    "mnist": {
        "input_dim": 784,
        "classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashion_mnist": {
        "input_dim": 784,
        "classes": 10,
        "mean": (0.2860406,),
        "std": (0.35302424,),
    },
}

ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
}

CLASSIC_METHODS = ["lecun", "he", "xavier"]

FEATURES = [
    "fan_in",
    "fan_out",
    "depth",
    "fan_ratio",
    "log_fan_in",
    "log_fan_out",
    "dataset_input_var",
    "dataset_input_absmean",
    "dataset_input_kurtosis",
]


@dataclass
class Config:
    dataset: str
    layers: tuple
    activation: str
    output_dir: str
    data_dir: str = "./data"

    # split
    val_fraction: float = 0.15
    search_train_size: int = 12000
    search_val_size: int = 3000

    # seeds
    discovery_seeds: tuple = (11, 22, 33)
    selection_seeds: tuple = (41, 42, 43, 44, 45, 46, 47, 48, 49, 50)
    final_test_seeds: tuple = tuple(range(101, 121))

    # GA variances
    population_size: int = 24
    generations: int = 8
    elite_fraction: float = 0.20
    crossover_fraction: float = 0.35
    mutation_fraction: float = 0.30
    random_fraction: float = 0.15

    variance_min: float = 1e-5
    variance_max: float = 0.5
    mutation_sigma_log: float = 0.30

    # candidate eval
    probe_epochs: int = 4
    elite_recheck_epochs: int = 8
    validation_epochs: int = 15

    # fitness
    abs_loss_weight: float = 0.20
    abs_final_weight: float = 0.10
    minimax_aulc_weight: float = 0.35
    minimax_final_weight: float = 0.20
    acc_margin_weight: float = 0.05
    std_penalty_weight: float = 0.10

    min_win_rate: float = 0.90
    min_practical_margin: float = 0.0
    bootstrap_samples: int = 5000

    # elites / PySR
    elite_recheck_count: int = 10
    pysr_iterations: int = 1000
    max_pysr_formulas_to_select: int = 16

    # formula evolution
    formula_generations: int = 4
    formula_population_size: int = 10
    formula_mutation_sigma_log: float = 0.08
    formula_complexity_penalty: float = 1e-4

    # training
    batch_size: int = 256
    num_workers: int = 2
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0

    deterministic: bool = True


# =============================================================================
# UTIL
# =============================================================================

def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def stable_hash_vector(v: np.ndarray, decimals: int = 10):
    arr = np.round(np.asarray(v, dtype=float), decimals)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def aulc(vals):
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return np.nan
    if len(arr) == 1:
        return float(arr[0])
    x = np.linspace(0.0, 1.0, len(arr))
    return float(np.trapz(arr, x))


def bootstrap_ci(values, n_boot=5000, seed=1234):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boot[i] = np.mean(sample)

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


# =============================================================================
# MODEL
# =============================================================================

def make_act(name: str):
    return ACTS[name]()


class FixedMLP(nn.Module):
    def __init__(self, input_dim: int, layers: tuple, activation_name: str):
        super().__init__()

        dims = [input_dim] + list(layers)
        mods = []

        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                mods.append(make_act(activation_name))

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(torch.flatten(x, 1))

    def linears(self):
        return [m for m in self.modules() if isinstance(m, nn.Linear)]


# =============================================================================
# INITIALIZATIONS
# =============================================================================

@torch.no_grad()
def init_variance_vector(model: FixedMLP, variances: np.ndarray):
    linears = model.linears()

    if len(linears) != len(variances):
        raise ValueError("Número de variâncias != número de camadas.")

    for layer, var in zip(linears, variances):
        var = max(float(var), 1e-12)
        nn.init.normal_(layer.weight, 0.0, math.sqrt(var))

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

        nn.init.normal_(layer.weight, 0.0, math.sqrt(var))

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


# =============================================================================
# DATA
# =============================================================================

def load_dataset(cfg: Config):
    info = DATA_INFO[cfg.dataset]

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])

    if cfg.dataset == "mnist":
        train = datasets.MNIST(
            cfg.data_dir,
            train=True,
            download=True,
            transform=tfm,
        )
        test = datasets.MNIST(
            cfg.data_dir,
            train=False,
            download=True,
            transform=tfm,
        )

    elif cfg.dataset == "fashion_mnist":
        train = datasets.FashionMNIST(
            cfg.data_dir,
            train=True,
            download=True,
            transform=tfm,
        )
        test = datasets.FashionMNIST(
            cfg.data_dir,
            train=False,
            download=True,
            transform=tfm,
        )
    else:
        raise ValueError(cfg.dataset)

    return train, test


def split_train_val(full_train, frac):
    n_val = int(round(len(full_train) * frac))
    n_train = len(full_train) - n_val

    g = torch.Generator().manual_seed(20260820)

    return random_split(
        full_train,
        [n_train, n_val],
        generator=g,
    )


def deterministic_subset(ds, n: int, seed: int):
    if n <= 0 or n >= len(ds):
        return ds

    rng = np.random.default_rng(seed)

    idx = rng.choice(
        len(ds),
        size=n,
        replace=False,
    )

    return Subset(
        ds,
        idx.tolist(),
    )


def make_loaders(train_ds, val_ds, test_ds, cfg: Config, seed: int):
    g = torch.Generator().manual_seed(seed)

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    tr = DataLoader(
        train_ds,
        shuffle=True,
        generator=g,
        **common,
    )

    va = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )

    te = None

    if test_ds is not None:
        te = DataLoader(
            test_ds,
            shuffle=False,
            **common,
        )

    return tr, va, te


@torch.no_grad()
def estimate_dataset_stats(ds, max_samples=4096):
    loader = DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
    )

    chunks = []
    seen = 0

    for x, _ in loader:
        x = x.float().reshape(x.size(0), -1)

        chunks.append(x)
        seen += x.size(0)

        if seen >= max_samples:
            break

    X = torch.cat(chunks, 0)[:max_samples].reshape(-1)

    mean = float(X.mean().item())
    var = float(X.var(unbiased=False).item())
    absmean = float(X.abs().mean().item())

    centered = X - X.mean()
    m2 = torch.mean(centered ** 2)
    m4 = torch.mean(centered ** 4)

    kurtosis = float(
        (m4 / (m2 * m2 + 1e-12)).item()
    )

    return {
        "dataset_input_mean": mean,
        "dataset_input_var": var,
        "dataset_input_absmean": absmean,
        "dataset_input_kurtosis": kurtosis,
    }


# =============================================================================
# TRAIN
# =============================================================================

def optimizer_for(model, cfg):
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def grad_norm(model):
    s = 0.0

    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            s += float((g * g).sum().item())

    return math.sqrt(max(s, 0.0))


def train_epoch(model, loader, opt, criterion):
    model.train()

    total = 0
    loss_sum = 0.0
    correct = 0
    grad_sum = 0.0
    grad_steps = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        opt.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            return None

        loss.backward()

        gn = grad_norm(model)

        if not math.isfinite(gn):
            return None

        opt.step()

        bs = y.size(0)

        total += bs
        loss_sum += float(loss.item()) * bs
        correct += int(
            (logits.argmax(1) == y).sum().item()
        )

        grad_sum += gn
        grad_steps += 1

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "grad_norm": grad_sum / max(grad_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total = 0
    loss_sum = 0.0
    correct = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            return None

        bs = y.size(0)

        total += bs
        loss_sum += float(loss.item()) * bs
        correct += int(
            (logits.argmax(1) == y).sum().item()
        )

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
    }


def run_probe(
    cfg: Config,
    seed: int,
    epochs: int,
    train_ds,
    val_ds,
    method: Optional[str] = None,
    variance_vector: Optional[np.ndarray] = None,
):
    seed_all(seed, cfg.deterministic)

    tr, va, _ = make_loaders(
        train_ds,
        val_ds,
        None,
        cfg,
        seed,
    )

    model = FixedMLP(
        DATA_INFO[cfg.dataset]["input_dim"],
        cfg.layers,
        cfg.activation,
    ).to(DEVICE)

    if variance_vector is not None:
        init_variance_vector(
            model,
            variance_vector,
        )
    else:
        init_classic(
            model,
            method,
        )

    opt = optimizer_for(model, cfg)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []
    grads = []

    failed = False

    for _ in range(epochs):
        t = train_epoch(
            model,
            tr,
            opt,
            criterion,
        )

        if t is None:
            failed = True
            break

        v = evaluate(
            model,
            va,
            criterion,
        )

        if v is None:
            failed = True
            break

        losses.append(v["loss"])
        accs.append(v["acc"])
        grads.append(t["grad_norm"])

    result = {
        "failed": failed,
        "final_val_loss": losses[-1] if losses else np.nan,
        "final_val_acc": accs[-1] if accs else np.nan,
        "val_loss_aulc": aulc(losses) if losses else np.nan,
        "val_acc_aulc": aulc(accs) if accs else np.nan,
        "grad_norm_mean": (
            float(np.mean(grads))
            if grads
            else np.nan
        ),
    }

    del model, opt, criterion, tr, va

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# BASELINE REFERENCE
# =============================================================================

def baseline_reference(
    cfg: Config,
    seeds: tuple,
    epochs: int,
    train_ds,
    val_ds,
    out_csv: Path,
):
    rows = []

    for seed in seeds:
        for method in CLASSIC_METHODS:
            r = run_probe(
                cfg=cfg,
                seed=seed,
                epochs=epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                method=method,
            )

            row = {
                "seed": seed,
                "method": method,
                **r,
            }

            rows.append(row)
            append_csv(out_csv, row)

    df = pd.DataFrame(rows)

    ref = {}

    for seed in seeds:
        g = df[df.seed == seed].copy()

        ref[seed] = {}

        for method in CLASSIC_METHODS:
            mr = g[g.method == method].iloc[0]

            ref[seed][method] = {
                "val_loss_aulc": float(mr.val_loss_aulc),
                "final_val_loss": float(mr.final_val_loss),
                "val_acc_aulc": float(mr.val_acc_aulc),
                "final_val_acc": float(mr.final_val_acc),
            }

    return ref, df


# =============================================================================
# MINIMAX FITNESS
# =============================================================================

def candidate_improvements_vs_all(
    candidate_metrics: Dict,
    baseline_ref_seed: Dict[str, Dict],
):
    """
    Positive improvement = candidate better.
    For loss:
        improvement = baseline - candidate
    For accuracy:
        improvement = candidate - baseline
    """
    per_baseline = {}

    for b in CLASSIC_METHODS:
        base = baseline_ref_seed[b]

        per_baseline[b] = {
            "aulc_loss_improvement":
                base["val_loss_aulc"]
                - candidate_metrics["val_loss_aulc"],

            "final_loss_improvement":
                base["final_val_loss"]
                - candidate_metrics["final_val_loss"],

            "acc_aulc_improvement":
                candidate_metrics["val_acc_aulc"]
                - base["val_acc_aulc"],

            "final_acc_improvement":
                candidate_metrics["final_val_acc"]
                - base["final_val_acc"],
        }

    return per_baseline


def evaluate_candidate_minimax(
    cfg: Config,
    variance_vector: np.ndarray,
    seeds: tuple,
    epochs: int,
    train_ds,
    val_ds,
    baseline_ref: Dict[int, Dict],
):
    seed_rows = []

    for seed in seeds:
        cand = run_probe(
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            train_ds=train_ds,
            val_ds=val_ds,
            variance_vector=variance_vector,
        )

        if cand["failed"]:
            seed_rows.append({
                "seed": seed,
                "failed": True,
                "seed_score": 1e6,
                "worst_aulc_improvement": -1e6,
                "worst_final_loss_improvement": -1e6,
                "worst_acc_aulc_improvement": -1e6,
                **cand,
            })
            continue

        imp = candidate_improvements_vs_all(
            cand,
            baseline_ref[seed],
        )

        worst_aulc_imp = min(
            imp[b]["aulc_loss_improvement"]
            for b in CLASSIC_METHODS
        )

        worst_final_imp = min(
            imp[b]["final_loss_improvement"]
            for b in CLASSIC_METHODS
        )

        worst_acc_imp = min(
            imp[b]["acc_aulc_improvement"]
            for b in CLASSIC_METHODS
        )

        # Lower score is better.
        # Positive improvements reduce score.
        seed_score = (
            cfg.abs_loss_weight
            * cand["val_loss_aulc"]

            + cfg.abs_final_weight
            * cand["final_val_loss"]

            - cfg.minimax_aulc_weight
            * worst_aulc_imp

            - cfg.minimax_final_weight
            * worst_final_imp

            - cfg.acc_margin_weight
            * worst_acc_imp
        )

        seed_rows.append({
            "seed": seed,
            "failed": False,
            "seed_score": float(seed_score),
            "worst_aulc_improvement": float(worst_aulc_imp),
            "worst_final_loss_improvement": float(worst_final_imp),
            "worst_acc_aulc_improvement": float(worst_acc_imp),
            "all_improvements": imp,
            **cand,
        })

    scores = np.array(
        [r["seed_score"] for r in seed_rows],
        dtype=float,
    )

    robust_score = (
        float(np.mean(scores))
        + cfg.std_penalty_weight
        * float(np.std(scores, ddof=1))
        if len(scores) > 1
        else float(np.mean(scores))
    )

    aulc_imps = np.array(
        [r["worst_aulc_improvement"] for r in seed_rows],
        dtype=float,
    )

    final_imps = np.array(
        [r["worst_final_loss_improvement"] for r in seed_rows],
        dtype=float,
    )

    win_mask = (
        (aulc_imps > cfg.min_practical_margin)
        & (final_imps > cfg.min_practical_margin)
    )

    win_rate = float(
        np.mean(win_mask.astype(float))
    )

    aulc_ci = bootstrap_ci(
        aulc_imps,
        n_boot=cfg.bootstrap_samples,
        seed=111,
    )

    final_ci = bootstrap_ci(
        final_imps,
        n_boot=cfg.bootstrap_samples,
        seed=222,
    )

    return {
        "robust_score": robust_score,
        "win_rate": win_rate,
        "mean_worst_aulc_improvement": float(np.mean(aulc_imps)),
        "median_worst_aulc_improvement": float(np.median(aulc_imps)),
        "lcb95_worst_aulc_improvement": aulc_ci[0],
        "ucb95_worst_aulc_improvement": aulc_ci[1],
        "mean_worst_final_improvement": float(np.mean(final_imps)),
        "median_worst_final_improvement": float(np.median(final_imps)),
        "lcb95_worst_final_improvement": final_ci[0],
        "ucb95_worst_final_improvement": final_ci[1],
        "seed_rows": seed_rows,
    }


def passes_certificate(cfg: Config, ev: Dict):
    return (
        ev["win_rate"] >= cfg.min_win_rate
        and ev["lcb95_worst_aulc_improvement"] > cfg.min_practical_margin
        and ev["lcb95_worst_final_improvement"] > cfg.min_practical_margin
    )


# =============================================================================
# GA
# =============================================================================

def sample_vector(rng, n_layers, cfg):
    lo = math.log(cfg.variance_min)
    hi = math.log(cfg.variance_max)

    return np.exp(
        rng.uniform(
            lo,
            hi,
            size=n_layers,
        )
    )


def mutate_vector(v, rng, cfg):
    z = np.log(np.asarray(v, dtype=float))

    z += rng.normal(
        0.0,
        cfg.mutation_sigma_log,
        size=len(z),
    )

    return np.clip(
        np.exp(z),
        cfg.variance_min,
        cfg.variance_max,
    )


def crossover(a, b, rng, cfg):
    la = np.log(np.asarray(a))
    lb = np.log(np.asarray(b))

    alpha = rng.uniform(
        0.15,
        0.85,
        size=len(la),
    )

    child = np.exp(
        alpha * la
        + (1 - alpha) * lb
    )

    return np.clip(
        child,
        cfg.variance_min,
        cfg.variance_max,
    )


def tournament(pop, rng, k=3):
    ids = rng.choice(
        len(pop),
        size=min(k, len(pop)),
        replace=False,
    )

    cand = [pop[i] for i in ids]

    cand.sort(
        key=lambda x: x["robust_score"]
    )

    return cand[0]


def genetic_search(
    cfg: Config,
    train_ds,
    val_ds,
    baseline_ref,
    out_dir,
):
    rng = np.random.default_rng(20260820)

    n_layers = len(cfg.layers)

    population = [
        sample_vector(rng, n_layers, cfg)
        for _ in range(cfg.population_size)
    ]

    history = []
    csv_path = out_dir / "ga_population.csv"

    for gen in range(cfg.generations):
        print("\n" + "=" * 96)
        print(f"GA GENERATION {gen+1}/{cfg.generations}")
        print("=" * 96)

        evaluated = []

        for i, vec in enumerate(population):
            ev = evaluate_candidate_minimax(
                cfg=cfg,
                variance_vector=vec,
                seeds=cfg.discovery_seeds,
                epochs=cfg.probe_epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                baseline_ref=baseline_ref,
            )

            item = {
                "generation": gen,
                "index": i,
                "variance_vector": vec.copy(),
                **{k: v for k, v in ev.items() if k != "seed_rows"},
                "certified": passes_certificate(cfg, ev),
            }

            evaluated.append(item)
            history.append(item)

            append_csv(
                csv_path,
                {
                    "generation": gen,
                    "index": i,
                    "variance_vector": json.dumps([float(x) for x in vec]),
                    "robust_score": item["robust_score"],
                    "win_rate": item["win_rate"],
                    "lcb95_worst_aulc_improvement":
                        item["lcb95_worst_aulc_improvement"],
                    "lcb95_worst_final_improvement":
                        item["lcb95_worst_final_improvement"],
                    "certified": item["certified"],
                }
            )

            print(
                f"g={gen} i={i:02d} "
                f"score={item['robust_score']:.5f} "
                f"win={item['win_rate']:.2f} "
                f"LCB_AULC={item['lcb95_worst_aulc_improvement']:+.5f} "
                f"LCB_FINAL={item['lcb95_worst_final_improvement']:+.5f} "
                f"cert={item['certified']}"
            )

        evaluated.sort(
            key=lambda x: x["robust_score"]
        )

        best = evaluated[0]

        print(
            "\nBest generation:",
            best["robust_score"],
            "win=", best["win_rate"],
            "var=", np.round(
                best["variance_vector"],
                8,
            ).tolist(),
        )

        elite_n = max(
            2,
            int(round(
                cfg.population_size
                * cfg.elite_fraction
            ))
        )

        cross_n = int(round(
            cfg.population_size
            * cfg.crossover_fraction
        ))

        mut_n = int(round(
            cfg.population_size
            * cfg.mutation_fraction
        ))

        rand_n = (
            cfg.population_size
            - elite_n
            - cross_n
            - mut_n
        )

        next_pop = [
            x["variance_vector"].copy()
            for x in evaluated[:elite_n]
        ]

        for _ in range(cross_n):
            a = tournament(evaluated, rng)
            b = tournament(evaluated, rng)

            next_pop.append(
                crossover(
                    a["variance_vector"],
                    b["variance_vector"],
                    rng,
                    cfg,
                )
            )

        for _ in range(mut_n):
            p = tournament(evaluated, rng)

            next_pop.append(
                mutate_vector(
                    p["variance_vector"],
                    rng,
                    cfg,
                )
            )

        for _ in range(rand_n):
            next_pop.append(
                sample_vector(
                    rng,
                    n_layers,
                    cfg,
                )
            )

        population = next_pop[:cfg.population_size]

    # unique hall-of-fame
    unique = {}

    for x in history:
        h = stable_hash_vector(
            x["variance_vector"]
        )

        if (
            h not in unique
            or x["robust_score"]
            < unique[h]["robust_score"]
        ):
            unique[h] = x

    hof = list(unique.values())

    hof.sort(
        key=lambda x: x["robust_score"]
    )

    return hof


# =============================================================================
# ELITE RECHECK
# =============================================================================

def recheck_elites(
    cfg: Config,
    hof,
    train_ds,
    val_ds,
    selection_ref,
    out_dir,
):
    rows = []
    csv_path = out_dir / "elite_recheck.csv"

    for rank, item in enumerate(
        hof[:cfg.elite_recheck_count],
        start=1,
    ):
        ev = evaluate_candidate_minimax(
            cfg=cfg,
            variance_vector=item["variance_vector"],
            seeds=cfg.selection_seeds,
            epochs=cfg.elite_recheck_epochs,
            train_ds=train_ds,
            val_ds=val_ds,
            baseline_ref=selection_ref,
        )

        cert = passes_certificate(
            cfg,
            ev,
        )

        row = {
            "rank": rank,
            "variance_vector":
                item["variance_vector"].copy(),
            "evaluation": ev,
            "certified": cert,
        }

        rows.append(row)

        append_csv(
            csv_path,
            {
                "rank": rank,
                "variance_vector":
                    json.dumps(
                        [
                            float(x)
                            for x
                            in item["variance_vector"]
                        ]
                    ),
                "robust_score": ev["robust_score"],
                "win_rate": ev["win_rate"],
                "lcb95_worst_aulc_improvement":
                    ev["lcb95_worst_aulc_improvement"],
                "lcb95_worst_final_improvement":
                    ev["lcb95_worst_final_improvement"],
                "certified": cert,
            }
        )

        print(
            f"elite#{rank} "
            f"score={ev['robust_score']:.5f} "
            f"win={ev['win_rate']:.2f} "
            f"LCB_AULC={ev['lcb95_worst_aulc_improvement']:+.5f} "
            f"LCB_FINAL={ev['lcb95_worst_final_improvement']:+.5f} "
            f"cert={cert}"
        )

    certified = [
        r for r in rows
        if r["certified"]
    ]

    if certified:
        return certified

    print(
        "\nNenhum elite passou certificado duro; "
        "seguindo com melhores reavaliados como fallback."
    )

    rows.sort(
        key=lambda x:
        x["evaluation"]["robust_score"]
    )

    return rows[:max(
        2,
        min(4, len(rows)),
    )]


# =============================================================================
# PySR TRAINING TABLE
# =============================================================================

def build_pysr_table(
    cfg: Config,
    elites,
    dataset_stats,
):
    dims = [
        DATA_INFO[cfg.dataset]["input_dim"]
    ] + list(cfg.layers)

    L = len(cfg.layers)
    rows = []

    robust_scores = np.array(
        [
            e["evaluation"]["robust_score"]
            for e in elites
        ],
        dtype=float,
    )

    best = np.min(robust_scores)

    quality = 1.0 / (
        1.0
        + (robust_scores - best)
    )

    for ei, elite in enumerate(elites):
        vec = elite["variance_vector"]
        q = float(quality[ei])

        for seed_row in elite["evaluation"]["seed_rows"]:
            seed_cert_weight = 1.25

            if seed_row["worst_aulc_improvement"] <= 0:
                seed_cert_weight = 0.75

            for l in range(L):
                fi = dims[l]
                fo = dims[l + 1]

                rows.append({
                    "elite_index": ei,
                    "selection_seed": seed_row["seed"],
                    "layer_index": l,

                    "fan_in": fi,
                    "fan_out": fo,
                    "depth": (l + 1) / L,
                    "fan_ratio": fo / fi,
                    "log_fan_in": math.log(float(fi)),
                    "log_fan_out": math.log(float(fo)),

                    "dataset_input_var":
                        dataset_stats["dataset_input_var"],

                    "dataset_input_absmean":
                        dataset_stats["dataset_input_absmean"],

                    "dataset_input_kurtosis":
                        dataset_stats["dataset_input_kurtosis"],

                    "target_variance": float(vec[l]),

                    "sample_weight":
                        q * seed_cert_weight,
                })

    return pd.DataFrame(rows)


# =============================================================================
# PySR
# =============================================================================

def fit_pysr(cfg, table, out_dir):
    X = table[FEATURES].copy()

    y = table[
        "target_variance"
    ].to_numpy(dtype=float)

    weights = table[
        "sample_weight"
    ].to_numpy(dtype=float)

    model = PySRRegressor(
        niterations=cfg.pysr_iterations,
        populations=24,
        population_size=48,

        binary_operators=[
            "+",
            "-",
            "*",
            "/",
        ],

        unary_operators=[
            "sqrt",
            "log",
            "exp",
            "square",
        ],

        maxsize=16,
        maxdepth=8,

        complexity_of_operators={
            "+": 1,
            "-": 1,
            "*": 1,
            "/": 2,
            "sqrt": 2,
            "log": 2,
            "exp": 3,
            "square": 2,
        },

        complexity_of_constants=2,

        model_selection="best",
        random_state=42,
        deterministic=True,
        parallelism="serial",
        verbosity=1,
        progress=True,
    )

    model.fit(
        X,
        y,
        weights=weights,
        variable_names=FEATURES,
    )

    eqs = model.equations_.copy()

    eqs.to_csv(
        out_dir / "pysr_equations.csv",
        index=False,
    )

    return model, eqs


# =============================================================================
# FORMULA FEATURE FRAME
# =============================================================================

def layer_feature_frame(
    cfg: Config,
    dataset_stats: Dict,
):
    dims = [
        DATA_INFO[cfg.dataset]["input_dim"]
    ] + list(cfg.layers)

    L = len(cfg.layers)

    rows = []

    for l in range(L):
        fi = dims[l]
        fo = dims[l + 1]

        rows.append({
            "fan_in": fi,
            "fan_out": fo,
            "depth": (l + 1) / L,
            "fan_ratio": fo / fi,
            "log_fan_in": math.log(float(fi)),
            "log_fan_out": math.log(float(fo)),
            "dataset_input_var":
                dataset_stats["dataset_input_var"],
            "dataset_input_absmean":
                dataset_stats["dataset_input_absmean"],
            "dataset_input_kurtosis":
                dataset_stats["dataset_input_kurtosis"],
        })

    return pd.DataFrame(rows)


def equation_shortlist(
    eqs: pd.DataFrame,
    n: int,
):
    work = eqs.copy()

    if "loss" in work.columns:
        work = work.sort_values(
            ["loss", "complexity"],
            ascending=[True, True],
        )

    elif "score" in work.columns:
        work = work.sort_values(
            ["score", "complexity"],
            ascending=[False, True],
        )

    else:
        work = work.sort_values(
            "complexity"
        )

    return list(
        work.index[:n]
    )


def predict_equation(
    model,
    eq_index,
    X,
):
    try:
        pred = np.asarray(
            model.predict(
                X,
                index=eq_index,
            ),
            dtype=float,
        ).reshape(-1)
    except Exception:
        return None

    if not np.all(np.isfinite(pred)):
        return None

    pred = np.clip(
        pred,
        1e-10,
        10.0,
    )

    return pred


# =============================================================================
# FORMULA DIRECT EVOLUTION
# =============================================================================

def mutate_formula_vector(
    variances: np.ndarray,
    rng,
    sigma: float,
    vmin: float,
    vmax: float,
):
    """
    Proxy for direct formula evolution:
    mutate the variance vector produced by a symbolic formula.
    The symbolic formula remains the seed/template; mutated candidates compete
    directly in network training. The best evolved vector is then optionally
    redistilled by a second symbolic regression pass.
    """
    z = np.log(
        np.asarray(
            variances,
            dtype=float,
        )
    )

    z += rng.normal(
        0.0,
        sigma,
        size=len(z),
    )

    return np.clip(
        np.exp(z),
        vmin,
        vmax,
    )


def direct_formula_competition(
    cfg: Config,
    model,
    eqs: pd.DataFrame,
    dataset_stats,
    train_ds,
    val_ds,
    selection_ref,
    out_dir,
):
    Xlayers = layer_feature_frame(
        cfg,
        dataset_stats,
    )

    seeds = []

    for eq_index in equation_shortlist(
        eqs,
        cfg.max_pysr_formulas_to_select,
    ):
        vec = predict_equation(
            model,
            eq_index,
            Xlayers,
        )

        if vec is None:
            continue

        complexity = (
            float(eqs.loc[eq_index, "complexity"])
            if "complexity" in eqs.columns
            else np.nan
        )

        equation = (
            str(eqs.loc[eq_index, "equation"])
            if "equation" in eqs.columns
            else str(eq_index)
        )

        ev = evaluate_candidate_minimax(
            cfg=cfg,
            variance_vector=vec,
            seeds=cfg.selection_seeds,
            epochs=cfg.elite_recheck_epochs,
            train_ds=train_ds,
            val_ds=val_ds,
            baseline_ref=selection_ref,
        )

        score = (
            ev["robust_score"]
            + cfg.formula_complexity_penalty
            * (
                complexity
                if np.isfinite(complexity)
                else 0.0
            )
        )

        seeds.append({
            "origin": "pysr",
            "eq_index": eq_index,
            "equation": equation,
            "complexity": complexity,
            "variance_vector": vec.copy(),
            "evaluation": ev,
            "score": score,
            "certified": passes_certificate(cfg, ev),
        })

    if not seeds:
        raise RuntimeError(
            "Nenhuma equação PySR válida."
        )

    seeds.sort(
        key=lambda x: x["score"]
    )

    rng = np.random.default_rng(9090)

    population = seeds[
        :min(
            cfg.formula_population_size,
            len(seeds)
        )
    ]

    csv_path = (
        out_dir
        / "formula_direct_evolution.csv"
    )

    for gen in range(
        cfg.formula_generations
    ):
        evaluated = []

        for i, item in enumerate(population):
            if gen == 0:
                candidate = item
            else:
                candidate = item

            append_csv(
                csv_path,
                {
                    "generation": gen,
                    "index": i,
                    "origin": candidate["origin"],
                    "eq_index": candidate.get("eq_index"),
                    "equation": candidate.get("equation"),
                    "complexity": candidate.get("complexity"),
                    "variance_vector":
                        json.dumps(
                            [
                                float(x)
                                for x
                                in candidate["variance_vector"]
                            ]
                        ),
                    "score": candidate["score"],
                    "win_rate":
                        candidate["evaluation"]["win_rate"],
                    "lcb95_aulc":
                        candidate["evaluation"][
                            "lcb95_worst_aulc_improvement"
                        ],
                    "lcb95_final":
                        candidate["evaluation"][
                            "lcb95_worst_final_improvement"
                        ],
                    "certified":
                        candidate["certified"],
                }
            )

            evaluated.append(candidate)

        evaluated.sort(
            key=lambda x: x["score"]
        )

        if gen == cfg.formula_generations - 1:
            population = evaluated
            break

        elite_count = max(
            2,
            min(
                4,
                len(evaluated)
            ),
        )

        next_pop = evaluated[
            :elite_count
        ]

        while len(next_pop) < (
            cfg.formula_population_size
        ):
            parent = evaluated[
                rng.integers(
                    0,
                    min(
                        max(3, elite_count),
                        len(evaluated),
                    )
                )
            ]

            vec = mutate_formula_vector(
                parent["variance_vector"],
                rng,
                cfg.formula_mutation_sigma_log,
                cfg.variance_min,
                cfg.variance_max,
            )

            ev = evaluate_candidate_minimax(
                cfg=cfg,
                variance_vector=vec,
                seeds=cfg.selection_seeds,
                epochs=cfg.elite_recheck_epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                baseline_ref=selection_ref,
            )

            complexity = parent.get(
                "complexity",
                np.nan,
            )

            score = (
                ev["robust_score"]
                + cfg.formula_complexity_penalty
                * (
                    complexity
                    if np.isfinite(complexity)
                    else 0.0
                )
            )

            next_pop.append({
                "origin": "formula_mutation",
                "eq_index": parent.get("eq_index"),
                "equation": parent.get("equation"),
                "complexity": complexity,
                "variance_vector": vec,
                "evaluation": ev,
                "score": score,
                "certified": passes_certificate(cfg, ev),
            })

        population = next_pop

    population.sort(
        key=lambda x: x["score"]
    )

    certified = [
        x for x in population
        if x["certified"]
    ]

    best = (
        certified[0]
        if certified
        else population[0]
    )

    return best


# =============================================================================
# FINAL TEST
# =============================================================================

def full_final_run(
    cfg: Config,
    method: str,
    seed: int,
    formula_variances,
    train_ds,
    val_ds,
    test_ds,
):
    seed_all(
        seed,
        cfg.deterministic,
    )

    tr, va, te = make_loaders(
        train_ds,
        val_ds,
        test_ds,
        cfg,
        seed,
    )

    model = FixedMLP(
        DATA_INFO[cfg.dataset]["input_dim"],
        cfg.layers,
        cfg.activation,
    ).to(DEVICE)

    if method == "tailored_symbolic_v4":
        init_variance_vector(
            model,
            formula_variances,
        )
    else:
        init_classic(
            model,
            method,
        )

    opt = optimizer_for(
        model,
        cfg,
    )

    criterion = nn.CrossEntropyLoss()

    val_losses = []
    val_accs = []
    grads = []
    failed = False

    for _ in range(
        cfg.validation_epochs
    ):
        t = train_epoch(
            model,
            tr,
            opt,
            criterion,
        )

        if t is None:
            failed = True
            break

        v = evaluate(
            model,
            va,
            criterion,
        )

        if v is None:
            failed = True
            break

        val_losses.append(
            v["loss"]
        )

        val_accs.append(
            v["acc"]
        )

        grads.append(
            t["grad_norm"]
        )

    test = (
        evaluate(
            model,
            te,
            criterion,
        )
        if not failed
        else None
    )

    return {
        "method": method,
        "seed": seed,
        "failed": failed,

        "final_val_loss":
            val_losses[-1]
            if val_losses
            else np.nan,

        "final_val_acc":
            val_accs[-1]
            if val_accs
            else np.nan,

        "val_loss_aulc":
            aulc(val_losses)
            if val_losses
            else np.nan,

        "val_acc_aulc":
            aulc(val_accs)
            if val_accs
            else np.nan,

        "grad_norm_mean":
            float(np.mean(grads))
            if grads
            else np.nan,

        "test_loss":
            test["loss"]
            if test
            else np.nan,

        "test_acc":
            test["acc"]
            if test
            else np.nan,
    }


# =============================================================================
# FINAL CERTIFICATE
# =============================================================================

def rank_biserial(diff):
    if not SCIPY_AVAILABLE:
        return np.nan

    d = np.asarray(diff, dtype=float)

    d = d[
        np.isfinite(d)
    ]

    d = d[
        d != 0
    ]

    if not len(d):
        return 0.0

    ranks = rankdata(
        np.abs(d)
    )

    wp = ranks[
        d > 0
    ].sum()

    wm = ranks[
        d < 0
    ].sum()

    return float(
        (wp - wm)
        / (wp + wm)
    )


def final_pairwise_and_certificate(
    cfg: Config,
    df: pd.DataFrame,
):
    if not SCIPY_AVAILABLE:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    focal = df[
        df.method
        == "tailored_symbolic_v4"
    ]

    stat_rows = []
    cert_rows = []

    for baseline in CLASSIC_METHODS:
        base = df[
            df.method == baseline
        ]

        merged = focal.merge(
            base,
            on="seed",
            suffixes=("_sym", "_base"),
        )

        # certificate focused on losses
        for metric in [
            "val_loss_aulc",
            "final_val_loss",
            "test_loss",
        ]:
            a = pd.to_numeric(
                merged[
                    f"{metric}_sym"
                ],
                errors="coerce",
            ).to_numpy()

            b = pd.to_numeric(
                merged[
                    f"{metric}_base"
                ],
                errors="coerce",
            ).to_numpy()

            ok = (
                np.isfinite(a)
                & np.isfinite(b)
            )

            a = a[ok]
            b = b[ok]

            if len(a) < 2:
                continue

            improvement = (
                b - a
            )

            lo, hi = bootstrap_ci(
                improvement,
                n_boot=cfg.bootstrap_samples,
                seed=123,
            )

            win_rate = float(
                np.mean(
                    (
                        improvement
                        > cfg.min_practical_margin
                    ).astype(float)
                )
            )

            cert = (
                win_rate
                >= cfg.min_win_rate
                and lo
                > cfg.min_practical_margin
            )

            cert_rows.append({
                "baseline": baseline,
                "metric": metric,
                "n": len(improvement),
                "mean_improvement": float(
                    np.mean(improvement)
                ),
                "median_improvement": float(
                    np.median(improvement)
                ),
                "lcb95": lo,
                "ucb95": hi,
                "win_rate": win_rate,
                "certificate_passed": cert,
            })

        # full stats
        for metric in [
            "final_val_loss",
            "final_val_acc",
            "val_loss_aulc",
            "val_acc_aulc",
            "test_loss",
            "test_acc",
        ]:
            a = pd.to_numeric(
                merged[
                    f"{metric}_sym"
                ],
                errors="coerce",
            ).to_numpy()

            b = pd.to_numeric(
                merged[
                    f"{metric}_base"
                ],
                errors="coerce",
            ).to_numpy()

            ok = (
                np.isfinite(a)
                & np.isfinite(b)
            )

            a = a[ok]
            b = b[ok]

            if len(a) < 2:
                continue

            lower_better = (
                "loss" in metric
            )

            improvement = (
                b - a
                if lower_better
                else a - b
            )

            lo, hi = bootstrap_ci(
                improvement,
                n_boot=cfg.bootstrap_samples,
                seed=321,
            )

            d = (
                a - b
            )

            sd = np.std(
                d,
                ddof=1,
            )

            dz = (
                np.mean(d) / sd
                if sd > 0
                else np.nan
            )

            tt = stats.ttest_rel(
                a,
                b,
            )

            try:
                ww = stats.wilcoxon(
                    a,
                    b,
                )
                wp = float(
                    ww.pvalue
                )
            except Exception:
                wp = np.nan

            wins = int(
                np.sum(
                    improvement
                    > cfg.min_practical_margin
                )
            )

            losses = int(
                np.sum(
                    improvement
                    < -cfg.min_practical_margin
                )
            )

            ties = (
                len(improvement)
                - wins
                - losses
            )

            stat_rows.append({
                "comparison":
                    f"tailored_symbolic_v4_vs_{baseline}",

                "metric": metric,
                "n": len(a),

                "symbolic_mean":
                    float(np.mean(a)),

                "baseline_mean":
                    float(np.mean(b)),

                "mean_improvement":
                    float(np.mean(improvement)),

                "median_improvement":
                    float(np.median(improvement)),

                "lcb95_improvement": lo,
                "ucb95_improvement": hi,

                "cohen_dz": float(dz),

                "rank_biserial":
                    rank_biserial(
                        improvement
                    ),

                "paired_t_p":
                    float(
                        tt.pvalue
                    ),

                "wilcoxon_p": wp,

                "wins": wins,
                "ties": ties,
                "losses": losses,
            })

    return (
        pd.DataFrame(stat_rows),
        pd.DataFrame(cert_rows),
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        choices=list(DATA_INFO),
        required=True,
    )

    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--activation",
        choices=sorted(ACTS),
        required=True,
    )

    p.add_argument(
        "--output-dir",
        default="./tailored_symbolic_v4_results",
    )

    p.add_argument(
        "--data-dir",
        default="./data",
    )

    p.add_argument(
        "--search-train-size",
        type=int,
        default=12000,
    )

    p.add_argument(
        "--search-val-size",
        type=int,
        default=3000,
    )

    p.add_argument(
        "--population-size",
        type=int,
        default=24,
    )

    p.add_argument(
        "--generations",
        type=int,
        default=8,
    )

    p.add_argument(
        "--probe-epochs",
        type=int,
        default=4,
    )

    p.add_argument(
        "--elite-recheck-count",
        type=int,
        default=10,
    )

    p.add_argument(
        "--elite-recheck-epochs",
        type=int,
        default=8,
    )

    p.add_argument(
        "--variance-min",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--variance-max",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--min-win-rate",
        type=float,
        default=0.90,
    )

    p.add_argument(
        "--min-practical-margin",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--pysr-iterations",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--max-pysr-formulas-to-select",
        type=int,
        default=16,
    )

    p.add_argument(
        "--formula-generations",
        type=int,
        default=4,
    )

    p.add_argument(
        "--formula-population-size",
        type=int,
        default=10,
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
        "--validation-epochs",
        type=int,
        default=15,
    )

    p.add_argument(
        "--discovery-seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )

    p.add_argument(
        "--selection-seeds",
        type=int,
        nargs="+",
        default=[
            41, 42, 43, 44, 45,
            46, 47, 48, 49, 50,
        ],
    )

    p.add_argument(
        "--final-test-seeds",
        type=int,
        nargs="+",
        default=list(range(101, 121)),
    )

    p.add_argument(
        "--non-deterministic",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    args, unknown = p.parse_known_args()

    if unknown:
        print(
            "Ignorando argumentos externos:",
            unknown,
        )

    return args


# =============================================================================
# MAIN
# =============================================================================

def main():
    a = parse_args()

    if (
        a.layers[-1]
        != DATA_INFO[a.dataset]["classes"]
    ):
        raise ValueError(
            "Última camada incompatível "
            "com número de classes."
        )

    cfg = Config(
        dataset=a.dataset,
        layers=tuple(a.layers),
        activation=a.activation,
        output_dir=a.output_dir,
        data_dir=a.data_dir,

        search_train_size=a.search_train_size,
        search_val_size=a.search_val_size,

        discovery_seeds=tuple(
            a.discovery_seeds
        ),

        selection_seeds=tuple(
            a.selection_seeds
        ),

        final_test_seeds=tuple(
            a.final_test_seeds
        ),

        population_size=a.population_size,
        generations=a.generations,
        probe_epochs=a.probe_epochs,

        elite_recheck_count=
            a.elite_recheck_count,

        elite_recheck_epochs=
            a.elite_recheck_epochs,

        variance_min=
            a.variance_min,

        variance_max=
            a.variance_max,

        min_win_rate=
            a.min_win_rate,

        min_practical_margin=
            a.min_practical_margin,

        pysr_iterations=
            a.pysr_iterations,

        max_pysr_formulas_to_select=
            a.max_pysr_formulas_to_select,

        formula_generations=
            a.formula_generations,

        formula_population_size=
            a.formula_population_size,

        batch_size=
            a.batch_size,

        num_workers=
            a.num_workers,

        lr=a.lr,
        momentum=a.momentum,
        weight_decay=a.weight_decay,

        validation_epochs=
            a.validation_epochs,

        deterministic=
            not a.non_deterministic,
    )

    out = Path(
        cfg.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_payload = asdict(cfg)

    config_payload["layers"] = list(
        cfg.layers
    )

    config_payload["discovery_seeds"] = list(
        cfg.discovery_seeds
    )

    config_payload["selection_seeds"] = list(
        cfg.selection_seeds
    )

    config_payload["final_test_seeds"] = list(
        cfg.final_test_seeds
    )

    with open(
        out / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    topology = [
        DATA_INFO[cfg.dataset]["input_dim"]
    ] + list(
        cfg.layers
    )

    print("=" * 100)
    print("TAILORED SYMBOLIC INITIALIZATION — V4")
    print("=" * 100)
    print("Device          :", DEVICE)
    print("Dataset         :", cfg.dataset)
    print("Topology        :", topology)
    print("Activation      :", cfg.activation)
    print("Discovery seeds :", cfg.discovery_seeds)
    print("Selection seeds :", cfg.selection_seeds)
    print("Final seeds     :", cfg.final_test_seeds)
    print("Min win-rate    :", cfg.min_win_rate)
    print("Min margin      :", cfg.min_practical_margin)

    if a.dry_run:
        discovery_eval = (
            cfg.population_size
            * cfg.generations
            * len(
                cfg.discovery_seeds
            )
        )

        discovery_baselines = (
            len(CLASSIC_METHODS)
            * len(
                cfg.discovery_seeds
            )
        )

        selection_baselines = (
            len(CLASSIC_METHODS)
            * len(
                cfg.selection_seeds
            )
        )

        elite_recheck = (
            cfg.elite_recheck_count
            * len(
                cfg.selection_seeds
            )
        )

        formula_selection_upper = (
            cfg.max_pysr_formulas_to_select
            * len(
                cfg.selection_seeds
            )
        )

        formula_evolution_upper = (
            cfg.formula_generations
            * cfg.formula_population_size
            * len(
                cfg.selection_seeds
            )
        )

        final_runs = (
            (
                1
                + len(
                    CLASSIC_METHODS
                )
            )
            * len(
                cfg.final_test_seeds
            )
        )

        print("\nDRY RUN")
        print(
            "GA candidate-seed evals :",
            discovery_eval,
        )
        print(
            "Discovery baselines      :",
            discovery_baselines,
        )
        print(
            "Selection baselines      :",
            selection_baselines,
        )
        print(
            "Elite recheck evals       :",
            elite_recheck,
        )
        print(
            "Formula selection upper   :",
            formula_selection_upper,
        )
        print(
            "Formula evolution upper   :",
            formula_evolution_upper,
        )
        print(
            "Final independent runs    :",
            final_runs,
        )

        return

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    full_train, test_ds = load_dataset(
        cfg
    )

    train_ds, val_ds = split_train_val(
        full_train,
        cfg.val_fraction,
    )

    search_train = deterministic_subset(
        train_ds,
        cfg.search_train_size,
        seed=777,
    )

    search_val = deterministic_subset(
        val_ds,
        cfg.search_val_size,
        seed=888,
    )

    dataset_stats = estimate_dataset_stats(
        search_train
    )

    with open(
        out / "dataset_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataset_stats,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # DISCOVERY BASELINES
    # -------------------------------------------------------------------------

    discovery_ref, _ = baseline_reference(
        cfg=cfg,
        seeds=cfg.discovery_seeds,
        epochs=cfg.probe_epochs,
        train_ds=search_train,
        val_ds=search_val,
        out_csv=
            out
            / "baseline_reference_discovery.csv",
    )

    # -------------------------------------------------------------------------
    # GA
    # -------------------------------------------------------------------------

    hof = genetic_search(
        cfg=cfg,
        train_ds=search_train,
        val_ds=search_val,
        baseline_ref=discovery_ref,
        out_dir=out,
    )

    # -------------------------------------------------------------------------
    # SELECTION BASELINES
    # -------------------------------------------------------------------------

    selection_ref, _ = baseline_reference(
        cfg=cfg,
        seeds=cfg.selection_seeds,
        epochs=cfg.elite_recheck_epochs,
        train_ds=search_train,
        val_ds=search_val,
        out_csv=
            out
            / "baseline_reference_selection.csv",
    )

    # -------------------------------------------------------------------------
    # RECHECK ELITES
    # -------------------------------------------------------------------------

    elites = recheck_elites(
        cfg=cfg,
        hof=hof,
        train_ds=search_train,
        val_ds=search_val,
        selection_ref=selection_ref,
        out_dir=out,
    )

    # -------------------------------------------------------------------------
    # PySR TABLE
    # -------------------------------------------------------------------------

    table = build_pysr_table(
        cfg=cfg,
        elites=elites,
        dataset_stats=dataset_stats,
    )

    table.to_csv(
        out
        / "pysr_training_points.csv",
        index=False,
    )

    print(
        "PySR rows:",
        len(table),
    )

    # -------------------------------------------------------------------------
    # PySR
    # -------------------------------------------------------------------------

    pysr_model, eqs = fit_pysr(
        cfg=cfg,
        table=table,
        out_dir=out,
    )

    # -------------------------------------------------------------------------
    # DIRECT FORMULA EVOLUTION
    # -------------------------------------------------------------------------

    best_formula = direct_formula_competition(
        cfg=cfg,
        model=pysr_model,
        eqs=eqs,
        dataset_stats=dataset_stats,
        train_ds=search_train,
        val_ds=search_val,
        selection_ref=selection_ref,
        out_dir=out,
    )

    final_variances = (
        best_formula[
            "variance_vector"
        ]
    )

    formula_certified = (
        best_formula[
            "certified"
        ]
    )

    with open(
        out / "best_formula.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "TAILORED SYMBOLIC INITIALIZATION V4\n"
        )

        f.write(
            f"Origin: {best_formula['origin']}\n"
        )

        f.write(
            f"PySR equation: "
            f"{best_formula.get('equation')}\n"
        )

        f.write(
            f"Selection score: "
            f"{best_formula['score']}\n"
        )

        f.write(
            f"Selection certified: "
            f"{formula_certified}\n"
        )

        f.write(
            "Variance vector: "
            + json.dumps(
                [
                    float(x)
                    for x
                    in final_variances
                ]
            )
            + "\n"
        )

    Xlayers = layer_feature_frame(
        cfg,
        dataset_stats,
    )

    Xlayers[
        "final_variance"
    ] = final_variances

    Xlayers[
        "final_std"
    ] = np.sqrt(
        final_variances
    )

    Xlayers.to_csv(
        out
        / "best_formula_by_layer.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("FORMULA / VECTOR FROZEN")
    print("=" * 100)
    print(
        "Certified on selection:",
        formula_certified,
    )
    print(
        "Variances:",
        np.round(
            final_variances,
            10,
        ).tolist(),
    )

    # -------------------------------------------------------------------------
    # FINAL TEST
    # -------------------------------------------------------------------------

    final_rows = []
    final_csv = (
        out
        / "final_test_runs.csv"
    )

    for seed in cfg.final_test_seeds:
        for method in (
            ["tailored_symbolic_v4"]
            + CLASSIC_METHODS
        ):
            r = full_final_run(
                cfg=cfg,
                method=method,
                seed=seed,
                formula_variances=final_variances,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
            )

            final_rows.append(
                r
            )

            append_csv(
                final_csv,
                r,
            )

            print(
                f"seed={seed} "
                f"{method:21s} "
                f"val_loss={r['final_val_loss']:.5f} "
                f"val_acc={r['final_val_acc']:.4f} "
                f"test_loss={r['test_loss']:.5f} "
                f"test_acc={r['test_acc']:.4f}"
            )

    final_df = pd.DataFrame(
        final_rows
    )

    summary = (
        final_df
        .groupby("method")
        .agg({
            "final_val_loss":
                ["mean", "std"],

            "final_val_acc":
                ["mean", "std"],

            "val_loss_aulc":
                ["mean", "std"],

            "val_acc_aulc":
                ["mean", "std"],

            "test_loss":
                ["mean", "std"],

            "test_acc":
                ["mean", "std"],
        })
    )

    summary.to_csv(
        out
        / "final_test_summary.csv"
    )

    stats_df, cert_df = (
        final_pairwise_and_certificate(
            cfg,
            final_df,
        )
    )

    if len(stats_df):
        stats_df.to_csv(
            out
            / "final_pairwise_stats.csv",
            index=False,
        )

    if len(cert_df):
        cert_df.to_csv(
            out
            / "final_certificate.csv",
            index=False,
        )

    overall_cert = False

    if len(cert_df):
        required = cert_df[
            cert_df.metric.isin(
                [
                    "val_loss_aulc",
                    "final_val_loss",
                ]
            )
        ]

        overall_cert = (
            len(required)
            == 2 * len(
                CLASSIC_METHODS
            )
            and bool(
                required[
                    "certificate_passed"
                ].all()
            )
        )

    with open(
        out
        / "certificate_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "TAILORED SYMBOLIC INITIALIZATION V4\n"
        )
        f.write(
            f"Overall empirical certificate: "
            f"{overall_cert}\n"
        )
        f.write(
            f"Criterion: win_rate >= "
            f"{cfg.min_win_rate} and "
            f"LCB95 improvement > "
            f"{cfg.min_practical_margin}\n"
        )

    print("\n" + "=" * 100)
    print("V4 FINALIZADA")
    print("=" * 100)
    print(
        "Empirical certificate:",
        overall_cert,
    )
    print(
        "Best formula:",
        out
        / "best_formula.txt",
    )
    print(
        "Final certificate:",
        out
        / "final_certificate.csv",
    )
    print(
        "Certificate summary:",
        out
        / "certificate_summary.txt",
    )


if __name__ == "__main__":
    main()
