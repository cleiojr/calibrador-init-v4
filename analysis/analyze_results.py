#!/usr/bin/env python3
"""Generate derived, publication-friendly analyses for Calibrador Init V4.

Reads each completed scenario under results/<scenario>/ and creates:
  derived/compact_summary.csv
  derived/ranking_by_metric.csv
  derived/v4_primary_evidence.csv
  derived/results_summary.md
  figures/loss_aulc_pairwise_ci.png

It also creates aggregate cross-scenario tables and figures under
results/aggregate/. No model training is performed.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCENARIOS = ["mnist_relu", "fashion_mnist_silu", "cifar10_gelu"]
CLASSICAL = {"he", "xavier", "lecun"}
METRICS = {
    "val_loss_aulc": ("Loss AULC", "lower"),
    "final_val_loss": ("Final validation loss", "lower"),
    "val_acc_aulc": ("Accuracy AULC", "higher"),
    "final_val_acc": ("Final validation accuracy", "higher"),
    "test_loss": ("Test loss", "lower"),
    "test_acc": ("Test accuracy", "higher"),
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fnum(x):
    try:
        if x is None or x == "":
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def integer(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def mean_std(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if not len(a):
        return math.nan, math.nan, math.nan
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, float(np.median(a))


def completed_row(r):
    # New final benchmark explicitly records completed. Older confirmed scripts
    # are considered complete unless failed=True or epochs_completed<epochs.
    if "completed" in r and r.get("completed", "") != "":
        return r["completed"].strip().lower() in {"true", "1", "yes"}
    if r.get("failed", "").strip().lower() in {"true", "1", "yes"}:
        return False
    if r.get("epochs_completed") and r.get("epochs"):
        return integer(r["epochs_completed"]) >= integer(r["epochs"])
    return True


def scenario_analysis(scenario: str):
    sdir = RESULTS / scenario
    derived = sdir / "derived"
    figures = sdir / "figures"
    derived.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    runs = read_csv(sdir / "runs.csv")
    pairwise = read_csv(sdir / "pairwise_statistics.csv")
    completed = [r for r in runs if completed_row(r)]

    methods = sorted({r["method"] for r in runs})
    compact = []
    metric_means = defaultdict(dict)
    for method in methods:
        attempted = [r for r in runs if r["method"] == method]
        good = [r for r in completed if r["method"] == method]
        row = {
            "scenario": scenario,
            "method": method,
            "n_attempted": len(attempted),
            "n_completed": len(good),
            "completion_rate": len(good) / len(attempted) if attempted else math.nan,
        }
        for metric, (label, direction) in METRICS.items():
            vals = [fnum(r.get(metric)) for r in good]
            mu, sd, med = mean_std(vals)
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}_median"] = med
            metric_means[metric][method] = mu
        compact.append(row)

    compact_fields = list(compact[0].keys()) if compact else []
    write_csv(derived / "compact_summary.csv", compact_fields, compact)

    rankings = []
    for metric, (label, direction) in METRICS.items():
        valid = [(m, v) for m, v in metric_means[metric].items() if np.isfinite(v)]
        valid.sort(key=lambda z: z[1], reverse=(direction == "higher"))
        for rank, (method, value) in enumerate(valid, 1):
            rankings.append({
                "scenario": scenario,
                "metric": metric,
                "metric_label": label,
                "direction": direction,
                "rank": rank,
                "method": method,
                "mean": value,
            })
    write_csv(derived / "ranking_by_metric.csv",
              ["scenario", "metric", "metric_label", "direction", "rank", "method", "mean"], rankings)

    # Primary evidence: V4 variants vs classical baselines for the two preregistered primary metrics.
    evidence = []
    for r in pairwise:
        focal = r.get("focal_method", "")
        baseline = r.get("baseline_method", "")
        metric = r.get("metric", "")
        if not focal.startswith("calibrador_v4_") or baseline not in CLASSICAL or metric not in {"val_loss_aulc", "final_val_loss"}:
            continue
        low, high = fnum(r.get("ci95_low")), fnum(r.get("ci95_high"))
        t_holm, w_holm = fnum(r.get("paired_t_p_holm")), fnum(r.get("wilcoxon_p_holm"))
        significant = ((np.isfinite(t_holm) and t_holm < 0.05) or
                       (np.isfinite(w_holm) and w_holm < 0.05))
        ci_positive = np.isfinite(low) and low > 0
        evidence.append({
            "scenario": scenario,
            "focal_method": focal,
            "baseline": baseline,
            "metric": metric,
            "n_pairs": integer(r.get("n_pairs")),
            "focal_mean": fnum(r.get("focal_mean")),
            "baseline_mean": fnum(r.get("baseline_mean")),
            "mean_improvement": fnum(r.get("mean_improvement")),
            "ci95_low": low,
            "ci95_high": high,
            "wins": integer(r.get("wins")),
            "losses": integer(r.get("losses")),
            "ties": integer(r.get("ties")),
            "win_rate": fnum(r.get("win_rate")),
            "paired_t_p_holm": t_holm,
            "wilcoxon_p_holm": w_holm,
            "rank_biserial_improvement": fnum(r.get("rank_biserial_improvement")),
            "ci_strictly_positive": ci_positive,
            "holm_significant": significant,
        })
    ev_fields = list(evidence[0].keys()) if evidence else []
    if evidence:
        write_csv(derived / "v4_primary_evidence.csv", ev_fields, evidence)

    # CI chart for Loss AULC. Positive means V4 is better by benchmark convention.
    aulc = [e for e in evidence if e["metric"] == "val_loss_aulc"]
    if aulc:
        labels = [f'{e["focal_method"].replace("calibrador_v4_", "V4-").title()} vs {e["baseline"].title()}' for e in aulc]
        means = np.asarray([e["mean_improvement"] for e in aulc], dtype=float)
        lows = np.asarray([e["ci95_low"] for e in aulc], dtype=float)
        highs = np.asarray([e["ci95_high"] for e in aulc], dtype=float)
        y = np.arange(len(aulc))
        fig, ax = plt.subplots(figsize=(8, max(3.5, .55 * len(aulc) + 1.5)))
        ax.errorbar(means, y, xerr=np.vstack([means-lows, highs-means]), fmt='o', capsize=4)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Mean Loss AULC improvement (positive = V4 better)")
        ax.set_title(f"{scenario}: paired Loss AULC improvement with 95% CI")
        ax.grid(axis='x', alpha=.25)
        fig.tight_layout()
        fig.savefig(figures / "loss_aulc_pairwise_ci.png", dpi=180)
        plt.close(fig)

    # Human-readable scenario note generated strictly from computed tables.
    loss_rank = [r for r in rankings if r["metric"] == "val_loss_aulc"]
    best_loss = loss_rank[0] if loss_rank else None
    md = [f"# Derived results — {scenario}", "",
          "These files are generated from the confirmatory benchmark outputs; no retraining or retuning is performed.", ""]
    if best_loss:
        md += [f"- Lowest mean Loss AULC: **{best_loss['method']}** ({best_loss['mean']:.6f})."]
    successes = [e for e in evidence if e["metric"] == "val_loss_aulc" and e["holm_significant"] and e["mean_improvement"] > 0]
    if successes:
        md.append("- Holm-significant V4 Loss AULC gains: " + "; ".join(
            f"{e['focal_method']} vs {e['baseline']} ({e['wins']}/{e['n_pairs']} wins, Δ={e['mean_improvement']:.6f})"
            for e in successes))
    else:
        md.append("- No V4 Loss AULC comparison reached Holm-corrected significance in this scenario.")
    md += ["", "See `compact_summary.csv`, `ranking_by_metric.csv`, `v4_primary_evidence.csv`, and the figure in `../figures/`.", ""]
    (derived / "results_summary.md").write_text("\n".join(md), encoding="utf-8")

    return compact, rankings, evidence


def aggregate(all_compact, all_rankings, all_evidence):
    adir = RESULTS / "aggregate"
    figs = adir / "figures"
    adir.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    # One row per scenario/method for compact cross-scenario review.
    fields = list(all_compact[0].keys()) if all_compact else []
    if all_compact:
        write_csv(adir / "all_scenarios_compact_summary.csv", fields, all_compact)
    if all_evidence:
        write_csv(adir / "v4_vs_classical_primary_evidence.csv", list(all_evidence[0].keys()), all_evidence)

    # Average rank per method on each metric; rank is computed only within scenario.
    by_metric_method = defaultdict(list)
    for r in all_rankings:
        by_metric_method[(r["metric"], r["method"])].append(r["rank"])
    avg_rank_rows = []
    for (metric, method), ranks in sorted(by_metric_method.items()):
        avg_rank_rows.append({
            "metric": metric,
            "metric_label": METRICS[metric][0],
            "method": method,
            "scenarios_present": len(ranks),
            "average_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
        })
    write_csv(adir / "average_rank_across_scenarios.csv",
              ["metric", "metric_label", "method", "scenarios_present", "average_rank", "median_rank", "best_rank", "worst_rank"], avg_rank_rows)

    # Evidence matrix: count comparisons with positive means / high win rate / significance.
    matrix = defaultdict(lambda: {"comparisons":0, "positive_mean":0, "win_rate_ge_0_8":0,
                                  "win_rate_ge_0_9":0, "ci_positive":0, "holm_significant":0})
    for e in all_evidence:
        if e["metric"] != "val_loss_aulc":
            continue
        k = e["focal_method"]
        m = matrix[k]
        m["comparisons"] += 1
        m["positive_mean"] += int(e["mean_improvement"] > 0)
        m["win_rate_ge_0_8"] += int(np.isfinite(e["win_rate"]) and e["win_rate"] >= .8)
        m["win_rate_ge_0_9"] += int(np.isfinite(e["win_rate"]) and e["win_rate"] >= .9)
        m["ci_positive"] += int(e["ci_strictly_positive"])
        m["holm_significant"] += int(e["holm_significant"] and e["mean_improvement"] > 0)
    matrix_rows = [{"focal_method": k, **v} for k, v in sorted(matrix.items())]
    if matrix_rows:
        write_csv(adir / "loss_aulc_evidence_counts.csv", list(matrix_rows[0].keys()), matrix_rows)

    # Average rank chart for Loss AULC, only methods present in >=2 scenarios to avoid misleading singleton ranks.
    chart_rows = [r for r in avg_rank_rows if r["metric"] == "val_loss_aulc" and r["scenarios_present"] >= 2]
    chart_rows.sort(key=lambda r: r["average_rank"])
    if chart_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [r["method"] for r in chart_rows]
        vals = [r["average_rank"] for r in chart_rows]
        ax.bar(labels, vals)
        ax.set_ylabel("Average within-scenario rank (lower is better)")
        ax.set_title("Loss AULC average rank across completed scenarios")
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.25)
        fig.tight_layout()
        fig.savefig(figs / "loss_aulc_average_rank.png", dpi=180)
        plt.close(fig)

    # Win-rate chart for Symbolic against classical baselines across scenarios.
    sym = [e for e in all_evidence if e["metric"] == "val_loss_aulc" and e["focal_method"] == "calibrador_v4_symbolic"]
    if sym:
        sym.sort(key=lambda e: (e["scenario"], e["baseline"]))
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f'{e["scenario"]}\nvs {e["baseline"]}' for e in sym]
        vals = [100*e["win_rate"] if np.isfinite(e["win_rate"]) else math.nan for e in sym]
        ax.bar(labels, vals)
        ax.axhline(90, linestyle='--', linewidth=1)
        ax.set_ylabel("Paired win rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("V4-Symbolic Loss AULC paired win rates")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=.25)
        fig.tight_layout()
        fig.savefig(figs / "v4_symbolic_loss_aulc_win_rates.png", dpi=180)
        plt.close(fig)

    # Aggregate README.
    lines = ["# Aggregate confirmatory results", "",
             "Generated automatically from the completed scenario result folders.", "",
             "Completed scenarios: MNIST+ReLU, Fashion-MNIST+SiLU, CIFAR-10+GELU.",
             "SVHN+Softplus is intentionally excluded until its confirmatory benchmark is available.", "",
             "## Files", "",
             "- `all_scenarios_compact_summary.csv`: descriptive statistics by scenario/method.",
             "- `v4_vs_classical_primary_evidence.csv`: paired primary-metric evidence for V4 vs He/Xavier/LeCun.",
             "- `average_rank_across_scenarios.csv`: within-scenario ranks aggregated across scenarios.",
             "- `loss_aulc_evidence_counts.csv`: counts of positive, high-win-rate, CI-positive and significant comparisons.",
             "- `figures/`: cross-scenario visual summaries.", "",
             "Important: cross-scenario ranks summarize relative position only; raw loss values are not averaged across different datasets.", ""]
    (adir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    all_compact, all_rankings, all_evidence = [], [], []
    for scenario in SCENARIOS:
        compact, rankings, evidence = scenario_analysis(scenario)
        all_compact.extend(compact)
        all_rankings.extend(rankings)
        all_evidence.extend(evidence)
    aggregate(all_compact, all_rankings, all_evidence)
    print("Derived analyses generated successfully.")


if __name__ == "__main__":
    main()
