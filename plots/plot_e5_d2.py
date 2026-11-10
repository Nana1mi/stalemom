"""Plot SPEC E5 Phase D2 results:
1. Individual "beta vs tuned A_T" figure for every (estimator, k, B) pair (§4).
2. Combined 2x2 overview figure comparing estimators across (k, B).
"""
import argparse
import csv
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/e5/d2/tuned_momentum.csv")
    parser.add_argument("--out", type=str, default="results/e5/d2")
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_best_eta"] == "True":
                rows.append({
                    "estimator": r["estimator"],
                    "k": int(r["k"]),
                    "B": int(r["B"]),
                    "beta": float(r["beta"]),
                    "best_eta": float(r["best_eta"]),
                    "A_T": float(r["best_A_T"]),
                    "A_T_se": float(r["best_A_T_se"]),
                    "final_gap": float(r["best_final_gap"]),
                    "final_gap_se": float(r["best_final_gap_se"]),
                })

    indiv_dir = os.path.join(args.out, "plots_by_condition")
    os.makedirs(indiv_dir, exist_ok=True)

    # 1. Individual plot for each (estimator, k, B) as required by §4
    rows.sort(key=lambda r: (r["estimator"], r["k"], r["B"], r["beta"]))
    for (est, k, B), grp in itertools.groupby(rows, key=lambda r: (r["estimator"], r["k"], r["B"])):
        items = sorted(list(grp), key=lambda r: r["beta"])
        betas = [r["beta"] for r in items]
        x = np.arange(len(betas))
        a_ts = [r["A_T"] for r in items]
        ses = [r["A_T_se"] for r in items]
        etas = [r["best_eta"] for r in items]

        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
        ax.errorbar(x, a_ts, yerr=ses, fmt="o-", color="#1f77b4", lw=2.0, capsize=4, markersize=6)
        for xi, yi, sei, etai in zip(x, a_ts, ses, etas):
            ax.annotate(
                rf"$\eta^*={etai:.3g}$" + "\n" + f"{yi:.4f}",
                xy=(xi, yi + sei),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in betas])
        ax.set_xlabel(r"Momentum $\beta$", fontsize=11)
        ax.set_ylabel(r"Tuned Average Gap $A_T \pm \mathrm{SE}$", fontsize=11)
        ax.set_title(f"D2 Tuned Momentum: {est} (k={k}, B={B})", fontsize=12, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        y_span = max(a_ts) - min(a_ts) + 0.01
        ax.set_ylim(max(0.0, min(a_ts) - 0.15 * y_span), max(a_ts) + 0.35 * y_span)
        fig.tight_layout()

        fname = f"d2_{est}_k{k}_B{B}.png"
        fig.savefig(os.path.join(indiv_dir, fname), bbox_inches="tight")
        plt.close(fig)

    # 2. Combined 2x2 overview across (k, B)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    kb_pairs = [(0, 4), (0, 32), (16, 4), (16, 32)]
    colors = {
        "onpolicy": "#1f77b4",
        "is": "#2ca02c",
        "tis": "#d62728",
        "tis_debiased": "#9467bd",
    }
    markers = {
        "onpolicy": "o",
        "is": "s",
        "tis": "^",
        "tis_debiased": "D",
    }

    for ax, (k, B) in zip(axes.ravel(), kb_pairs):
        for est in ["onpolicy", "is", "tis", "tis_debiased"]:
            sub = [r for r in rows if r["estimator"] == est and r["k"] == k and r["B"] == B]
            sub.sort(key=lambda r: r["beta"])
            x = np.arange(len(sub))
            a_ts = [r["A_T"] for r in sub]
            ses = [r["A_T_se"] for r in sub]
            ax.errorbar(
                x,
                a_ts,
                yerr=ses,
                fmt=f"{markers[est]}-",
                color=colors[est],
                label=est,
                lw=1.8,
                capsize=3,
                markersize=5,
            )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0.0", "0.6", "0.9", "0.99"])
        ax.set_xlabel(r"Momentum $\beta$", fontsize=10.5)
        ax.set_ylabel(r"Tuned $A_T \pm \mathrm{SE}$ (lower is better)", fontsize=10.5)
        ax.set_title(f"Staleness k={k}, Batch Size B={B}", fontsize=11.5, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle(
        r"SPEC E5 Phase D2: Tuned Momentum Comparison $A_T^*(\beta)$ Across Estimators ($R=200$)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    overview_path = os.path.join(args.out, "d2_tuned_momentum_overview.png")
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 16 individual plots to {indiv_dir} and overview to {overview_path}")


if __name__ == "__main__":
    main()
