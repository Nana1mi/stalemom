"""Plot SPEC E5 Phase D1 results:
1. Estimator parity validation: H_hat vs H_exact_mean, C2_hat vs C2_exact_mean.
2. C1 (single-batch cosine) vs C2_exact_h, C2_exact_mean, and C2_hat across beta at t=16, 64.
"""
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/e5/d1/summary.csv")
    parser.add_argument("--out", type=str, default="results/e5/d1")
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    os.makedirs(args.out, exist_ok=True)

    # 1. Parity plot: H_hat vs H_exact_mean and C2_hat vs C2_exact_mean (for t=16 where SNR > 0.08)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    t16 = [r for r in rows if int(r["t"]) == 16]

    colors = {"onpolicy": "#1f77b4", "is": "#2ca02c", "tis": "#d62728"}
    for est in ["onpolicy", "is", "tis"]:
        sub = [r for r in t16 if r["estimator"] == est]
        hx = [float(r["H_exact_mean_mean"]) for r in sub]
        hy = [float(r["H_hat_mean"]) for r in sub]
        hse = [float(r["H_hat_se"]) for r in sub]
        ax1.errorbar(hx, hy, yerr=hse, fmt="o", color=colors[est], label=est, alpha=0.8, capsize=2.5)

        cx = [float(r["C2_exact_mean_mean"]) for r in sub]
        cy = [float(r["C2_hat_mean"]) for r in sub]
        cse = [float(r["C2_hat_se"]) for r in sub]
        ax2.errorbar(cx, cy, yerr=cse, fmt="o", color=colors[est], label=est, alpha=0.8, capsize=2.5)

    all_h = [float(r["H_exact_mean_mean"]) for r in t16]
    h_lim = [min(all_h) - 0.01, max(all_h) + 0.02]
    ax1.plot(h_lim, h_lim, "k--", alpha=0.6, label="y = x (exact)")
    ax1.set_xlabel(r"Exact $\|\mathbb{E}[g]\|^2$", fontsize=11)
    ax1.set_ylabel(r"Estimated $\hat{H} = \langle \bar{A}, \bar{B} \rangle \pm \mathrm{SE}$", fontsize=11)
    ax1.set_title(r"Unbiased Signal Norm Estimation ($\hat{H}$ vs Exact)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend()

    ax2.plot([-0.8, 1.0], [-0.8, 1.0], "k--", alpha=0.6, label="y = x (exact)")
    ax2.set_xlabel(r"Exact $C_{2,\mathrm{mean}} = \cos(m_{t-1}, h_t + b_t)$", fontsize=11)
    ax2.set_ylabel(r"Estimated $\hat{C}_2 \pm \mathrm{SE}$", fontsize=11)
    ax2.set_title(r"Mean-Gradient Cosine Recovery ($\hat{C}_2$ vs Exact)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend()

    fig.suptitle("SPEC E5 Phase D1: Estimator Validation on Frozen Bandit Checkpoints (t=16, R=200)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p1_path = os.path.join(args.out, "d1_estimator_validation.png")
    fig.savefig(p1_path, bbox_inches="tight")
    plt.close(fig)

    # 2. C1 vs C2 alignment across beta at t=16
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180, sharey=True)
    panels = [
        ("onpolicy", 0, 4, "On-Policy (k=0, B=4: High Noise)"),
        ("onpolicy", 0, 32, "On-Policy (k=0, B=32: Low Noise)"),
        ("tis", 16, 32, "Async TIS (k=16, B=32, c=1.2)"),
    ]

    for ax, (est, k, B, title) in zip(axes, panels):
        sub = [r for r in t16 if r["estimator"] == est and int(r["k"]) == k and int(r["B"]) == B]
        sub.sort(key=lambda r: float(r["beta"]))
        x = np.arange(len(sub))

        c1 = [float(r["C1_mean"]) for r in sub]
        c1_se = [float(r["C1_se"]) for r in sub]
        c2_h = [float(r["C2_exact_h_mean"]) for r in sub]
        c2_h_se = [float(r["C2_exact_h_se"]) for r in sub]
        c2_m = [float(r["C2_exact_mean_mean"]) for r in sub]
        c2_m_se = [float(r["C2_exact_mean_se"]) for r in sub]
        c2_hat = [float(r["C2_hat_mean"]) for r in sub]
        c2_hat_se = [float(r["C2_hat_se"]) for r in sub]

        ax.errorbar(x, c1, yerr=c1_se, fmt="s--", color="#d62728", lw=1.8, capsize=3, label=r"$C_1 = \cos(m_{t-1}, g_t)$ (Single-batch)")
        ax.errorbar(x, c2_h, yerr=c2_h_se, fmt="o-", color="#1f77b4", lw=1.8, capsize=3, label=r"$C_{2,\mathrm{exact}}(h_t) = \cos(m_{t-1}, h_t)$")
        if est == "tis" and k > 0:
            ax.errorbar(x, c2_m, yerr=c2_m_se, fmt="^-", color="#2ca02c", lw=1.8, capsize=3, label=r"$C_{2,\mathrm{exact}}(h_t+b_t)$")
        ax.errorbar(x, c2_hat, yerr=c2_hat_se, fmt="D:", color="#9467bd", lw=1.6, capsize=3, label=r"$\hat{C}_2$ (Unbiased split-half)")

        ax.axhline(0.0, color="black", ls=":", alpha=0.6)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0.0", "0.6", "0.9", "0.99"])
        ax.set_xlabel(r"Momentum $\beta$", fontsize=10.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=8.5, loc="best")

    axes[0].set_ylabel("Cosine Alignment", fontsize=11)
    fig.suptitle(r"SPEC E5 Phase D1: Single-Batch Cosine $C_1$ vs True Alignment $C_2$ at $t=16$ ($R=200$)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p2_path = os.path.join(args.out, "d1_c1_vs_c2_alignment.png")
    fig.savefig(p2_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved D1 plots to {p1_path} and {p2_path}")


if __name__ == "__main__":
    main()
