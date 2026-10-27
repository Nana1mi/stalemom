"""Plot zero-crossing experiment results:
Shows the paired effect (tis - tis_debiased) +/- SE versus rho*lambda.
"""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    csv_path = "results/zero_crossing/zero_crossing_summary.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "k": int(row["k"]),
                "eta_lam": float(row["eta_lam"]),
                "rho_lam": float(row["rho_lam"]),
                "det_effect": float(row["det_effect"]),
                "stoch_effect": float(row["stoch_effect"]),
                "stoch_se": float(row["stoch_se"]),
            })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    eta_lams = [0.03, 0.1, 0.3]
    ks = [4, 8, 12, 16]
    colors = {4: "#1f77b4", 8: "#ff7f0e", 12: "#2ca02c", 16: "#d62728"}

    for ax, eta_lam in zip(axes, eta_lams):
        for k in ks:
            sub = [d for d in data if d["k"] == k and abs(d["eta_lam"] - eta_lam) < 1e-5]
            if not sub:
                continue
            sub.sort(key=lambda d: d["rho_lam"])
            rhos = [d["rho_lam"] for d in sub]
            effects = [d["stoch_effect"] for d in sub]
            ses = [d["stoch_se"] for d in sub]

            ax.errorbar(
                rhos,
                effects,
                yerr=ses,
                fmt="o-",
                color=colors[k],
                label=f"k={k}",
                capsize=3,
                markersize=4,
                linewidth=1.2,
            )

        ax.axhline(0, color="black", linestyle="--", alpha=0.7, linewidth=1.0)
        ax.axvline(0.25, color="gray", linestyle=":", alpha=0.7, label=r"Safe bound $\rho=1/(4\lambda)=0.25$")
        ax.set_xscale("log")
        ax.set_title(rf"$\eta\lambda = {eta_lam}$", fontsize=12)
        ax.set_xlabel(r"$\rho\lambda = \eta\lambda / (1 - \beta)$", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(r"Paired Effect $\Delta = \mathrm{TIS} - \mathrm{TIS}_{\mathrm{deb}}$", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower left")

    plt.suptitle(
        r"Zero-Crossing Localization: Truncation Bias Effect vs Effective Stepsize $\rho\lambda$",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = "results/zero_crossing/zero_crossing_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
