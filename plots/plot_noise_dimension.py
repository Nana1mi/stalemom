"""Plot noise dimension and n_good effects on zero-crossing boundary."""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    csv_summary = "results/noise_dimension/noise_dimension_summary.csv"
    csv_crossings = "results/noise_dimension/noise_dimension_crossings.csv"

    if not os.path.exists(csv_summary):
        print(f"File not found: {csv_summary}")
        return

    data = []
    with open(csv_summary, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append({
                "dim": r["dimension"],
                "B": int(r["B"]),
                "n_good": int(r["n_good"]),
                "rho_lam": float(r["rho_lam"]),
                "beta": float(r["beta"]),
                "stoch_effect": float(r["stoch_effect"]),
                "stoch_se": float(r["stoch_se"]),
                "mean_kl_pq": float(r["mean_kl_pq"]),
            })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)

    # Panel 1: Effect vs rho_lam across B
    ax1 = axes[0]
    b_vals = [4, 8, 16, 32, 64, 128]
    cmap_b = plt.cm.viridis(np.linspace(0.1, 0.9, len(b_vals)))

    for B, color in zip(b_vals, cmap_b):
        sub = [d for d in data if d["dim"] == "batch_size" and d["B"] == B]
        sub.sort(key=lambda d: d["rho_lam"])
        rhos = [d["rho_lam"] for d in sub]
        effs = [d["stoch_effect"] for d in sub]
        ses = [d["stoch_se"] for d in sub]
        ax1.errorbar(rhos, effs, yerr=ses, fmt="o-", color=color, label=f"B = {B}", capsize=3, lw=1.5)

    ax1.axhline(0, color="black", ls="--", alpha=0.7)
    ax1.axvline(0.25, color="gray", ls=":", alpha=0.7, label=r"Safe Bound $\rho=0.25$")
    ax1.set_xscale("log")
    ax1.set_title(r"Paired Effect vs $\rho\lambda$ across Batch Size $B$ ($n_{\mathrm{good}}=2$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"$\rho\lambda = \eta\lambda / (1 - \beta)$", fontsize=10)
    ax1.set_ylabel(r"Paired Effect $\Delta = \mathrm{TIS} - \mathrm{TIS}_{\mathrm{deb}}$", fontsize=10)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(fontsize=8.5, loc="lower left")

    # Panel 2: Effect vs rho_lam across n_good
    ax2 = axes[1]
    ng_vals = [2, 3, 5]
    colors_ng = {2: "#1f77b4", 3: "#ff7f0e", 5: "#d62728"}

    for ng in ng_vals:
        sub = [d for d in data if d["n_good"] == ng and d["B"] == 32]
        sub.sort(key=lambda d: d["rho_lam"])
        rhos = [d["rho_lam"] for d in sub]
        effs = [d["stoch_effect"] for d in sub]
        ses = [d["stoch_se"] for d in sub]
        ax2.errorbar(rhos, effs, yerr=ses, fmt="s-", color=colors_ng[ng], label=rf"$n_{{\mathrm{{good}}}} = {ng}$", capsize=3, lw=1.5)

    ax2.axhline(0, color="black", ls="--", alpha=0.7)
    ax2.axvline(0.25, color="gray", ls=":", alpha=0.7, label=r"Safe Bound $\rho=0.25$")
    ax2.set_xscale("log")
    ax2.set_title(r"Paired Effect vs $\rho\lambda$ across Reward Difficulty $n_{\mathrm{good}}$ ($B=32$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"$\rho\lambda = \eta\lambda / (1 - \beta)$", fontsize=10)
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(fontsize=9, loc="lower left")

    # Panel 3: Effect vs Policy Shift mean_kl_pq
    ax3 = axes[2]
    for d in data:
        marker = "o" if d["dim"] == "batch_size" else "s"
        c = "blue" if d["stoch_effect"] > 0 else "red"
        ax3.scatter(d["mean_kl_pq"], d["stoch_effect"], color=c, alpha=0.6, s=28, marker=marker)

    ax3.axhline(0, color="black", ls="--", alpha=0.7)
    ax3.set_title(r"Paired Effect vs Policy Shift $\mathrm{KL}(p_t \parallel q_t)$", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"Mean Policy Shift $\mathrm{KL}(p_t \parallel q_t)$", fontsize=10)
    ax3.set_ylabel(r"Paired Effect $\Delta = \mathrm{TIS} - \mathrm{TIS}_{\mathrm{deb}}$", fontsize=10)
    ax3.grid(True, ls=":", alpha=0.4)

    # Annotations on scatter
    ax3.scatter([], [], color="blue", label="Costly (+, safe/stable)")
    ax3.scatter([], [], color="red", label="Beneficial (-, overshoot/noisy)")
    ax3.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        r"Unified Mechanism: How Noise ($B$), Task Difficulty ($n_{\mathrm{good}}$), and Drift ($\mathrm{KL}$) Shift the Zero-Crossing",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_path = "results/noise_dimension/noise_and_difficulty_shift.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
