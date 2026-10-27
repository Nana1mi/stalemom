"""Plot robustness check results across dimensions:
Shows paired effect Delta = TIS - TIS_deb for beta=0.6 and beta=0.99.
"""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    csv_path = "results/robustness/robustness_summary.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "dim": row["dimension"],
                "var": row["variation"],
                "beta": float(row["beta"]),
                "stoch_effect": float(row["stoch_effect"]),
                "stoch_se": float(row["stoch_se"]),
            })

    # Group into categories:
    categories = [
        ("init_seed", "Initial Seeds (0-4)"),
        ("clipping_c", "Clipping Threshold c"),
        ("batch_size", "Batch Size B"),
        ("gradient_type", "Optimizer Type"),
        ("n_good", "Good Actions (n_good)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)

    for ax, (cat, title) in zip(axes, categories):
        sub = [d for d in data if d["dim"] == cat]
        # Get unique variations
        vars_unique = []
        for d in sub:
            if d["var"] not in vars_unique:
                vars_unique.append(d["var"])

        x = np.arange(len(vars_unique))
        width = 0.35

        eff_06 = [next(d["stoch_effect"] for d in sub if d["var"] == v and abs(d["beta"] - 0.6) < 1e-4) for v in vars_unique]
        se_06 = [next(d["stoch_se"] for d in sub if d["var"] == v and abs(d["beta"] - 0.6) < 1e-4) for v in vars_unique]

        eff_99 = [next(d["stoch_effect"] for d in sub if d["var"] == v and abs(d["beta"] - 0.99) < 1e-4) for v in vars_unique]
        se_99 = [next(d["stoch_se"] for d in sub if d["var"] == v and abs(d["beta"] - 0.99) < 1e-4) for v in vars_unique]

        ax.bar(x - width / 2, eff_06, width, yerr=se_06, capsize=3, label=r"$\beta=0.6$ (Safe)", color="#2b5c8f", alpha=0.85)
        ax.bar(x + width / 2, eff_99, width, yerr=se_99, capsize=3, label=r"$\beta=0.99$ (Overshoot)", color="#d95f02", alpha=0.85)

        ax.axhline(0, color="black", linestyle="--", alpha=0.7, linewidth=0.8)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        # Simplify labels
        clean_labels = [v.replace("init_seed=", "seed ").replace("clipping_c=", "c=").replace("batch_size=", "B=") for v in vars_unique]
        ax.set_xticklabels(clean_labels, rotation=35, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(r"Paired Effect $\Delta = \mathrm{TIS} - \mathrm{TIS}_{\mathrm{deb}}$", fontsize=10)
    axes[0].legend(fontsize=8.5, loc="lower left")

    plt.suptitle(
        r"Robustness of Truncation Bias Sign Flip Across Problem Settings",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = "results/robustness/robustness_bars.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
