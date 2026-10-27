"""Plot stepsize control results comparing exact, is, tis_mean, and tis across eta."""
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_stepsize(csv_path: str, out_dir: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    os.makedirs(out_dir, exist_ok=True)
    betas = [0.9, 0.99]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150, sharey=True)

    colors = {
        "exact": "black",
        "is": "#2ca02c",
        "tis_mean": "#ff7f0e",
        "tis": "#d62728",
    }
    labels = {
        "exact": "Exact NPG",
        "is": "Full IS",
        "tis_mean": "TIS Mean",
        "tis": "TIS",
    }

    for ax, b in zip(axes, betas):
        sub = [r for r in rows if abs(float(r["beta"]) - b) < 1e-4]
        
        for est in ["exact", "is", "tis_mean", "tis"]:
            est_sub = [r for r in sub if r["estimator"] == est]
            est_sub.sort(key=lambda r: float(r["eta"]))
            
            etas = [float(r["eta"]) for r in est_sub]
            a_t = [float(r["A_T_pessimistic"]) for r in est_sub]
            se = [float(r["A_T_pessimistic_se"]) if r["A_T_pessimistic_se"] != "" else 0.0 for r in est_sub]
            
            ax.errorbar(
                etas, a_t, yerr=se, fmt="o-",
                color=colors[est], label=labels[est],
                capsize=3, lw=1.6
            )

        # Highlight eta=0.1
        ax.axvline(0.1, color="gray", ls=":", alpha=0.7, label=r"Reference $\eta=0.1$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_title(rf"$\beta = {b}$")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"Average normalized gap $A_T$ (lower is better)")
    fig.suptitle(r"Stepsize Control: $A_T$ vs $\eta$ across Estimators ($k=16, c=1.2$)", fontsize=14, y=1.02)
    fig.tight_layout()
    
    out_fig = os.path.join(out_dir, "stepsize_control.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stepsize control plot to {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/stepsize_control/stepsize_summary.csv")
    parser.add_argument("--out", type=str, default="results/stepsize_control/")
    args = parser.parse_args()
    plot_stepsize(args.csv, args.out)


if __name__ == "__main__":
    main()
