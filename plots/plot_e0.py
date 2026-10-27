"""Generate plots for E0-R replication."""
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_e0(csv_path: str, out_dir: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    os.makedirs(out_dir, exist_ok=True)
    inits = sorted(list(set(r["init"] for r in rows)))

    for init_type in inits:
        sub = [r for r in rows if r["init"] == init_type]
        sub.sort(key=lambda x: float(x["eta"]))

        etas = np.array([float(r["eta"]) for r in sub])
        mean_fixed = np.array([float(r["mean_fixed"]) for r in sub])
        mean_scaled = np.array([float(r["mean_scaled"]) for r in sub])
        std_fixed = np.array([float(r["std_fixed"]) for r in sub])
        std_scaled = np.array([float(r["std_scaled"]) for r in sub])
        snr_fixed = np.array([float(r["snr_fixed"]) for r in sub])
        snr_scaled = np.array([float(r["snr_scaled"]) for r in sub])
        win_fixed = np.array([float(r["win_fixed"]) for r in sub])
        win_scaled = np.array([float(r["win_scaled"]) for r in sub])

        fig, axes = plt.subplots(1, 4, figsize=(18, 4), dpi=150)

        # 1. Mean gain vs eta
        ax = axes[0]
        # For log-log, only positive values can be plotted; clip or take absolute for display if needed
        pos_fixed = mean_fixed > 0
        pos_scaled = mean_scaled > 0
        ax.plot(etas[pos_fixed], mean_fixed[pos_fixed], "o-", label=r"Fixed $\beta=0.9$")
        ax.plot(etas[pos_scaled], mean_scaled[pos_scaled], "s-", label=r"Scaled $\beta=1-4\eta$")
        # Reference O(eta) line for fixed beta
        ref_eta = mean_fixed[-1] * (etas / etas[-1])
        ax.plot(etas, ref_eta, "k:", alpha=0.5, label=r"Ref $\propto \eta$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_ylabel(r"Mean gain $\mathbb{E}[J^\beta - J^0]$")
        ax.set_title("Mean Gain vs $\eta$ (log-log)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        # 2. Std vs eta
        ax = axes[1]
        ax.plot(etas, std_fixed, "o-", label=r"Fixed $\beta=0.9$")
        ax.plot(etas, std_scaled, "s-", label=r"Scaled $\beta=1-4\eta$")
        ref_std = std_scaled[0] * np.sqrt(etas / etas[0])
        ax.plot(etas, ref_std, "k--", alpha=0.6, label=r"Ref $\propto \sqrt{\eta}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_ylabel(r"Std $\sigma[J^\beta - J^0]$")
        ax.set_title("Std vs $\eta$ (log-log)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        # 3. SNR vs eta
        ax = axes[2]
        ax.plot(etas, snr_fixed, "o-", label=r"Fixed $\beta=0.9$")
        ax.plot(etas, snr_scaled, "s-", label=r"Scaled $\beta=1-4\eta$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_ylabel(r"SNR $|\mu| / \sigma$")
        ax.set_title("SNR vs $\eta$ (log-log)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        # 4. Win rate vs eta
        ax = axes[3]
        ax.plot(etas, win_fixed, "o-", label=r"Fixed $\beta=0.9$")
        ax.plot(etas, win_scaled, "s-", label=r"Scaled $\beta=1-4\eta$")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.7, label="Tie (0.5)")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_ylabel(r"Win rate $\mathbb{P}(J^\beta > J^0)$")
        ax.set_title("Win Rate vs $\eta$ (semilog-x)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        fig.suptitle(f"E0-R Replication: Synchronous Momentum ({init_type} init)", fontsize=14, y=1.02)
        fig.tight_layout()

        out_fig = os.path.join(out_dir, f"e0_repro_{init_type}.png")
        fig.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/e0_repro/e0_summary.csv")
    parser.add_argument("--out", type=str, default="results/e0_repro/")
    args = parser.parse_args()
    plot_e0(args.csv, args.out)


if __name__ == "__main__":
    main()
