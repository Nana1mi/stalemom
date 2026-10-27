"""Generate plots for E1 grid experiments."""
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def load_summary(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def momentum_label(row: dict) -> str:
    if row["rho_lam"] != "":
        val = float(row["rho_lam"])
        if val == 0.0:
            return "No mom (rho=0)"
        return rf"rho={val:g}"
    elif row["beta_fixed"] != "":
        val = float(row["beta_fixed"])
        if val > 0.999:
            return r"beta=1-1e-6"
        return rf"beta={val:g}"
    return "Unknown"


def plot_e1(summary_csv: str, results_dir: str, out_dir: str):
    rows = load_summary(summary_csv)
    os.makedirs(out_dir, exist_ok=True)

    valid_rows = [r for r in rows if r["skipped"] != "True" and r["skipped"] != True]

    # 1. Truncation rate heatmaps for tis
    tis_rows = [r for r in valid_rows if r["estimator"] == "tis"]
    cs = sorted(list(set(float(r["c"]) for r in tis_rows if r["c"] != "")))
    ks = sorted(list(set(int(r["k"]) for r in tis_rows)))
    eta_lams = sorted(list(set(float(r["eta_lam"]) for r in tis_rows)))

    if cs:
        fig, axes = plt.subplots(1, len(cs), figsize=(5 * len(cs), 4), dpi=150, sharey=True)
        if len(cs) == 1:
            axes = [axes]

        for ax, c_val in zip(axes, cs):
            matrix = np.zeros((len(ks), len(eta_lams)))
            for i, k_val in enumerate(ks):
                for j, eta_val in enumerate(eta_lams):
                    matching = [
                        float(r["mean_trunc_rate"])
                        for r in tis_rows
                        if abs(float(r["c"]) - c_val) < 1e-5
                        and int(r["k"]) == k_val
                        and abs(float(r["eta_lam"]) - eta_val) < 1e-6
                    ]
                    matrix[i, j] = np.mean(matching) if matching else 0.0

            im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.1, matrix.max()))
            ax.set_xticks(range(len(eta_lams)))
            ax.set_xticklabels([f"{e:g}" for e in eta_lams], rotation=30)
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels(ks)
            ax.set_xlabel(r"$\eta\lambda$")
            ax.set_title(f"Truncation Rate (c = {c_val:g})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_ylabel("Staleness lag $k$")
        fig.suptitle("E1: Mean Truncation Rate Heatmaps", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "e1_truncation_heatmaps.png"), bbox_inches="tight")
        plt.close(fig)
        print("Saved truncation heatmaps.")

    # 2. Gain region / Optimal A_T per momentum setting
    all_mom_labels = sorted(list(set(momentum_label(r) for r in valid_rows)))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    for ax_idx, (est, c_target, title) in enumerate([("is", None, "Full IS (no truncation)"), ("tis", 1.5, "TIS (c = 1.5)")]):
        ax = axes[ax_idx]
        subset = [
            r for r in valid_rows
            if r["estimator"] == est and (c_target is None or abs(float(r["c"]) - c_target) < 1e-4)
        ]
        
        x = np.arange(len(ks))
        width = 0.8 / len(all_mom_labels)
        
        for m_idx, m_label in enumerate(all_mom_labels):
            best_a_t = []
            best_se = []
            for k_val in ks:
                matching = [
                    r for r in subset
                    if int(r["k"]) == k_val and momentum_label(r) == m_label and r["A_T_pessimistic"] != ""
                ]
                if matching:
                    best_row = min(matching, key=lambda r: float(r["A_T_pessimistic"]))
                    best_a_t.append(float(best_row["A_T_pessimistic"]))
                    best_se.append(float(best_row["A_T_pessimistic_se"]))
                else:
                    best_a_t.append(np.nan)
                    best_se.append(0.0)
                    
            offset = (m_idx - len(all_mom_labels) / 2 + 0.5) * width
            ax.bar(x + offset, best_a_t, width, yerr=best_se, capsize=2, label=m_label)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_xlabel("Staleness lag $k$")
        ax.set_ylabel(r"Optimal $A_T$ (lower is better)")
        ax.set_title(title)
        ax.grid(True, axis="y", ls="--", alpha=0.5)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(r"E1: Optimal Normalized Gap $A_T$ across Momentum Settings", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e1_momentum_gain_regions.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved momentum gain regions plot.")

    # 3. Gap vs t trajectory curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax_idx, est in enumerate(["is", "tis"]):
        ax = axes[ax_idx]
        c_req = 1.5 if est == "tis" else None
        
        for m_label in ["No mom (rho=0)", "rho=0.25", "beta=0.9", "beta=0.99"]:
            match = [
                r for r in valid_rows
                if r["estimator"] == est
                and int(r["k"]) == 8
                and abs(float(r["eta_lam"]) - 0.01) < 1e-6
                and momentum_label(r) == m_label
                and (c_req is None or abs(float(r["c"]) - c_req) < 1e-4)
            ]
            if not match:
                continue
            npz_file = match[0]["npz_file"]
            npz_path = os.path.join(results_dir, npz_file)
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            gap = data["gap"]
            gap_clean = np.nan_to_num(gap, nan=1.0)
            median_gap = np.median(gap_clean, axis=0)
            q25 = np.percentile(gap_clean, 25, axis=0)
            q75 = np.percentile(gap_clean, 75, axis=0)
            t_axis = np.arange(len(median_gap))

            line = ax.plot(t_axis, median_gap, label=m_label)[0]
            ax.fill_between(t_axis, q25, q75, color=line.get_color(), alpha=0.15)

        ax.set_xlabel("Step $t$")
        ax.set_ylabel("Normalized Gap")
        ax.set_title(f"{est.upper()} (k=8, $\eta\lambda=0.01$)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()

    fig.suptitle("E1: Gap vs Step $t$ (Median + IQR)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e1_gap_trajectories.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved gap trajectories plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/e1/summary.csv")
    parser.add_argument("--results", type=str, default="results/e1/")
    parser.add_argument("--out", type=str, default="results/e1/")
    args = parser.parse_args()
    plot_e1(args.summary, args.results, args.out)


if __name__ == "__main__":
    main()
