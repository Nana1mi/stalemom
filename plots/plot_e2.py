"""Generate plots for E2 causal debiasing experiment."""
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


def plot_e2(summary_csv: str, results_dir: str, out_dir: str):
    rows = load_summary(summary_csv)
    os.makedirs(out_dir, exist_ok=True)

    valid_rows = [r for r in rows if r["skipped"] != "True" and r["skipped"] != True]

    # Target settings for comparison: k=8, eta_lam=0.01, c=1.5
    mom_settings = [
        ("No momentum (rho=0)", lambda r: r["rho_lam"] != "" and abs(float(r["rho_lam"])) < 1e-5),
        ("Scaled momentum (rho=0.25)", lambda r: r["rho_lam"] != "" and abs(float(r["rho_lam"]) - 0.25) < 1e-4),
        ("Long memory (beta=1-1e-6)", lambda r: r["beta_fixed"] != "" and float(r["beta_fixed"]) > 0.99),
    ]

    estimators = ["exact", "is", "tis_mean", "tis", "tis_debiased"]
    colors = {
        "exact": "black",
        "is": "#2ca02c",
        "tis_mean": "#ff7f0e",
        "tis": "#d62728",
        "tis_debiased": "#9467bd",
    }
    labels = {
        "exact": "Exact NPG",
        "is": "Full IS (noise only)",
        "tis_mean": "TIS Mean (bias only)",
        "tis": "TIS (bias + noise)",
        "tis_debiased": "TIS Debiased (noise only)",
    }

    # 1. 5 paths gap vs t curves faceted by momentum
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=True)

    for ax, (mom_title, mom_cond) in zip(axes, mom_settings):
        for est in estimators:
            matching = [
                r for r in valid_rows
                if r["estimator"] == est
                and int(r["k"]) in (8, 0 if est in ("exact", "onpolicy") else 8)
                and abs(float(r["eta_lam"]) - 0.01) < 1e-4
                and mom_cond(r)
            ]
            if not matching:
                continue
            npz_file = matching[0]["npz_file"]
            npz_path = os.path.join(results_dir, npz_file)
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            gap = np.nan_to_num(data["gap"], nan=1.0)
            median_gap = np.median(gap, axis=0)
            q25 = np.percentile(gap, 25, axis=0)
            q75 = np.percentile(gap, 75, axis=0)
            t_axis = np.arange(len(median_gap))

            line = ax.plot(t_axis, median_gap, color=colors[est], label=labels[est], lw=1.8)[0]
            if est not in ("exact", "tis_mean"):
                ax.fill_between(t_axis, q25, q75, color=colors[est], alpha=0.12)

        ax.set_title(mom_title)
        ax.set_xlabel("Step $t$")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Normalized Gap")
    fig.suptitle(r"E2: Causal Debiasing Comparison Across 5 Paths ($k=8, \eta\lambda=0.01$)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e2_paths_gap.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved E2 paths gap plot.")

    # 2. Decomposition inner products vs t for TIS under long memory vs rho=0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax_idx, (mom_title, mom_cond) in enumerate([
        ("Scaled Momentum (rho=0.25)", lambda r: r["rho_lam"] != "" and abs(float(r["rho_lam"]) - 0.25) < 1e-4),
        ("Long Memory (beta=1-1e-6)", lambda r: r["beta_fixed"] != "" and float(r["beta_fixed"]) > 0.99),
    ]):
        ax = axes[ax_idx]
        matching = [
            r for r in valid_rows
            if r["estimator"] == "tis"
            and int(r["k"]) == 8
            and abs(float(r["eta_lam"]) - 0.01) < 1e-4
            and mom_cond(r)
        ]
        if not matching:
            continue
        npz_file = matching[0]["npz_file"]
        data = np.load(os.path.join(results_dir, npz_file))
        diag_t = data["diag_t"]

        inner_track = np.mean(data["inner_track"], axis=0)
        inner_bias = np.mean(data["inner_bias_hist"], axis=0)
        inner_noise = np.mean(data["inner_noise"], axis=0)
        inner_update = np.mean(data["inner_update"], axis=0)

        ax.plot(diag_t, inner_track, "o-", label=r"Tracking error: $\langle \nabla J, \hat{E}^h - h \rangle$")
        ax.plot(diag_t, inner_bias, "s-", label=r"History bias: $\langle \nabla J, \hat{E}^b \rangle$")
        ax.plot(diag_t, inner_noise, "^-", label=r"Smoothed noise: $\langle \nabla J, \hat{E}^\xi \rangle$")
        ax.plot(diag_t, inner_update, "k--", alpha=0.7, label=r"Total update: $\langle \nabla J, \hat{m} \rangle$")

        ax.axhline(0.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Diagnostic step $t$")
        ax.set_ylabel(r"Inner Product with $\nabla J(\theta_t)$")
        ax.set_title(mom_title)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(r"E2: Momentum Error Decomposition Inner Products for TIS ($k=8, \eta\lambda=0.01$)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e2_decomposition_inner_products.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved E2 decomposition inner products plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/e2/summary.csv")
    parser.add_argument("--results", type=str, default="results/e2/")
    parser.add_argument("--out", type=str, default="results/e2/")
    args = parser.parse_args()
    plot_e2(args.summary, args.results, args.out)


if __name__ == "__main__":
    main()
