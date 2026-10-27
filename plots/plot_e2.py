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

    # Target settings: k=16, eta_lam=0.1, c=1.2
    betas = [0.0, 0.6, 0.9, 0.99]
    beta_labels = [
        r"$\beta=0$ ($\rho=0.10$, safe)",
        r"$\beta=0.6$ ($\rho=0.25$, boundary)",
        r"$\beta=0.9$ ($\rho=1.0$, outside safe)",
        r"$\beta=0.99$ ($\rho=10$, overshoot)",
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
        "exact": "Exact NPG (no noise, no bias)",
        "is": "Full IS (noise only)",
        "tis_mean": "TIS Mean (bias only)",
        "tis": "TIS (bias + noise)",
        "tis_debiased": "TIS Debiased (noise only)",
    }

    # 1. 5 paths gap vs t curves faceted by beta
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=150, sharey=True)

    for ax, b_val, b_label in zip(axes, betas, beta_labels):
        for est in estimators:
            matching = [
                r for r in valid_rows
                if r["estimator"] == est
                and int(r["k"]) in (16, 0 if est in ("exact", "onpolicy") else 16)
                and abs(float(r["eta_lam"]) - 0.1) < 1e-4
                and r["beta_fixed"] != "" and abs(float(r["beta_fixed"]) - b_val) < 1e-4
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
                ax.fill_between(t_axis, q25, q75, color=colors[est], alpha=0.10)

        ax.set_title(b_label, fontsize=11)
        ax.set_xlabel("Step $t$")
        ax.grid(True, ls="--", alpha=0.5)

    axes[0].set_ylabel("Normalized Gap")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(r"E2: Causal Debiasing Across 5 Paths ($k=16, \eta\lambda=0.1, c=1.2, R=200$)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e2_paths_gap.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved E2 paths gap plot.")

    # 2. Sign flip of truncation bias effect across beta
    paired_csv = os.path.join(results_dir, "paired_summary.csv")
    if os.path.exists(paired_csv):
        p_rows = load_summary(paired_csv)
        # Filter for k=16, eta_lam=0.1
        sub_p = [
            r for r in p_rows
            if int(r["k"]) == 16 and abs(float(r["eta_lam"]) - 0.1) < 1e-4
        ]
        if sub_p:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            b_vals = []
            det_effects = []
            stoch_effects = []
            stoch_ses = []

            for r in sub_p:
                mom_str = r["momentum"]
                if "beta_" in mom_str:
                    b = float(mom_str.replace("beta_", ""))
                    b_vals.append(b)
                    det_effects.append(float(r["det_bias_effect"]))
                    stoch_effects.append(float(r["stoch_bias_effect"]))
                    stoch_ses.append(float(r["stoch_bias_se"]))

            order = np.argsort(b_vals)
            b_arr = np.array(b_vals)[order]
            det_arr = np.array(det_effects)[order]
            stoch_arr = np.array(stoch_effects)[order]
            se_arr = np.array(stoch_ses)[order]

            ax.axhline(0.0, color="black", ls="--", alpha=0.7)
            ax.plot(b_arr, det_arr, "s--", color="#ff7f0e", label="Deterministic bias: tis_mean - exact")
            ax.errorbar(b_arr, stoch_arr, yerr=se_arr, fmt="o-", color="#d62728", capsize=4,
                        label=r"Stochastic paired bias: tis - tis_debiased ($\pm$SE)")

            ax.axvspan(-0.05, 0.6, color="green", alpha=0.1, label=r"Safe region ($\rho \leq 0.25$)")
            ax.axvspan(0.6, 1.05, color="red", alpha=0.1, label="Overshoot / outside safe region")

            ax.set_xlabel(r"Momentum coefficient $\beta$")
            ax.set_ylabel(r"Bias Effect on $A_T$ (positive = slower, negative = faster)")
            ax.set_title(r"E2: Sign Flip of Truncation Bias Effect ($k=16, \eta\lambda=0.1, c=1.2$)")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=9)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "e2_bias_sign_flip.png"), bbox_inches="tight")
            plt.close(fig)
            print("Saved E2 bias sign flip plot.")

    # 3. Decomposition inner products vs t for TIS under beta=0.6 vs beta=0.99
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax_idx, (b_val, title) in enumerate([
        (0.6, r"$\beta=0.6$ (Safe region $\rho=0.25$)"),
        (0.99, r"$\beta=0.99$ (Overshoot $\rho=10.0$)"),
    ]):
        ax = axes[ax_idx]
        matching = [
            r for r in valid_rows
            if r["estimator"] == "tis"
            and int(r["k"]) == 16
            and abs(float(r["eta_lam"]) - 0.1) < 1e-4
            and r["beta_fixed"] != "" and abs(float(r["beta_fixed"]) - b_val) < 1e-4
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

        ax.plot(diag_t, inner_track, "o-", label=r"Tracking: $\langle \nabla J, \hat{E}^h - h \rangle$")
        ax.plot(diag_t, inner_bias, "s-", label=r"History bias: $\langle \nabla J, \hat{E}^b \rangle$")
        ax.plot(diag_t, inner_noise, "^-", label=r"Noise: $\langle \nabla J, \hat{E}^\xi \rangle$")
        ax.plot(diag_t, inner_update, "k--", alpha=0.7, label=r"Total: $\langle \nabla J, \hat{m} \rangle$")

        ax.axhline(0.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Diagnostic step $t$")
        ax.set_ylabel(r"Inner Product with $\nabla J(\theta_t)$")
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(r"E2: Momentum Error Decomposition Inner Products for TIS ($k=16, \eta\lambda=0.1$)", fontsize=14, y=1.02)
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
