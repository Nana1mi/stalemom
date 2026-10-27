"""Plot gap(t) trajectories on log scale for beta=0.99, eta=0.1, k=16, c=1.2.
Compares exact, tis_mean, tis, and tis_debiased to visualize momentum overshoot
and truncation rebound suppression.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from stalemom.sim import simulate


def main():
    out_dir = "results/stepsize_control"
    os.makedirs(out_dir, exist_ok=True)

    T = 512
    k = 16
    c = 1.2
    eta = 0.1
    beta = 0.99
    omb = 1.0 - beta
    B = 32
    R = 200
    seed = 12345

    prob = {
        "K": 10,
        "n_good": 2,
        "lam": 1.0,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }

    # 1. Exact
    cfg_exact = {
        "problem": prob,
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1},
        "estimator": "exact",
        "k": 0,
        "eta": eta,
        "omb": omb,
        "c": None,
    }
    res_exact = simulate(cfg_exact)
    gap_exact = res_exact["gap"][0]

    # 2. TIS Mean (deterministic bias)
    cfg_mean = {
        "problem": prob,
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1},
        "estimator": "tis_mean",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_mean = simulate(cfg_mean)
    gap_mean = res_mean["gap"][0]

    # 3. TIS (stochastic with truncation)
    cfg_tis = {
        "problem": prob,
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1},
        "estimator": "tis",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_tis = simulate(cfg_tis)
    gap_tis_mean = np.mean(res_tis["gap"], axis=0)
    gap_tis_se = np.std(res_tis["gap"], axis=0, ddof=1) / np.sqrt(R)

    # 4. TIS Debiased (stochastic without truncation bias)
    cfg_deb = {
        "problem": prob,
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1},
        "estimator": "tis_debiased",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_deb = simulate(cfg_deb)
    gap_deb_mean = np.mean(res_deb["gap"], axis=0)
    gap_deb_se = np.std(res_deb["gap"], axis=0, ddof=1) / np.sqrt(R)

    t = np.arange(T + 1)

    # Two-panel plot: Left panel full T=512, Right panel zoom-in on early rebound t in [0, 80]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200)

    # Styling
    colors = {
        "exact": "#000000",
        "tis_mean": "#ff7f0e",
        "tis": "#2ca02c",
        "tis_deb": "#d62728",
    }

    for ax, t_max, title in [(ax1, 512, "Full Trajectory (t = 0 to 512)"), (ax2, 80, "Early Rebound Window (t = 0 to 80)")]:
        sub_t = t[t <= t_max]

        # Exact
        ax.plot(sub_t, gap_exact[:len(sub_t)], color=colors["exact"], lw=2.0, label="Exact NPG (no momentum dampening)")

        # TIS Mean
        ax.plot(sub_t, gap_mean[:len(sub_t)], color=colors["tis_mean"], lw=2.0, ls="--", label=r"TIS Mean (exact $\mathbb{E}$ + truncation bias)")

        # TIS Debiased (stochastic)
        ax.plot(sub_t, gap_deb_mean[:len(sub_t)], color=colors["tis_deb"], lw=1.6, label=r"TIS Debiased (stochastic, unbiased)")
        ax.fill_between(
            sub_t,
            np.maximum(1e-5, gap_deb_mean[:len(sub_t)] - gap_deb_se[:len(sub_t)]),
            gap_deb_mean[:len(sub_t)] + gap_deb_se[:len(sub_t)],
            color=colors["tis_deb"],
            alpha=0.15,
        )

        # TIS (stochastic with truncation)
        ax.plot(sub_t, gap_tis_mean[:len(sub_t)], color=colors["tis_tis" if "tis_tis" in colors else "tis"], lw=1.8, label=r"TIS (stochastic with truncation bias)")
        ax.fill_between(
            sub_t,
            np.maximum(1e-5, gap_tis_mean[:len(sub_t)] - gap_tis_se[:len(sub_t)]),
            gap_tis_mean[:len(sub_t)] + gap_tis_se[:len(sub_t)],
            color=colors["tis"],
            alpha=0.15,
        )

        ax.set_yscale("log")
        ax.set_xlabel("Optimization Step t", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    # Add annotations on right panel
    ax2.annotate(
        "Exact rebounds:\n0.0086 (t=16) -> 0.2099 (t=32)",
        xy=(32, gap_exact[32]),
        xytext=(36, 0.35),
        arrowprops=dict(arrowstyle="->", color=colors["exact"], lw=1.5),
        fontsize=9,
        fontweight="bold",
        color=colors["exact"],
    )
    ax2.annotate(
        "TIS Debiased rebounds:\n0.0708 (t=16) -> 0.2465 (t=32)",
        xy=(32, gap_deb_mean[32]),
        xytext=(42, 0.12),
        arrowprops=dict(arrowstyle="->", color=colors["tis_deb"], lw=1.5),
        fontsize=9,
        fontweight="bold",
        color=colors["tis_deb"],
    )
    ax2.annotate(
        "TIS & TIS Mean stay suppressed\n(no rebound)",
        xy=(32, gap_mean[32]),
        xytext=(15, 0.0003),
        arrowprops=dict(arrowstyle="->", color=colors["tis_mean"], lw=1.5),
        fontsize=9,
        fontweight="bold",
        color=colors["tis_mean"],
    )

    ax1.set_ylabel(r"Optimality Gap $1 - J(\theta_t)/J^*$ (Log Scale)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax2.legend(loc="lower left", fontsize=9.5)

    fig.suptitle(
        r"Momentum Overshoot and Rebound Suppression by Truncation Bias ($\beta=0.99, \eta=0.1, k=16, c=1.2$)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, "gap_trajectory_overshoot.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overshoot trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
