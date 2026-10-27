"""Zero-crossing experiment:
Find the exact zero-crossing point rho* where truncation bias flips from costly (positive)
to damping/beneficial (negative).
Grid:
  rho_lam in {0.1, 0.25, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0}
  k in {4, 8, 12, 16}
  eta_lam in {0.03, 0.1, 0.3}
"""
import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate


def run_condition(task: dict) -> dict:
    k = task["k"]
    eta_lam = task["eta_lam"]
    rho_lam = task["rho_lam"]
    R = task["R"]
    seed = task["seed"]
    lam = 1.0
    eta = eta_lam / lam
    rho = rho_lam / lam
    c = 1.2
    T = 512
    B = 32

    omb = eta / rho

    common_prob = {
        "K": 10,
        "n_good": 2,
        "lam": lam,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }

    # 1. Exact
    cfg_exact = {
        "problem": dict(common_prob),
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "exact",
        "k": 0,
        "eta": eta,
        "omb": omb,
        "c": None,
        "mismatch": "lag",
    }
    res_exact = simulate(cfg_exact)
    a_t_exact = float(np.mean(res_exact["gap"][0, 1:]))

    # 2. TIS Mean
    cfg_mean = {
        "problem": dict(common_prob),
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis_mean",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
        "mismatch": "lag",
    }
    res_mean = simulate(cfg_mean)
    a_t_mean = float(np.mean(res_mean["gap"][0, 1:]))
    det_effect = a_t_mean - a_t_exact

    # 3. TIS
    cfg_tis = {
        "problem": dict(common_prob),
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
        "mismatch": "lag",
    }
    res_tis = simulate(cfg_tis)
    a_t_tis_runs = np.mean(res_tis["gap"][:, 1:], axis=1)
    mean_kl_pq = float(np.mean(res_tis["kl_pq"]))
    mean_trunc_rate = float(np.mean(res_tis["trunc_rate"]))

    # 4. TIS Debiased
    cfg_deb = {
        "problem": dict(common_prob),
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis_debiased",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
        "mismatch": "lag",
    }
    res_deb = simulate(cfg_deb)
    a_t_deb_runs = np.mean(res_deb["gap"][:, 1:], axis=1)

    # Paired differences
    diff_runs = a_t_tis_runs - a_t_deb_runs
    stoch_effect = float(np.mean(diff_runs))
    stoch_se = float(np.std(diff_runs, ddof=1) / np.sqrt(R))
    t_stat = float(stoch_effect / stoch_se) if stoch_se > 1e-12 else 0.0

    return {
        "k": k,
        "eta_lam": eta_lam,
        "rho_lam": rho_lam,
        "beta": 1.0 - omb,
        "det_effect": det_effect,
        "stoch_effect": stoch_effect,
        "stoch_se": stoch_se,
        "t_stat": t_stat,
        "mean_kl_pq": mean_kl_pq,
        "mean_trunc_rate": mean_trunc_rate,
        "a_t_exact": a_t_exact,
        "a_t_mean": a_t_mean,
        "a_t_tis": float(np.mean(a_t_tis_runs)),
        "a_t_deb": float(np.mean(a_t_deb_runs)),
    }


def find_significant_zero_crossing(items: list[dict]) -> tuple[float, str]:
    """Find zero crossing using linear interpolation, but only if both sides have |t| > 3.
    Returns (rho_star, note).
    """
    items_sorted = sorted(items, key=lambda r: r["rho_lam"])
    t_stats = [r["t_stat"] for r in items_sorted]
    rhos = [r["rho_lam"] for r in items_sorted]
    effects = [r["stoch_effect"] for r in items_sorted]

    has_sig_pos = any(t > 3.0 for t in t_stats)
    has_sig_neg = any(t < -3.0 for t in t_stats)

    if not has_sig_pos and not has_sig_neg:
        return float("nan"), "No significant effect (|t| <= 3 throughout)"
    if has_sig_pos and not has_sig_neg:
        return float("nan"), "Entirely positive (costly) across all tested rho"
    if has_sig_neg and not has_sig_pos:
        return float("nan"), "Entirely negative (beneficial/damping) across tested rho"

    # Both positive and negative sides are significant (|t| > 3)
    for i in range(len(effects) - 1):
        y1, y2 = effects[i], effects[i + 1]
        x1, x2 = rhos[i], rhos[i + 1]
        if (y1 >= 0 and y2 <= 0) or (y1 <= 0 and y2 >= 0):
            if abs(y2 - y1) < 1e-12:
                return x1, "Significant sign flip"
            # Linear interpolation on rho
            x_cross = float(x1 - y1 * (x2 - x1) / (y2 - y1))
            return x_cross, "Significant sign flip"

    return float("nan"), "No sign crossing observed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/zero_crossing/")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rho_lams = [0.1, 0.25, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    ks = [4, 8, 12, 16]
    eta_lams = [0.03, 0.1, 0.3]

    tasks = []
    for k, eta_lam, rho_lam in itertools.product(ks, eta_lams, rho_lams):
        if rho_lam < eta_lam - 1e-6:
            continue
        tasks.append({
            "k": k,
            "eta_lam": eta_lam,
            "rho_lam": rho_lam,
            "R": args.R,
            "seed": 12345,
        })

    print(f"Running {len(tasks)} tasks for zero-crossing analysis...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_condition, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["k"], r["eta_lam"], r["rho_lam"]))

    # Save CSV
    csv_path = os.path.join(args.out, "zero_crossing_summary.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute zero-crossings per (k, eta_lam)
    crossings = []
    print("\n=== ROBUST ZERO-CROSSING ESTIMATES (|t| > 3 on both sides) ===")
    for (k, eta_lam), group in itertools.groupby(results, key=lambda r: (r["k"], r["eta_lam"])):
        items = list(group)
        rho_zero, note = find_significant_zero_crossing(items)
        cross_info = {
            "k": k,
            "eta_lam": eta_lam,
            "rho_star_stoch": rho_zero,
            "status": note,
        }
        crossings.append(cross_info)
        rho_str = f"{rho_zero:.3f}" if not np.isnan(rho_zero) else "N/A"
        print(f"k={k:2d}, eta_lam={eta_lam:.2f} | rho* = {rho_str:>6} ({note})")

    cross_csv = os.path.join(args.out, "zero_crossing_points.csv")
    with open(cross_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(crossings[0].keys()))
        writer.writeheader()
        writer.writerows(crossings)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f} s. Saved to {csv_path} and {cross_csv}")


if __name__ == "__main__":
    main()
