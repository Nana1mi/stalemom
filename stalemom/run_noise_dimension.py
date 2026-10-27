"""Noise and Reward Dimension Experiment:
In eta*lam = 0.1, k = 16:
Sweep:
  1. B in {4, 8, 16, 32, 64, 128} x rho*lam in {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0} (with n_good=2)
  2. n_good in {2, 3, 5} x rho*lam in {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0} (with B=32)
Output:
  - Paired effect (tis - tis_debiased) +/- SE
  - mean_kl_pq and mean_trunc_rate
  - Zero-crossing point rho* tracking how noise (B) and landscape difficulty (n_good) shift the critical boundary.
"""
import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate


def run_noise_condition(task: dict) -> dict:
    dim = task["dimension"]
    B = task["B"]
    n_good = task["n_good"]
    rho_lam = task["rho_lam"]
    R = task["R"]
    seed = task["seed"]

    lam = 1.0
    eta_lam = 0.1
    eta = eta_lam / lam
    rho = rho_lam / lam
    k = 16
    c = 1.2
    T = 512

    omb = eta / rho
    beta = 1.0 - omb

    prob = {
        "K": 10,
        "n_good": n_good,
        "lam": lam,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }

    # 1. Exact
    cfg_exact = {
        "problem": prob,
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "exact",
        "k": 0,
        "eta": eta,
        "omb": omb,
        "c": None,
    }
    res_exact = simulate(cfg_exact)
    a_t_exact = float(np.mean(res_exact["gap"][0, 1:]))

    # 2. TIS Mean
    cfg_mean = {
        "problem": prob,
        "run": {"R": 1, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis_mean",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_mean = simulate(cfg_mean)
    a_t_mean = float(np.mean(res_mean["gap"][0, 1:]))
    det_effect = a_t_mean - a_t_exact

    # 3. TIS
    cfg_tis = {
        "problem": prob,
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_tis = simulate(cfg_tis)
    a_t_tis_runs = np.mean(res_tis["gap"][:, 1:], axis=1)
    mean_kl_pq = float(np.mean(res_tis["kl_pq"]))
    mean_trunc_rate = float(np.mean(res_tis["trunc_rate"]))

    # 4. TIS Debiased
    cfg_deb = {
        "problem": prob,
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis_debiased",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
    }
    res_deb = simulate(cfg_deb)
    a_t_deb_runs = np.mean(res_deb["gap"][:, 1:], axis=1)

    diff_runs = a_t_tis_runs - a_t_deb_runs
    stoch_effect = float(np.mean(diff_runs))
    stoch_se = float(np.std(diff_runs, ddof=1) / np.sqrt(R))
    t_stat = float(stoch_effect / stoch_se) if stoch_se > 1e-12 else 0.0

    return {
        "dimension": dim,
        "B": B,
        "n_good": n_good,
        "eta_lam": eta_lam,
        "rho_lam": rho_lam,
        "beta": beta,
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


def find_zero_crossing(rhos: list[float], effects: list[float], t_stats: list[float]) -> tuple[float, str]:
    has_sig_pos = any(t > 3.0 for t in t_stats)
    has_sig_neg = any(t < -3.0 for t in t_stats)

    if not has_sig_pos and not has_sig_neg:
        return float("nan"), "No significant effect (|t| <= 3 throughout)"
    if has_sig_pos and not has_sig_neg:
        return float("nan"), "Entirely positive (costly) across all tested rho"
    if has_sig_neg and not has_sig_pos:
        return float("nan"), "Entirely negative (beneficial/damping) across tested rho"

    for i in range(len(effects) - 1):
        y1, y2 = effects[i], effects[i + 1]
        x1, x2 = rhos[i], rhos[i + 1]
        if (y1 >= 0 and y2 <= 0) or (y1 <= 0 and y2 >= 0):
            if abs(y2 - y1) < 1e-12:
                return x1, "Significant sign flip"
            return float(x1 - y1 * (x2 - x1) / (y2 - y1)), "Significant sign flip"

    return float("nan"), "No sign crossing observed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/noise_dimension/")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rho_lams = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    batch_sizes = [4, 8, 16, 32, 64, 128]
    n_goods = [2, 3, 5]

    tasks = []

    # 1. Sweep B (with n_good=2)
    for B in batch_sizes:
        for rho in rho_lams:
            tasks.append({
                "dimension": "batch_size",
                "B": B,
                "n_good": 2,
                "rho_lam": rho,
                "R": args.R,
                "seed": 12345,
            })

    # 2. Sweep n_good (with B=32)
    for ng in [3, 5]:  # n_good=2 is already run in batch_size at B=32
        for rho in rho_lams:
            tasks.append({
                "dimension": "n_good",
                "B": 32,
                "n_good": ng,
                "rho_lam": rho,
                "R": args.R,
                "seed": 12345,
            })

    print(f"Running {len(tasks)} noise dimension conditions...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_noise_condition, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["dimension"], r["B"], r["n_good"], r["rho_lam"]))

    # Save summary CSV
    csv_path = os.path.join(args.out, "noise_dimension_summary.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute zero crossings
    crossings = []

    # Group by B
    print("\n=== ZERO-CROSSING vs BATCH SIZE B (n_good=2) ===")
    b_results = [r for r in results if r["dimension"] == "batch_size"]
    for B, group in itertools.groupby(b_results, key=lambda r: r["B"]):
        items = sorted(list(group), key=lambda r: r["rho_lam"])
        rhos = [r["rho_lam"] for r in items]
        effects = [r["stoch_effect"] for r in items]
        t_stats = [r["t_stat"] for r in items]

        rho_star, status = find_zero_crossing(rhos, effects, t_stats)
        crossings.append({
            "dimension": "batch_size",
            "param_name": "B",
            "param_val": B,
            "rho_star": rho_star,
            "status": status,
        })
        val_str = f"{rho_star:.3f}" if not np.isnan(rho_star) else "N/A"
        print(f"B={B:3d} | rho* = {val_str:>6} ({status})")

    # Group by n_good (include n_good=2 from B=32)
    print("\n=== ZERO-CROSSING vs n_good (B=32) ===")
    ng_items_all = [r for r in results if (r["dimension"] == "n_good" or (r["dimension"] == "batch_size" and r["B"] == 32))]
    ng_items_all.sort(key=lambda r: (r["n_good"], r["rho_lam"]))

    for ng, group in itertools.groupby(ng_items_all, key=lambda r: r["n_good"]):
        items = sorted(list(group), key=lambda r: r["rho_lam"])
        rhos = [r["rho_lam"] for r in items]
        effects = [r["stoch_effect"] for r in items]
        t_stats = [r["t_stat"] for r in items]

        rho_star, status = find_zero_crossing(rhos, effects, t_stats)
        crossings.append({
            "dimension": "n_good",
            "param_name": "n_good",
            "param_val": ng,
            "rho_star": rho_star,
            "status": status,
        })
        val_str = f"{rho_star:.3f}" if not np.isnan(rho_star) else "N/A"
        print(f"n_good={ng:2d} | rho* = {val_str:>6} ({status})")

    cross_csv = os.path.join(args.out, "noise_dimension_crossings.csv")
    with open(cross_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(crossings[0].keys()))
        writer.writeheader()
        writer.writerows(crossings)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f} s. Saved to {csv_path} and {cross_csv}")


if __name__ == "__main__":
    main()
