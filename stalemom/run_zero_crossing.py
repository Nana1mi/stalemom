"""Zero-crossing experiment:
Find the exact zero-crossing point rho* where truncation bias flips from costly (positive)
to damping/beneficial (negative).
Grid:
  rho_lam in {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0}
  k in {4, 8, 12, 16}
  eta_lam in {0.03, 0.1, 0.3}
"""
import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    return {
        "k": k,
        "eta_lam": eta_lam,
        "rho_lam": rho_lam,
        "beta": 1.0 - omb,
        "det_effect": det_effect,
        "stoch_effect": stoch_effect,
        "stoch_se": stoch_se,
        "a_t_exact": a_t_exact,
        "a_t_mean": a_t_mean,
        "a_t_tis": float(np.mean(a_t_tis_runs)),
        "a_t_deb": float(np.mean(a_t_deb_runs)),
    }


def find_zero_crossing(xs: list[float], ys: list[float]) -> float:
    """Find zero crossing using linear interpolation between sign change."""
    for i in range(len(ys) - 1):
        y1, y2 = ys[i], ys[i + 1]
        x1, x2 = xs[i], xs[i + 1]
        if (y1 >= 0 and y2 <= 0) or (y1 <= 0 and y2 >= 0):
            if abs(y2 - y1) < 1e-12:
                return x1
            # Linear interpolation: x = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            return float(x1 - y1 * (x2 - x1) / (y2 - y1))
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/zero_crossing/")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rho_lams = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    ks = [4, 8, 12, 16]
    eta_lams = [0.03, 0.1, 0.3]

    tasks = []
    for k, eta_lam, rho_lam in itertools.product(ks, eta_lams, rho_lams):
        if rho_lam < eta_lam:
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
    print("\n=== ZERO-CROSSING ESTIMATES ===")
    for (k, eta_lam), group in itertools.groupby(results, key=lambda r: (r["k"], r["eta_lam"])):
        items = list(group)
        rhos = [r["rho_lam"] for r in items]
        det_ys = [r["det_effect"] for r in items]
        stoch_ys = [r["stoch_effect"] for r in items]

        rho_det_zero = find_zero_crossing(rhos, det_ys)
        rho_stoch_zero = find_zero_crossing(rhos, stoch_ys)

        cross_info = {
            "k": k,
            "eta_lam": eta_lam,
            "rho_star_det": rho_det_zero,
            "rho_star_stoch": rho_stoch_zero,
        }
        crossings.append(cross_info)
        print(f"k={k:2d}, eta_lam={eta_lam:.2f} | rho* (det) = {rho_det_zero:.3f}, rho* (stoch) = {rho_stoch_zero:.3f}")

    cross_csv = os.path.join(args.out, "zero_crossing_points.csv")
    with open(cross_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(crossings[0].keys()))
        writer.writeheader()
        writer.writerows(crossings)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f} s. Saved to {csv_path} and {cross_csv}")


if __name__ == "__main__":
    main()
