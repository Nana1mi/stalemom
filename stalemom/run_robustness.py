"""Robustness sweep across:
1. init_seed in {0, 1, 2, 3, 4}
2. n_good in {2, 5}
3. B in {1, 32, 128}
4. c in {1.2, 1.5, 2.0}
5. gradient_type: NPG (eta=0.1) vs Vanilla Policy Gradient (eta=0.1, eta=1.0)
Testing both safe regime (beta=0.6, rho=0.25) and overshoot regime (beta=0.99, rho=10.0).
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate


def run_robust_condition(task: dict) -> dict:
    dim = task["dimension"]
    var_label = task["variation"]
    beta = task["beta"]
    R = task["R"]
    seed = task["seed"]
    T = task.get("T", 512)
    B = task.get("B", 32)
    k = task.get("k", 16)
    c = task.get("c", 1.2)
    lam = task.get("lam", 1.0)
    eta = task.get("eta", 0.1)
    init_seed = task.get("init_seed", 0)
    n_good = task.get("n_good", 2)
    gradient_type = task.get("gradient_type", "npg")

    omb = 1.0 - beta

    prob = {
        "K": 10,
        "n_good": n_good,
        "lam": lam,
        "init": "random",
        "init_seed": init_seed,
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
        "gradient_type": gradient_type,
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
        "gradient_type": gradient_type,
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
        "gradient_type": gradient_type,
    }
    res_tis = simulate(cfg_tis)
    a_t_tis_runs = np.mean(res_tis["gap"][:, 1:], axis=1)
    trunc_rate = float(np.mean(res_tis["trunc_rate"]))

    # 4. TIS Debiased
    cfg_deb = {
        "problem": prob,
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": T + 1, "return_traj": False},
        "estimator": "tis_debiased",
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": c,
        "gradient_type": gradient_type,
    }
    res_deb = simulate(cfg_deb)
    a_t_deb_runs = np.mean(res_deb["gap"][:, 1:], axis=1)

    diff_runs = a_t_tis_runs - a_t_deb_runs
    stoch_effect = float(np.mean(diff_runs))
    stoch_se = float(np.std(diff_runs, ddof=1) / np.sqrt(R))

    return {
        "dimension": dim,
        "variation": var_label,
        "beta": beta,
        "gradient_type": gradient_type,
        "eta": eta,
        "B": B,
        "c": c,
        "init_seed": init_seed,
        "n_good": n_good,
        "trunc_rate": trunc_rate,
        "det_effect": det_effect,
        "stoch_effect": stoch_effect,
        "stoch_se": stoch_se,
        "a_t_exact": a_t_exact,
        "a_t_mean": a_t_mean,
        "a_t_tis": float(np.mean(a_t_tis_runs)),
        "a_t_deb": float(np.mean(a_t_deb_runs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/robustness/")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tasks = []
    betas = [0.6, 0.99]

    # Baseline defaults:
    base = {
        "R": args.R,
        "seed": 12345,
        "T": 512,
        "B": 32,
        "k": 16,
        "c": 1.2,
        "lam": 1.0,
        "eta": 0.1,
        "init_seed": 0,
        "n_good": 2,
        "gradient_type": "npg",
    }

    # 1. Init seeds in {0, 1, 2, 3, 4}
    for s in [0, 1, 2, 3, 4]:
        for beta in betas:
            t = dict(base, dimension="init_seed", variation=f"init_seed={s}", init_seed=s, beta=beta)
            tasks.append(t)

    # 2. n_good in {2, 5}
    for ng in [2, 5]:
        for beta in betas:
            t = dict(base, dimension="n_good", variation=f"n_good={ng}", n_good=ng, beta=beta)
            tasks.append(t)

    # 3. Batch size B in {1, 32, 128}
    for b in [1, 32, 128]:
        for beta in betas:
            t = dict(base, dimension="batch_size", variation=f"B={b}", B=b, beta=beta)
            tasks.append(t)

    # 4. Truncation c in {1.2, 1.5, 2.0}
    for c_val in [1.2, 1.5, 2.0]:
        for beta in betas:
            t = dict(base, dimension="clipping_c", variation=f"c={c_val}", c=c_val, beta=beta)
            tasks.append(t)

    # 5. Gradient type: NPG (eta=0.1) vs Vanilla (eta=0.1, eta=1.0)
    for gtype, eta_val in [("npg", 0.1), ("vanilla", 0.1), ("vanilla", 1.0)]:
        for beta in betas:
            t = dict(
                base,
                dimension="gradient_type",
                variation=f"{gtype} (eta={eta_val})",
                gradient_type=gtype,
                eta=eta_val,
                beta=beta,
            )
            tasks.append(t)

    print(f"Running {len(tasks)} robustness conditions with R={args.R}...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_robust_condition, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["dimension"], r["variation"], r["beta"]))

    csv_path = os.path.join(args.out, "robustness_summary.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f} s. Saved to {csv_path}")

    # Print summary table
    print("\n=== ROBUSTNESS SUMMARY ===")
    print(f"{'Dimension':<15} {'Variation':<22} {'beta':<6} {'Effect (TIS-deb)':<18} {'SE':<10} {'Sign Flip?'}")
    print("-" * 80)
    for res in results:
        sig = "Positive (+)" if res["stoch_effect"] > 0 else "Negative (-)"
        print(
            f"{res['dimension']:<15} {res['variation']:<22} {res['beta']:<6.2f} "
            f"{res['stoch_effect']:+10.5f}        {res['stoch_se']:8.5f}   {sig}"
        )


if __name__ == "__main__":
    main()
