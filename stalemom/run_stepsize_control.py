"""Stepsize control experiment:
Test whether truncation bias simply acts as stepsize shrinkage.
Sweeps exact, is, tis_mean, and tis over eta in [0.01, 2.0] under beta in {0.9, 0.99}.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate
from stalemom.metrics import compute_run_metrics


def run_single(task: dict) -> dict:
    cfg = task["cfg"]
    run_idx = task["run_idx"]
    t0 = time.time()
    result = simulate(cfg)
    metrics = compute_run_metrics(result, cfg)
    sim_time = time.time() - t0

    return {
        "run_id": run_idx,
        "estimator": cfg["estimator"],
        "beta": cfg["beta_val"],
        "eta": cfg["eta"],
        "k": cfg["k"],
        "c": cfg["c"] if cfg["c"] is not None else "",
        "sim_time": sim_time,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/stepsize_control/")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    betas = [0.9, 0.99]
    # 10 log-spaced points from 0.01 to 0.3, plus 0.1 explicitly, plus 0.5, 1.0, 2.0
    base_etas = list(np.geomspace(0.01, 0.3, 10))
    extended_etas = [0.1, 0.5, 1.0, 2.0]
    all_etas = sorted(list(set([round(e, 5) for e in base_etas + extended_etas])))

    estimators = ["exact", "is", "tis_mean", "tis"]
    k = 16
    c = 1.2
    T = 512
    B = 32

    tasks = []
    run_idx = 0

    for b in betas:
        omb = 1.0 - b
        for eta in all_etas:
            for est in estimators:
                eff_k = 0 if est == "exact" else k
                eff_c = None if est in ("exact", "is") else c
                eff_R = 1 if est in ("exact", "tis_mean") else args.R

                cfg = {
                    "problem": {
                        "K": 10,
                        "n_good": 2,
                        "lam": 1.0,
                        "init": "random",
                        "init_seed": 0,
                        "init_scale": 1.0,
                        "ref": "init",
                    },
                    "run": {
                        "R": eff_R,
                        "seed": 12345,
                        "T": T,
                        "B": B,
                        "diag_every": 8,
                        "return_traj": False,
                    },
                    "estimator": est,
                    "k": eff_k,
                    "eta": float(eta),
                    "omb": float(omb),
                    "c": eff_c,
                    "mismatch": "lag",
                    "beta_val": b,
                }
                tasks.append({"cfg": cfg, "run_idx": run_idx})
                run_idx += 1

    print(f"Running {len(tasks)} stepsize control conditions...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_single, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["beta"], r["estimator"], r["eta"]))

    csv_path = os.path.join(args.out, "stepsize_summary.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f} s. Saved to {csv_path}")


if __name__ == "__main__":
    main()
