"""Stepsize control experiment:
Test whether truncation bias simply acts as stepsize shrinkage.
Compares exact and is over eta in [0.01, 0.3] (10 log-spaced points) against
tis_mean and tis at eta=0.1 under beta in {0.9, 0.99}.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from stalemom.sim import simulate
from stalemom.metrics import compute_run_metrics


def run_single(cfg: dict, out_dir: str, run_idx: int) -> dict:
    t0 = time.time()
    result = simulate(cfg)
    metrics = compute_run_metrics(result, cfg)
    sim_time = time.time() - t0

    npz_filename = f"stepsize_run_{run_idx:04d}.npz"
    npz_path = os.path.join(out_dir, npz_filename)
    np.savez_compressed(
        npz_path,
        gap=result["gap"],
        J=result["J"],
        kl_pq=result["kl_pq"],
        trunc_rate=result["trunc_rate"],
        lag=result["lag"],
        diag_t=result["diag_t"],
    )

    return {
        "run_id": run_idx,
        "npz_file": npz_filename,
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
    etas = np.geomspace(0.01, 0.3, 10)
    estimators = ["exact", "is", "tis_mean", "tis"]
    k = 16
    c = 1.2
    T = 512
    B = 32

    tasks = []
    run_idx = 0

    for b in betas:
        omb = 1.0 - b
        for eta in etas:
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
                tasks.append((cfg, args.out, run_idx))
                run_idx += 1

    print(f"Running {len(tasks)} tasks for stepsize control...")
    t0 = time.time()
    summary_rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, cfg, out, idx): idx for (cfg, out, idx) in tasks}
        for fut in as_completed(futures):
            summary_rows.append(fut.result())

    summary_rows.sort(key=lambda r: r["run_id"])

    csv_path = os.path.join(args.out, "stepsize_summary.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f} s. Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
