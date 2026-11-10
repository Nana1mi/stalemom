"""SPEC E5 - Phase D2: Tuned momentum comparison on bandit (local CPU).

Grid:
  estimator in {onpolicy, is, tis, tis_debiased}
  k in {0, 16}
  B in {4, 32}
  beta in {0, 0.6, 0.9, 0.99}
  eta in np.geomspace(0.01, 1.0, 11)
  R = 200, T = 512, c = 1.2
"""
import argparse
import csv
import datetime
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate


def write_env_file(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env_path = os.path.join(out_dir, "env.txt")
    lines = []
    lines.append(f"timestamp: {datetime.datetime.now().isoformat()}")
    lines.append(f"python: {sys.version}")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True
        ).strip()
    except Exception as e:
        git_hash = f"unavailable ({e})"
    lines.append(f"git_commit: {git_hash}")

    lines.append("\n=== pip freeze ===")
    try:
        pip_out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], cwd=os.getcwd(), text=True
        ).strip()
    except Exception as e:
        pip_out = f"unavailable ({e})"
    lines.append(pip_out)

    lines.append("\n=== nvidia-smi ===")
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi"], cwd=os.getcwd(), text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as e:
        smi_out = f"unavailable ({e})"
    lines.append(smi_out)

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_single_d2(task: dict) -> dict:
    estimator = task["estimator"]
    k = int(task["k"])
    B = int(task["B"])
    beta = float(task["beta"])
    eta = float(task["eta"])
    R = int(task["R"])
    T = int(task["T"])
    c = float(task["c"])
    seed = int(task["seed"])

    omb = 1.0 - beta
    eff_c = c if estimator in ("tis", "tis_debiased", "tis_mean") else None

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
            "R": R,
            "seed": seed,
            "T": T,
            "B": B,
            "diag_every": T + 1,
            "return_traj": False,
        },
        "estimator": estimator,
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": eff_c,
        "mismatch": "lag",
    }

    t0 = time.time()
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            res = simulate(cfg)
        elapsed = time.time() - t0
        gap = res["gap"]  # (R, T+1)
        diverged = res["diverged"]  # (R,)
        diverge_rate = float(np.mean(diverged))

        # Per SPEC §2.4 & §4: Diverged runs are recorded as inf, not skipped.
        if np.any(diverged) or not np.all(np.isfinite(gap[:, 1:])):
            A_T = float("inf")
            A_T_se = float("nan")
            final_gap = float("inf")
            final_gap_se = float("nan")
        else:
            run_a_t = np.mean(gap[:, 1:], axis=1)
            A_T = float(np.mean(run_a_t))
            A_T_se = float(np.std(run_a_t, ddof=1) / np.sqrt(R)) if R > 1 else 0.0
            run_final = gap[:, -1]
            final_gap = float(np.mean(run_final))
            final_gap_se = float(np.std(run_final, ddof=1) / np.sqrt(R)) if R > 1 else 0.0

        mean_trunc_rate = float(np.mean(res["trunc_rate"]))
        mean_kl_pq = float(np.mean(res["kl_pq"]))
    except ValueError:
        elapsed = time.time() - t0
        A_T = float("inf")
        A_T_se = float("nan")
        final_gap = float("inf")
        final_gap_se = float("nan")
        diverge_rate = float("nan")
        mean_trunc_rate = float("nan")
        mean_kl_pq = float("nan")

    return {
        "estimator": estimator,
        "k": k,
        "B": B,
        "beta": beta,
        "eta": eta,
        "A_T": A_T,
        "A_T_se": A_T_se,
        "final_gap": final_gap,
        "final_gap_se": final_gap_se,
        "diverge_rate": diverge_rate,
        "mean_trunc_rate": mean_trunc_rate,
        "mean_kl_pq": mean_kl_pq,
        "sim_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SPEC E5 Phase D2: Tuned momentum on bandit.")
    parser.add_argument("--out", type=str, default="results/e5/d2")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--c", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_env_file(args.out)
    log_path = os.path.join(args.out, "run_log.txt")

    estimators = ["onpolicy", "is", "tis", "tis_debiased"]
    ks = [0, 16]
    Bs = [4, 32]
    betas = [0.0, 0.6, 0.9, 0.99]
    etas = [float(x) for x in np.geomspace(0.01, 1.0, 11)]

    tasks = []
    for est, k, B, beta, eta in itertools.product(estimators, ks, Bs, betas, etas):
        tasks.append({
            "estimator": est,
            "k": k,
            "B": B,
            "beta": beta,
            "eta": eta,
            "R": args.R,
            "T": args.T,
            "c": args.c,
            "seed": args.seed,
        })

    t0 = time.time()
    msg = f"[D2] Starting {len(tasks)} configurations with R={args.R}, T={args.T}..."
    print(msg)
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(msg + "\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_single_d2, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["estimator"], r["k"], r["B"], r["beta"], r["eta"]))

    # Annotate best eta per (estimator, k, B, beta)
    best_rows = []
    for (est, k, B, beta), group in itertools.groupby(
        results, key=lambda r: (r["estimator"], r["k"], r["B"], r["beta"])
    ):
        grp_list = list(group)
        best_item = min(grp_list, key=lambda r: r["A_T"] if not np.isnan(r["A_T"]) else float("inf"))
        best_eta = best_item["eta"]
        best_at = best_item["A_T"]
        best_at_se = best_item["A_T_se"]
        best_fg = best_item["final_gap"]
        best_fg_se = best_item["final_gap_se"]

        for r in grp_list:
            r["is_best_eta"] = (r["eta"] == best_eta)
            r["best_eta"] = best_eta
            r["best_A_T"] = best_at
            r["best_A_T_se"] = best_at_se
            r["best_final_gap"] = best_fg
            r["best_final_gap_se"] = best_fg_se

        best_rows.append(dict(best_item))

    csv_path = os.path.join(args.out, "tuned_momentum.csv")
    fieldnames = [
        "estimator",
        "k",
        "B",
        "beta",
        "eta",
        "A_T",
        "A_T_se",
        "final_gap",
        "final_gap_se",
        "diverge_rate",
        "mean_trunc_rate",
        "mean_kl_pq",
        "is_best_eta",
        "best_eta",
        "best_A_T",
        "best_A_T_se",
        "best_final_gap",
        "best_final_gap_se",
        "sim_time",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_csv_path = os.path.join(args.out, "best_eta_summary.csv")
    with open(best_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    elapsed = time.time() - t0
    done_msg = f"[D2] Completed {len(results)} configs in {elapsed:.2f}s. Saved to {csv_path}"
    print(done_msg)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(done_msg + "\n")
        for b in best_rows:
            lf.write(
                f"  {b['estimator']:<13} k={b['k']:<2} B={b['B']:<2} beta={b['beta']:<4} "
                f"-> best_eta={b['eta']:.5f}, A_T={b['A_T']:.5f} +/- {b['A_T_se']:.5f}, "
                f"final_gap={b['final_gap']:.5f}\n"
            )

    with open(os.path.join(args.out, "DONE"), "w", encoding="utf-8") as df:
        df.write(f"DONE at {datetime.datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
