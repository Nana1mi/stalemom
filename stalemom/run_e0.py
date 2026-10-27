"""Run E0-R (reproduction of Wang et al. 2026 §6.1)."""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from stalemom.sim import simulate


def run_e0_task(args: dict) -> dict:
    """Run 3 groups (no_mom, fixed_beta, scaled_beta) for a given T and init."""
    init_type = args["init"]
    T = args["T"]
    R = args["R"]
    base_seed = args["base_seed"]
    lam = 1.0
    eta = 1.0 / T

    common_prob = {
        "K": 10,
        "n_good": 3,
        "lam": lam,
        "init": init_type,
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }

    # Group 1: No momentum (omb = 1.0)
    cfg_0 = {
        "problem": dict(common_prob),
        "run": {"R": R, "seed": base_seed + 1, "T": T, "B": 1, "diag_every": T + 1, "return_traj": False},
        "estimator": "onpolicy",
        "k": 0,
        "eta": eta,
        "omb": 1.0,
        "c": None,
        "mismatch": "lag",
    }
    res_0 = simulate(cfg_0)
    J_0 = res_0["J"][:, -1]

    # Group 2: Fixed beta = 0.9 (omb = 0.1)
    cfg_fixed = {
        "problem": dict(common_prob),
        "run": {"R": R, "seed": base_seed + 2, "T": T, "B": 1, "diag_every": T + 1, "return_traj": False},
        "estimator": "onpolicy",
        "k": 0,
        "eta": eta,
        "omb": 0.1,
        "c": None,
        "mismatch": "lag",
    }
    res_fixed = simulate(cfg_fixed)
    J_fixed = res_fixed["J"][:, -1]

    # Group 3: Scaled beta = 1 - 4*eta (rho = 0.25 -> omb = 4*eta)
    cfg_scaled = {
        "problem": dict(common_prob),
        "run": {"R": R, "seed": base_seed + 3, "T": T, "B": 1, "diag_every": T + 1, "return_traj": False},
        "estimator": "onpolicy",
        "k": 0,
        "eta": eta,
        "omb": 4.0 * eta,
        "c": None,
        "mismatch": "lag",
    }
    res_scaled = simulate(cfg_scaled)
    J_scaled = res_scaled["J"][:, -1]

    # Statistics: diff = J_beta - J_0
    diff_fixed = J_fixed - J_0
    mean_fixed = float(np.mean(diff_fixed))
    std_fixed = float(np.std(diff_fixed, ddof=1))
    snr_fixed = float(abs(mean_fixed) / std_fixed) if std_fixed > 0 else 0.0
    win_fixed = float(np.mean(diff_fixed > 0.0))

    diff_scaled = J_scaled - J_0
    mean_scaled = float(np.mean(diff_scaled))
    std_scaled = float(np.std(diff_scaled, ddof=1))
    snr_scaled = float(abs(mean_scaled) / std_scaled) if std_scaled > 0 else 0.0
    win_scaled = float(np.mean(diff_scaled > 0.0))

    return {
        "init": init_type,
        "T": T,
        "eta": eta,
        "mean_fixed": mean_fixed,
        "std_fixed": std_fixed,
        "snr_fixed": snr_fixed,
        "win_fixed": win_fixed,
        "mean_scaled": mean_scaled,
        "std_scaled": std_scaled,
        "snr_scaled": snr_scaled,
        "win_scaled": win_scaled,
    }


def main():
    parser = argparse.ArgumentParser(description="Run E0-R replication")
    parser.add_argument("--out", type=str, default="results/e0_repro/", help="Output dir")
    parser.add_argument("--R", type=int, default=20000, help="Runs per group")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Workers")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    Ts = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    inits = ["uniform", "random"]

    tasks = []
    task_idx = 0
    for init_type in inits:
        for T in Ts:
            tasks.append({
                "init": init_type,
                "T": T,
                "R": args.R,
                "base_seed": 100000 + task_idx * 10,
            })
            task_idx += 1

    print(f"Launching {len(tasks)} tasks for E0-R (R={args.R})...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_e0_task, tasks):
            results.append(res)
            print(
                f"[{res['init']}] T={res['T']:4d}, eta={res['eta']:.5f} | "
                f"Fixed: mean={res['mean_fixed']:+.4e}, std={res['std_fixed']:.4e}, SNR={res['snr_fixed']:.3f}, win={res['win_fixed']:.3f} | "
                f"Scaled: mean={res['mean_scaled']:+.4e}, std={res['std_scaled']:.4e}, SNR={res['snr_scaled']:.3f}, win={res['win_scaled']:.3f}"
            )

    elapsed = time.time() - t0
    print(f"E0-R completed in {elapsed:.2f} s.")

    # Save summary CSV
    csv_path = os.path.join(args.out, "e0_summary.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save NPZ
    npz_path = os.path.join(args.out, "e0_results.npz")
    save_arrays = {k: np.array([r[k] for r in results]) for k in fieldnames if k != "init"}
    save_arrays["init"] = np.array([r["init"] for r in results])
    np.savez_compressed(npz_path, **save_arrays)
    print(f"Saved results to {csv_path} and {npz_path}")


if __name__ == "__main__":
    main()
