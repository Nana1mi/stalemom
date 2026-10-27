import argparse
import csv
import itertools
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import yaml

from stalemom.sim import simulate
from stalemom.metrics import compute_run_metrics


def get_git_commit() -> str:
    """Retrieve the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def expand_sweep(config_data: dict) -> list[tuple[dict, bool, str]]:
    """Expand cartesian product of sweep parameters.

    Returns a list of tuples: (single_cfg, is_skipped, skip_reason)
    """
    prob_cfg = config_data["problem"]
    run_cfg = config_data["run"]
    sweep_cfg = config_data.get("sweep", {})

    lam = float(prob_cfg["lam"])

    # Extract sweep lists with defaults if not present
    estimators = sweep_cfg.get("estimator", ["tis"])
    ks = sweep_cfg.get("k", [0])
    eta_lams = sweep_cfg.get("eta_lam", [0.01])
    momentums = sweep_cfg.get("momentum", [{"rho_lam": 0.0}])
    cs = sweep_cfg.get("c", [None])
    mismatches = sweep_cfg.get("mismatch", ["lag"])

    expanded = []

    for est in estimators:
        # If estimator is exact/onpolicy, k has no effect, so only evaluate k=0
        eff_ks = [0] if est in ("exact", "onpolicy") else ks
        # If estimator is exact/onpolicy/is, c has no effect ("is 只跑一次")
        eff_cs = [None] if est in ("exact", "onpolicy", "is") else cs

        for k, eta_lam, mom, c, mismatch in itertools.product(
            eff_ks, eta_lams, momentums, eff_cs, mismatches
        ):
            eta = float(eta_lam) / lam
            skip = False
            reason = ""
            omb = 1.0

            # Determine momentum / omb
            mom_desc = {}
            if "rho_lam" in mom:
                rho_lam = float(mom["rho_lam"])
                rho = rho_lam / lam
                mom_desc["rho_lam"] = rho_lam
                mom_desc["rho"] = rho
                if rho == 0.0:
                    omb = 1.0
                elif rho < eta:
                    skip = True
                    reason = f"rho ({rho:.5g}) < eta ({eta:.5g})"
                    omb = float("nan")
                else:
                    omb = float(eta / rho)
            elif "beta_fixed" in mom:
                beta_fixed = float(mom["beta_fixed"])
                mom_desc["beta_fixed"] = beta_fixed
                omb = float(1.0 - beta_fixed)
            else:
                raise ValueError(f"Unrecognized momentum specification: {mom}")

            single_cfg = {
                "problem": dict(prob_cfg),
                "run": dict(run_cfg),
                "estimator": est,
                "k": int(k),
                "eta_lam": float(eta_lam),
                "eta": float(eta),
                "momentum": mom_desc,
                "omb": omb,
                "c": float(c) if c is not None else None,
                "mismatch": mismatch,
            }
            expanded.append((single_cfg, skip, reason))

    return expanded


def run_single_task(args: tuple[int, dict, str]) -> dict:
    """Execute a single simulation and save its .npz."""
    run_idx, single_cfg, out_dir = args
    t0 = time.time()
    result = simulate(single_cfg)
    sim_time = time.time() - t0

    metrics = compute_run_metrics(result, single_cfg)

    # Save .npz
    npz_filename = f"run_{run_idx:04d}.npz"
    npz_path = os.path.join(out_dir, npz_filename)

    save_dict = {
        "gap": result["gap"],
        "J": result["J"],
        "kl_pq": result["kl_pq"],
        "trunc_rate": result["trunc_rate"],
        "lag": result["lag"],
        "diag_t": result["diag_t"],
        "bias_norm": result["bias_norm"],
        "inner_update": result["inner_update"],
        "inner_track": result["inner_track"],
        "inner_bias_hist": result["inner_bias_hist"],
        "inner_noise": result["inner_noise"],
        "norm_track": result["norm_track"],
        "norm_bias_hist": result["norm_bias_hist"],
        "norm_noise": result["norm_noise"],
        "H": result["H"],
        "diverged": result["diverged"],
        "diverged_step": result["diverged_step"],
        "J_star": result["J_star"],
        "r": result["r"],
        "logp_ref": result["logp_ref"],
    }
    if "theta" in result:
        save_dict["theta"] = result["theta"]
    if "g" in result:
        save_dict["g"] = result["g"]
    if "m_hat" in result:
        save_dict["m_hat"] = result["m_hat"]

    np.savez_compressed(npz_path, **save_dict)

    summary_row = {
        "run_id": run_idx,
        "npz_file": npz_filename,
        "estimator": single_cfg["estimator"],
        "k": single_cfg["k"],
        "eta_lam": single_cfg["eta_lam"],
        "eta": single_cfg["eta"],
        "omb": single_cfg["omb"],
        "c": single_cfg["c"] if single_cfg["c"] is not None else "",
        "mismatch": single_cfg["mismatch"],
        "sim_time_sec": sim_time,
        **metrics,
    }
    if "rho_lam" in single_cfg["momentum"]:
        summary_row["rho_lam"] = single_cfg["momentum"]["rho_lam"]
        summary_row["beta_fixed"] = ""
    else:
        summary_row["rho_lam"] = ""
        summary_row["beta_fixed"] = single_cfg["momentum"]["beta_fixed"]

    return summary_row


def main():
    parser = argparse.ArgumentParser(description="stalemom experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    expanded = expand_sweep(config_data)
    total_configs = len(expanded)
    skipped_configs = [item for item in expanded if item[1]]
    runnable_configs = [item for item in expanded if not item[1]]

    print(f"Total configurations: {total_configs}")
    print(f"Runnable: {len(runnable_configs)}, Skipped: {len(skipped_configs)}")
    for _, _, reason in skipped_configs:
        print(f"  [Skipped] {reason}")

    if args.dry_run:
        print("Dry run complete.")
        return

    os.makedirs(args.out, exist_ok=True)

    # Save copy of full config
    with open(os.path.join(args.out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    commit_hash = get_git_commit()
    np_version = np.__version__

    start_time = time.time()
    summary_rows = []
    total_diverged_runs = 0

    tasks = [(idx, item[0], args.out) for idx, item in enumerate(runnable_configs)]

    if args.workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            row = run_single_task(task)
            summary_rows.append(row)
            if row["diverge_rate"] > 0:
                total_diverged_runs += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_task, task): task[0] for task in tasks}
            for fut in as_completed(futures):
                row = fut.result()
                summary_rows.append(row)
                if row["diverge_rate"] > 0:
                    total_diverged_runs += 1

    # Add skipped rows
    for idx, (cfg, _, reason) in enumerate(skipped_configs, start=len(runnable_configs)):
        row = {
            "run_id": idx,
            "npz_file": "",
            "estimator": cfg["estimator"],
            "k": cfg["k"],
            "eta_lam": cfg["eta_lam"],
            "eta": cfg["eta"],
            "omb": cfg["omb"],
            "c": cfg["c"] if cfg["c"] is not None else "",
            "mismatch": cfg["mismatch"],
            "sim_time_sec": 0.0,
            "A_T": "",
            "A_T_se": "",
            "A_T_pessimistic": "",
            "A_T_pessimistic_se": "",
            "final_gap": "",
            "final_gap_se": "",
            "final_gap_pessimistic": "",
            "final_gap_pessimistic_se": "",
            "steps_to_90": "",
            "frac_reached_90": "",
            "mean_trunc_rate": "",
            "mean_kl_pq": "",
            "diverge_rate": "",
            "skipped": True,
            "skip_reason": reason,
        }
        if "rho_lam" in cfg["momentum"]:
            row["rho_lam"] = cfg["momentum"]["rho_lam"]
            row["beta_fixed"] = ""
        else:
            row["rho_lam"] = ""
            row["beta_fixed"] = cfg["momentum"]["beta_fixed"]
        summary_rows.append(row)

    # Sort rows by run_id
    summary_rows.sort(key=lambda r: r["run_id"])

    # Write summary.csv
    csv_path = os.path.join(args.out, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    total_time = time.time() - start_time

    # Write run_log.txt
    log_path = os.path.join(args.out, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Git commit: {commit_hash}\n")
        f.write(f"NumPy version: {np_version}\n")
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"Runnable configurations: {len(runnable_configs)}\n")
        f.write(f"Skipped configurations: {len(skipped_configs)}\n")
        for _, _, reason in skipped_configs:
            f.write(f"  - {reason}\n")
        f.write(f"Configurations with diverged runs: {total_diverged_runs}\n")
        f.write(f"Total elapsed time: {total_time:.2f} s\n")

    print(f"Done in {total_time:.2f} s. Results saved to {args.out}")


if __name__ == "__main__":
    main()
