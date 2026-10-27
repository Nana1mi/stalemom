"""Script to directly parse results/e2/summary.csv and paired_summary.csv
and regenerate the exact Section 2.1 table.
"""
import csv
import os
import numpy as np


def generate_table(e2_dir: str = "results/e2"):
    summary_path = os.path.join(e2_dir, "summary.csv")
    paired_path = os.path.join(e2_dir, "paired_summary.csv")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    paired_map = {}
    if os.path.exists(paired_path):
        with open(paired_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # key by (k, eta_lam, momentum)
                key = (int(r["k"]), float(r["eta_lam"]), r["momentum"])
                paired_map[key] = r

    betas = [0.0, 0.6, 0.9, 0.99]
    estimators = ["exact", "is", "tis", "tis_debiased", "tis_mean"]

    print("| β | ρ = η/(1−β) | exact | is | tis | tis_debiased | tis_mean | 确定性偏差效应 (tis_mean − exact) | 随机配对偏差效应 (tis − tis_debiased) ± SE |")
    print("|---|---|---|---|---|---|---|---|---|")

    table_data = []

    for b in betas:
        eta_val = 0.1
        k_val = 16
        mom_key = f"beta_{b}"
        rho = eta_val / (1.0 - b) if b < 1.0 else float("inf")

        res = {}
        for est in estimators:
            k_req = 0 if est == "exact" else k_val
            match = [
                r for r in summary_rows
                if r["estimator"] == est
                and int(r["k"]) == k_req
                and abs(float(r["eta_lam"]) - eta_val) < 1e-4
                and abs(float(r["beta_fixed"]) - b) < 1e-4
            ]
            if match:
                res[est] = float(match[0]["A_T_pessimistic"])

        p_info = paired_map.get((k_val, eta_val, mom_key), {})
        det_bias = float(p_info.get("det_bias_effect", res.get("tis_mean", 0) - res.get("exact", 0)))
        stoch_bias = float(p_info.get("stoch_bias_effect", 0.0))
        stoch_se = float(p_info.get("stoch_bias_se", 0.0))

        row_str = (
            f"| {b:g} | {rho:.2f} | "
            f"{res.get('exact', float('nan')):.5f} | "
            f"{res.get('is', float('nan')):.5f} | "
            f"{res.get('tis', float('nan')):.5f} | "
            f"{res.get('tis_debiased', float('nan')):.5f} | "
            f"{res.get('tis_mean', float('nan')):.5f} | "
            f"{det_bias:+.5f} | "
            f"{stoch_bias:+.4f} ± {stoch_se:.4f} |"
        )
        print(row_str)
        table_data.append(row_str)

    return table_data


if __name__ == "__main__":
    generate_table()
