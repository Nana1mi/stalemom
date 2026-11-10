"""Nested gradient diagnostics and CountSketch for LLM checkpoints (SPEC E5 §5, §9, §10)."""

from dataclasses import dataclass, field
import numpy as np

from stalemom.diag import CountSketch, nested_stats, compute_c3


@dataclass
class SetSummaryRecord:
    """Per-set summary stored for §5.4 leave-one-set-out jackknife without keeping 2.4GB gradients."""
    set_idx: int
    n_si: np.ndarray          # shape (R,)
    d_si: np.ndarray          # shape (R,)
    norm_sq_Ts: float         # ||T_s||^2
    sketch_Ts: np.ndarray     # shape (D,), float32 CountSketch of T_s


class OnlineLLMDiagAccumulator:
    """Online accumulator implementing SPEC E5 §5.1–§5.5 for large parameter dictionaries.

    Instead of holding S*R full gradient vectors in memory:
    - Holds only one set sum `T_s` and two half-sums `sum_A` and `sum_B` (on CPU fp32) to compute exact H = <A_bar, B_bar>.
    - Stores a `CountSketch` (default D = 2^22, ~16MB) for each set `T_s` to compute leave-one-set-out jackknife SEs
      and cross-checkpoint C3 without storing 16 * 2.4GB = 38GB of raw gradients.
    """

    def __init__(
        self,
        S: int,
        R: int,
        norm_m: float,
        sketch_dim: int = 2**22,
    ):
        if S < 2 or S % 2 != 0:
            raise ValueError(f"S must be an even integer >= 2, got {S}")
        if R < 2:
            raise ValueError(f"R must be >= 2, got {R}")
        self.S = S
        self.R = R
        self.norm_m = float(norm_m)
        self.sketch = CountSketch(D=sketch_dim)

        self.set_records: list[SetSummaryRecord] = []
        self.sum_A: dict[str, np.ndarray] = {}
        self.sum_B: dict[str, np.ndarray] = {}

        # Current set state
        self._cur_s: int | None = None
        self._cur_n: list[float] = []
        self._cur_d: list[float] = []
        self._cur_Ts: dict[str, np.ndarray] = {}

    def begin_set(self, s: int):
        if self._cur_s is not None:
            raise RuntimeError(f"Set {self._cur_s} was not finalized before starting set {s}")
        self._cur_s = s
        self._cur_n = []
        self._cur_d = []
        self._cur_Ts = {}

    def add_batch_grad(
        self,
        grad_dict: dict[str, np.ndarray],
        m_dict: dict[str, np.ndarray] | None = None,
    ):
        """Accumulate a single batch gradient g_{s,i} inside the active prompt-set s."""
        if self._cur_s is None:
            raise RuntimeError("Must call begin_set(s) before add_batch_grad")

        n_val = 0.0
        d_val = 0.0
        for name, g in grad_dict.items():
            g64 = np.asarray(g, dtype=np.float64)
            n_val += float(np.sum(g64 * g64))
            if m_dict is not None and name in m_dict and self.norm_m > 0.0:
                m64 = np.asarray(m_dict[name], dtype=np.float64)
                d_val += float(np.sum(m64 * g64))
            if name not in self._cur_Ts:
                self._cur_Ts[name] = g64.copy()
            else:
                self._cur_Ts[name] += g64

        self._cur_n.append(n_val)
        self._cur_d.append(d_val if self.norm_m > 0.0 else float("nan"))

    def end_set(self):
        """Finalize current prompt-set s: record ||T_s||^2, sketch(T_s), and accumulate into A or B."""
        if self._cur_s is None:
            raise RuntimeError("No active set to finalize")
        if len(self._cur_n) != self.R:
            raise ValueError(f"Expected R={self.R} rollouts in set {self._cur_s}, got {len(self._cur_n)}")

        norm_sq_Ts = 0.0
        target_half = self.sum_A if self._cur_s < (self.S // 2) else self.sum_B
        for name, ts in self._cur_Ts.items():
            norm_sq_Ts += float(np.sum(ts * ts))
            if name not in target_half:
                target_half[name] = ts.copy()
            else:
                target_half[name] += ts

        sketch_Ts = self.sketch.sketch_dict(self._cur_Ts)
        self.set_records.append(
            SetSummaryRecord(
                set_idx=self._cur_s,
                n_si=np.array(self._cur_n, dtype=np.float64),
                d_si=np.array(self._cur_d, dtype=np.float64),
                norm_sq_Ts=norm_sq_Ts,
                sketch_Ts=sketch_Ts,
            )
        )
        self._cur_s = None
        self._cur_Ts = {}

    def finalize(self) -> dict:
        """Compute all §5.3 estimators, §5.4 jackknife SEs, and §5.5 sketch accuracy verification."""
        if len(self.set_records) != self.S:
            raise ValueError(f"Expected S={self.S} finalized sets, got {len(self.set_records)}")

        half = self.S // 2
        N_A = half * self.R
        N_B = (self.S - half) * self.R

        # Exact <A_bar, B_bar> from the two half-sums
        dot_AB_exact = 0.0
        for name in self.sum_A:
            dot_AB_exact += float(np.sum(self.sum_A[name] * self.sum_B[name]))
        H_exact_split = dot_AB_exact / (N_A * N_B)

        # Sketch-based A_bar and B_bar
        sketch_A = np.sum([rec.sketch_Ts for rec in self.set_records[:half]], axis=0, dtype=np.float64) / N_A
        sketch_B = np.sum([rec.sketch_Ts for rec in self.set_records[half:]], axis=0, dtype=np.float64) / N_B
        H_sketch_split = float(np.dot(sketch_A, sketch_B))

        sketch_rel_err = (
            abs(H_sketch_split - H_exact_split) / abs(H_exact_split)
            if abs(H_exact_split) > 1e-12
            else abs(H_sketch_split - H_exact_split)
        )

        n_mat = np.stack([rec.n_si for rec in self.set_records], axis=0)   # (S, R)
        d_mat = np.stack([rec.d_si for rec in self.set_records], axis=0)   # (S, R)
        norm_sq_Ts = np.array([rec.norm_sq_Ts for rec in self.set_records], dtype=np.float64)

        N_bar = float(np.mean(n_mat))
        sum_n_s = np.sum(n_mat, axis=1)
        M_s = (norm_sq_Ts - sum_n_s) / (self.R * (self.R - 1))
        M = float(np.mean(M_s))

        H = H_exact_split
        var_total = N_bar - H
        var_rollout = N_bar - M
        var_prompt = M - H
        snr = float(H / var_total) if var_total > 0 else float("nan")

        if self.norm_m > 0.0:
            inner_m_h = float(np.mean(d_mat))
            inv_norm_g = np.where(n_mat > 0.0, 1.0 / np.sqrt(np.maximum(n_mat, 0.0)), 0.0)
            c1 = float(np.mean((d_mat / self.norm_m) * inv_norm_g))
            c1_pred = float(inner_m_h / (self.norm_m * np.sqrt(N_bar))) if N_bar > 0.0 else float("nan")
            c1_pred_legacy = float((inner_m_h / self.norm_m) * np.mean(inv_norm_g))
            c2_hat = float(inner_m_h / (self.norm_m * np.sqrt(H))) if H > 0.0 else float("nan")
        else:
            inner_m_h = float("nan")
            c1 = float("nan")
            c1_pred = float("nan")
            c1_pred_legacy = float("nan")
            c2_hat = float("nan")

        res = {
            "N_bar": N_bar,
            "H": H,
            "H_hat": H,
            "H_sketch": H_sketch_split,
            "sketch_rel_err": float(sketch_rel_err),
            "M": M,
            "var_total": var_total,
            "var_rollout": var_rollout,
            "var_prompt": var_prompt,
            "SNR": snr,
            "SNR_hat": snr,
            "inner_m_h": inner_m_h,
            "C1": c1,
            "C2_hat": c2_hat,
            "C1_pred": c1_pred,
            "C1_pred_legacy": c1_pred_legacy,
            "A_bar": sketch_A.astype(np.float32),
            "B_bar": sketch_B.astype(np.float32),
        }

        # §5.4 Leave-one-set-out jackknife using saved scalars and CountSketch of T_s
        sketches_Ts = np.stack([rec.sketch_Ts.astype(np.float64) for rec in self.set_records], axis=0)
        tot_sketch_A = sketch_A * N_A
        tot_sketch_B = sketch_B * N_B

        jk_keys = [
            "N_bar", "H", "H_hat", "M", "var_total", "var_rollout",
            "var_prompt", "SNR", "SNR_hat", "inner_m_h", "C1", "C2_hat", "C1_pred", "C1_pred_legacy",
        ]
        jk_vals = {k: [] for k in jk_keys}

        for leave_out in range(self.S):
            keep = [idx for idx in range(self.S) if idx != leave_out]
            n_sub = n_mat[keep]
            d_sub = d_mat[keep]
            M_s_sub = M_s[keep]

            if leave_out < half:
                A_sub = (tot_sketch_A - sketches_Ts[leave_out]) / ((half - 1) * self.R)
                B_sub = sketch_B
            else:
                A_sub = sketch_A
                B_sub = (tot_sketch_B - sketches_Ts[leave_out]) / ((self.S - half - 1) * self.R)

            H_sub = float(np.dot(A_sub, B_sub))
            N_sub = float(np.mean(n_sub))
            M_sub = float(np.mean(M_s_sub))
            vt_sub = N_sub - H_sub
            vr_sub = N_sub - M_sub
            vp_sub = M_sub - H_sub
            snr_sub = float(H_sub / vt_sub) if vt_sub > 0 else float("nan")

            if self.norm_m > 0.0:
                imh_sub = float(np.mean(d_sub))
                inv_ng_sub = np.where(n_sub > 0.0, 1.0 / np.sqrt(np.maximum(n_sub, 0.0)), 0.0)
                c1_sub = float(np.mean((d_sub / self.norm_m) * inv_ng_sub))
                c1p_sub = float(imh_sub / (self.norm_m * np.sqrt(N_sub))) if N_sub > 0.0 else float("nan")
                c1pl_sub = float((imh_sub / self.norm_m) * np.mean(inv_ng_sub))
                c2_sub = float(imh_sub / (self.norm_m * np.sqrt(H_sub))) if H_sub > 0.0 else float("nan")
            else:
                imh_sub = c1_sub = c1p_sub = c1pl_sub = c2_sub = float("nan")

            sub_map = {
                "N_bar": N_sub, "H": H_sub, "H_hat": H_sub, "M": M_sub,
                "var_total": vt_sub, "var_rollout": vr_sub, "var_prompt": vp_sub,
                "SNR": snr_sub, "SNR_hat": snr_sub, "inner_m_h": imh_sub,
                "C1": c1_sub, "C2_hat": c2_sub, "C1_pred": c1p_sub, "C1_pred_legacy": c1pl_sub,
            }
            for k in jk_keys:
                jk_vals[k].append(sub_map[k])

        for k in jk_keys:
            arr = np.array(jk_vals[k], dtype=np.float64)
            valid = arr[np.isfinite(arr)]
            if len(valid) >= 2:
                m_val = np.mean(valid)
                n_v = len(valid)
                se = float(np.sqrt(((n_v - 1) / n_v) * np.sum((valid - m_val) ** 2)))
            else:
                se = float("nan")
            res[f"{k}_jk_se"] = se

        return res
