"""GPU PyTorch CountSketch and online nested gradient diagnostics for LLM checkpoints (SPEC E5 §5, §9, §10)."""

from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from llm.runtime import tensor_inner64


class CountSketch:
    """GPU PyTorch CountSketch (§5.5) for cross-checkpoint and jackknife inner products.

    - Seeds a `torch.Generator` deterministically from parameter name SHA-256.
    - Generates `int32` bucket indices and +/-1 signs on-the-fly on GPU without caching.
    - Accumulates sketch values in GPU `fp32` using `index_add_`.
    - Computes inner products in `float64`.
    """

    def __init__(self, D: int = 2**22, device: str | torch.device | None = None):
        self.D = int(D)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def _name_to_seed(name: str) -> int:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        # Positive 63-bit integer seed for torch.Generator.manual_seed
        return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF

    def _generate_buckets_and_signs(
        self, name: str, numel: int, dev: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate int32 bucket indices and float32 +/-1 signs on-the-fly (no cache)."""
        seed = self._name_to_seed(name)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        buckets = torch.randint(0, self.D, (numel,), generator=gen, device=dev, dtype=torch.int32)
        signs = (
            torch.randint(0, 2, (numel,), generator=gen, device=dev, dtype=torch.int32)
            .to(dtype=torch.float32)
            .mul_(2.0)
            .sub_(1.0)
        )
        return buckets, signs

    @torch.no_grad()
    def sketch_tensor(
        self,
        name: str,
        tensor: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sketch a single tensor with parameter `name` into a D-dim fp32 GPU tensor using index_add_."""
        if not isinstance(tensor, torch.Tensor):
            t = torch.as_tensor(tensor, dtype=torch.float32, device=self.device)
        else:
            t = tensor.detach().to(device=self.device, dtype=torch.float32)
        flat = t.reshape(-1)
        numel = flat.numel()
        if out is None:
            out = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        if numel == 0:
            return out
        buckets, signs = self._generate_buckets_and_signs(name, numel, self.device)
        out.index_add_(0, buckets, flat * signs)
        return out

    @torch.no_grad()
    def sketch_dict(self, named_tensors: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
        """Sketch a dictionary of `{param_name: tensor}` into a single D-dim fp32 tensor on `self.device`."""
        out = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        for name, arr in named_tensors.items():
            self.sketch_tensor(name, arr, out=out)
        return out

    @torch.no_grad()
    def sketch_vector(self, vec: torch.Tensor | np.ndarray, name: str = "default") -> torch.Tensor:
        """Sketch a 1D vector."""
        return self.sketch_tensor(name, vec)

    @staticmethod
    @torch.no_grad()
    def inner(sketch_u: torch.Tensor | np.ndarray, sketch_v: torch.Tensor | np.ndarray) -> float:
        """Compute inner product <sketch_u, sketch_v> in float64."""
        if isinstance(sketch_u, torch.Tensor) and isinstance(sketch_v, torch.Tensor):
            return float(torch.dot(sketch_u.to(torch.float64), sketch_v.to(torch.float64)).item())
        u64 = np.asarray(sketch_u, dtype=np.float64)
        v64 = np.asarray(sketch_v, dtype=np.float64)
        return float(np.dot(u64, v64))


@dataclass
class SetSummaryRecord:
    """Per-set summary stored for §5.4 leave-one-set-out jackknife."""
    set_idx: int
    n_si: np.ndarray          # shape (R,), float64
    d_si: np.ndarray          # shape (R,), float64
    norm_sq_Ts: float         # ||T_s||^2, float64
    sketch_Ts: torch.Tensor   # shape (D,), float32 CountSketch of T_s


class OnlineLLMDiagAccumulator:
    """Online accumulator implementing SPEC E5 §5.1–§5.5 using GPU fp32 vector accumulation and float64 scalars.

    - Gradient vector accumulators (`_cur_Ts`, `sum_A`, `sum_B`) are GPU `fp32` (`torch.float32`) tensors.
    - All scalar quantities (`n_{s,i}`, `d_{s,i}`, `||T_s||^2`, `H`, `M`, `SNR`, `C1`, `C2_hat`, `C1_pred`, jackknife SEs)
      are accumulated and computed in `float64`.
    """

    def __init__(
        self,
        S: int,
        R: int,
        norm_m: float,
        sketch_dim: int = 2**22,
        device: str | torch.device | None = None,
    ):
        if S < 2 or S % 2 != 0:
            raise ValueError(f"S must be an even integer >= 2, got {S}")
        if R < 2:
            raise ValueError(f"R must be >= 2, got {R}")
        self.S = int(S)
        self.R = int(R)
        self.norm_m = float(norm_m)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.sketch = CountSketch(D=sketch_dim, device=self.device)

        self.set_records: list[SetSummaryRecord] = []
        self.sum_A: dict[str, torch.Tensor] = {}
        self.sum_B: dict[str, torch.Tensor] = {}

        self._cur_s: int | None = None
        self._cur_n: list[float] = []
        self._cur_d: list[float] = []
        self._cur_Ts: dict[str, torch.Tensor] = {}

    def begin_set(self, s: int):
        if self._cur_s is not None:
            raise RuntimeError(f"Set {self._cur_s} was not finalized before starting set {s}")
        self._cur_s = int(s)
        self._cur_n = []
        self._cur_d = []
        self._cur_Ts = {}

    @torch.no_grad()
    def add_batch_grad(
        self,
        grad_dict: dict[str, torch.Tensor | np.ndarray],
        m_dict: dict[str, torch.Tensor | np.ndarray] | None = None,
    ):
        """Accumulate a single batch gradient g_{s,i} (vectors in GPU fp32, scalars in float64)."""
        if self._cur_s is None:
            raise RuntimeError("Must call begin_set(s) before add_batch_grad")

        n_val = 0.0
        d_val = 0.0
        for name, g in grad_dict.items():
            if not isinstance(g, torch.Tensor):
                g_fp32 = torch.as_tensor(g, dtype=torch.float32, device=self.device)
            else:
                g_fp32 = g.detach().to(device=self.device, dtype=torch.float32)

            # Scalar norm squared accumulated in float64
            n_val += float(tensor_inner64(g_fp32).item())

            if m_dict is not None and name in m_dict and self.norm_m > 0.0:
                m_t = m_dict[name]
                if not isinstance(m_t, torch.Tensor):
                    m_fp32 = torch.as_tensor(m_t, dtype=torch.float32, device=self.device)
                else:
                    m_fp32 = m_t.detach().to(device=self.device, dtype=torch.float32)
                d_val += float(tensor_inner64(m_fp32, g_fp32).item())

            # Vector accumulation in GPU fp32
            if name not in self._cur_Ts:
                self._cur_Ts[name] = g_fp32.clone()
            else:
                self._cur_Ts[name].add_(g_fp32)

        self._cur_n.append(float(n_val))
        self._cur_d.append(float(d_val) if self.norm_m > 0.0 else float("nan"))

    @torch.no_grad()
    def end_set(self):
        """Finalize current prompt-set s: record ||T_s||^2 (float64), sketch(T_s) (GPU fp32), and add T_s to A or B."""
        if self._cur_s is None:
            raise RuntimeError("No active set to finalize")
        if len(self._cur_n) != self.R:
            raise ValueError(f"Expected R={self.R} rollouts in set {self._cur_s}, got {len(self._cur_n)}")

        norm_sq_Ts = 0.0
        target_half = self.sum_A if self._cur_s < (self.S // 2) else self.sum_B
        for name, ts in self._cur_Ts.items():
            norm_sq_Ts += float(tensor_inner64(ts).item())
            if name not in target_half:
                target_half[name] = ts.clone()
            else:
                target_half[name].add_(ts)

        sketch_Ts = self.sketch.sketch_dict(self._cur_Ts)
        self.set_records.append(
            SetSummaryRecord(
                set_idx=self._cur_s,
                n_si=np.array(self._cur_n, dtype=np.float64),
                d_si=np.array(self._cur_d, dtype=np.float64),
                norm_sq_Ts=float(norm_sq_Ts),
                sketch_Ts=sketch_Ts,
            )
        )
        self._cur_s = None
        self._cur_Ts = {}

    @torch.no_grad()
    def finalize(self) -> dict:
        """Compute all §5.3 estimators, §5.4 jackknife SEs, and §5.5 sketch accuracy verification in float64."""
        if len(self.set_records) != self.S:
            raise ValueError(f"Expected S={self.S} finalized sets, got {len(self.set_records)}")

        half = self.S // 2
        N_A = float(half * self.R)
        N_B = float((self.S - half) * self.R)

        # Exact <A_bar, B_bar> from the two GPU fp32 half-sums, reduced in float64
        dot_AB_exact = 0.0
        for name in self.sum_A:
            dot_AB_exact += float(tensor_inner64(self.sum_A[name], self.sum_B[name]).item())
        H_exact_split = float(dot_AB_exact / (N_A * N_B))

        # Sketch-based A_bar and B_bar (accumulated in fp32, inner product in float64)
        sketch_A_sum = torch.zeros(self.sketch.D, dtype=torch.float32, device=self.device)
        for rec in self.set_records[:half]:
            sketch_A_sum.add_(rec.sketch_Ts)
        sketch_A = sketch_A_sum / float(N_A)

        sketch_B_sum = torch.zeros(self.sketch.D, dtype=torch.float32, device=self.device)
        for rec in self.set_records[half:]:
            sketch_B_sum.add_(rec.sketch_Ts)
        sketch_B = sketch_B_sum / float(N_B)

        H_sketch_split = CountSketch.inner(sketch_A, sketch_B)
        sketch_rel_err = (
            abs(H_sketch_split - H_exact_split) / abs(H_exact_split)
            if abs(H_exact_split) > 1e-12
            else abs(H_sketch_split - H_exact_split)
        )

        n_mat = np.stack([rec.n_si for rec in self.set_records], axis=0)   # (S, R) float64
        d_mat = np.stack([rec.d_si for rec in self.set_records], axis=0)   # (S, R) float64
        norm_sq_Ts = np.array([rec.norm_sq_Ts for rec in self.set_records], dtype=np.float64)

        N_bar = float(np.mean(n_mat))
        sum_n_s = np.sum(n_mat, axis=1)
        M_s = (norm_sq_Ts - sum_n_s) / float(self.R * (self.R - 1))
        M = float(np.mean(M_s))

        H = H_exact_split
        var_total = N_bar - H
        var_rollout = N_bar - M
        var_prompt = M - H
        snr = float(H / var_total) if var_total > 0 else float("nan")

        if self.norm_m > 0.0:
            inner_m_h = float(np.mean(d_mat))
            safe_n = np.where(n_mat > 0.0, n_mat, 1.0)
            inv_norm_g = np.where(n_mat > 0.0, 1.0 / np.sqrt(safe_n), 0.0)
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

        sketch_A_np = sketch_A.detach().cpu().numpy().astype(np.float32)
        sketch_B_np = sketch_B.detach().cpu().numpy().astype(np.float32)

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
            "A_bar": sketch_A_np,
            "B_bar": sketch_B_np,
        }

        # §5.4 Leave-one-set-out jackknife using saved scalars and GPU fp32 CountSketch of T_s
        jk_keys = [
            "N_bar", "H", "H_hat", "M", "var_total", "var_rollout",
            "var_prompt", "SNR", "SNR_hat", "inner_m_h", "C1", "C2_hat", "C1_pred", "C1_pred_legacy",
        ]
        jk_vals = {k: [] for k in jk_keys}

        if half >= 2 and (self.S - half) >= 2:
            for leave_out in range(self.S):
                keep = [idx for idx in range(self.S) if idx != leave_out]
                n_sub = n_mat[keep]
                d_sub = d_mat[keep]
                M_s_sub = M_s[keep]

                if leave_out < half:
                    A_sub = (sketch_A_sum - self.set_records[leave_out].sketch_Ts) / float((half - 1) * self.R)
                    B_sub = sketch_B
                else:
                    A_sub = sketch_A
                    B_sub = (sketch_B_sum - self.set_records[leave_out].sketch_Ts) / float((self.S - half - 1) * self.R)

                H_sub = CountSketch.inner(A_sub, B_sub)
                N_sub = float(np.mean(n_sub))
                M_sub = float(np.mean(M_s_sub))
                vt_sub = N_sub - H_sub
                vr_sub = N_sub - M_sub
                vp_sub = M_sub - H_sub
                snr_sub = float(H_sub / vt_sub) if vt_sub > 0 else float("nan")

                if self.norm_m > 0.0:
                    imh_sub = float(np.mean(d_sub))
                    safe_n_sub = np.where(n_sub > 0.0, n_sub, 1.0)
                    inv_ng_sub = np.where(n_sub > 0.0, 1.0 / np.sqrt(safe_n_sub), 0.0)
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
        else:
            for k in jk_keys:
                res[f"{k}_jk_se"] = float("nan")

        return res
