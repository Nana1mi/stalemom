"""Storage paths, bounded reductions and telemetry shared by LLM entry points."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def output_directory(config: dict, default: str) -> Path:
    raw = Path(config.get("output_dir", config.get("out_dir", default)))
    if config.get("_paths_resolved") or not os.environ.get("STALEMOM_OUT"):
        return raw
    if raw.is_absolute():
        raise ValueError("Use relative output_dir with STALEMOM_OUT")
    suffix = Path(*raw.parts[1:]) if raw.parts and raw.parts[0] == "results" else raw
    return Path(os.environ["STALEMOM_OUT"]) / suffix


def checkpoint_directory(config: dict, out_dir: Path) -> Path:
    if config.get("_checkpoint_output_dir"):
        return Path(config["_checkpoint_output_dir"])
    root = os.environ.get("STALEMOM_CKPT")
    if not root:
        return out_dir / "checkpoints"
    result_root = Path(os.environ.get("STALEMOM_OUT", "results"))
    try:
        suffix = out_dir.relative_to(result_root)
    except ValueError:
        suffix = Path(out_dir.name)
    return Path(root) / suffix


def check_storage() -> dict:
    """Enforce the rented-server thresholds only when WORK is configured."""
    work = os.environ.get("WORK")
    if not work:
        return {}
    data = shutil.disk_usage(work).free / 2**30
    system = shutil.disk_usage(Path(work).anchor if os.name == "nt" else "/").free / 2**30
    minimum = float(os.environ.get("STALEMOM_MIN_FREE_GIB", "15"))
    if data < minimum or system < minimum:
        raise RuntimeError(f"Storage safety stop: data free={data:.3f} GiB, system free={system:.3f} GiB")
    return {"work": work, "data_free_gib": data, "system_free_gib": system, "minimum_free_gib": minimum}


def tensor_inner64(left, right=None, chunk_elements: int = 2**20):
    """A float64 reduction with bounded scratch space, including large embeddings."""
    import torch

    a = left.detach().reshape(-1)
    b = a if right is None else right.detach().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("Inner product shape mismatch")
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for start in range(0, a.numel(), chunk_elements):
        x = a[start : start + chunk_elements].to(dtype=torch.float64)
        y = x if right is None else b[start : start + chunk_elements].to(device=a.device, dtype=torch.float64)
        total.add_(torch.sum(x * y))
    return total


def process_gpu_gib(pid: int | None = None) -> float | None:
    """Physical process memory; separate from PyTorch's sleeping-pool counters."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        wanted = os.getpid() if pid is None else pid
        values = []
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 2 and fields[0] == str(wanted):
                values.append(float(fields[1]) / 1024)
        return sum(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_sampler = None


class MemorySampler:
    def __init__(self, log_path: Path, interval: float = 0.25):
        self.path = log_path
        self.interval = interval
        self.peak = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        import json

        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop.is_set():
                value = process_gpu_gib()
                if value is not None:
                    self.peak = max(self.peak or 0.0, value)
                    stream.write(json.dumps({"time": time.time(), "pid": os.getpid(), "gpu_gib": value}) + "\n")
                    stream.flush()
                self.stop.wait(self.interval)

    def __enter__(self):
        global _sampler
        _sampler = self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        global _sampler
        self.stop.set()
        self.thread.join(timeout=6)
        _sampler = None


def physical_peak_gib() -> float | None:
    if _sampler is not None:
        current = process_gpu_gib()
        if current is not None:
            _sampler.peak = max(_sampler.peak or 0.0, current)
        return _sampler.peak
    return process_gpu_gib()
