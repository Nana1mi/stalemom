"""Local end-to-end verification of the LLM training (llm/train.py) and diagnostics (llm/run_diag.py) pipeline.

Builds a tiny local Qwen2ForCausalLM model using the cached Qwen2.5/GPT2 tokenizer, runs L0 for 3 steps
with the HuggingFace rollout backend (`rollout_backend="hf"`), and runs nested checkpoint diagnostics
with S=2, R=2 to verify logs, checkpoints, and diagnostic CSVs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from llm.train import main as train_main
from llm.run_diag import run_checkpoint_diagnostics


def build_tiny_model(model_dir: Path, seed: int = 42, force_rebuild: bool = True) -> Path:
    """Create a tiny 2-layer Qwen2ForCausalLM and tokenizer in `model_dir`."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
        return model_dir

    torch.manual_seed(seed)
    # Load cached local tokenizer without network access
    tokenizer = None
    for candidate in ["Qwen/Qwen2.5-1.5B-Instruct", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError("No cached tokenizer found in ~/.cache/huggingface/hub")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)

    # Set transition weights along orthogonal basis vectors so the tiny model
    # stochastically emits "\\boxed{3}" (~50%) or "\\boxed{4}" (~50%), producing
    # non-zero rewards, advantages, gradients, and momentum buffers in 3 steps.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.02)
        model.model.norm.weight.data.fill_(1.0)
        for layer in model.model.layers:
            layer.input_layernorm.weight.data.fill_(1.0)
            layer.post_attention_layernorm.weight.data.fill_(1.0)

        tok_3 = tokenizer.encode("\\boxed{3}", add_special_tokens=False)
        tok_4 = tokenizer.encode("\\boxed{4}", add_special_tokens=False)
        # tok_3 is e.g. [t_bs, t_box, t_lbrace, t_3, t_rbrace]
        # Construct basis directions e_0 .. e_{L-1} in R^{hidden_size}
        L = len(tok_3)
        basis = torch.eye(cfg.hidden_size)[: L + 2] * 3.0

        # Default embedding for all tokens points along basis[0] -> transitions to tok_3[0]
        model.model.embed_tokens.weight.data.add_(basis[0].unsqueeze(0))
        model.lm_head.weight.data.mul_(0.01)
        model.lm_head.weight.data[tok_3[0]] = basis[0] * 3.0

        for idx in range(L - 1):
            cur_tok = tok_3[idx]
            next_tok_a = tok_3[idx + 1]
            next_tok_b = tok_4[idx + 1]
            b_vec = basis[idx + 1]
            model.model.embed_tokens.weight.data[cur_tok] = b_vec
            model.lm_head.weight.data[next_tok_a] = b_vec * 3.0
            model.lm_head.weight.data[next_tok_b] = b_vec * 3.0
            if next_tok_b != next_tok_a:
                # Also make next_tok_b transition to the closing brace
                model.model.embed_tokens.weight.data[next_tok_b] = basis[idx + 2]

        # After closing brace tok_3[-1], transition to eos_token_id
        if tokenizer.eos_token_id is not None:
            model.model.embed_tokens.weight.data[tok_3[-1]] = basis[L]
            model.lm_head.weight.data[tokenizer.eos_token_id] = basis[L] * 3.0

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def verify_local_l0(config_path: str = "llm/configs/l0/local_verify.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    model_dir = Path(cfg["model"])
    build_tiny_model(model_dir, seed=int(cfg.get("seed", 42)))

    # 1. Run 3 steps of training
    train_main(cfg)

    # 2. Run checkpoint diagnostics (S=2, R=2)
    run_checkpoint_diagnostics(cfg)

    # 3. Verify all expected outputs exist and are non-empty
    train_log = out_dir / "train_log.csv"
    summary_csv = out_dir / "summary.csv"
    diag_csv = out_dir / "diag_summary.csv"
    c3_csv = out_dir / "c3_matrix.csv"
    ckpt_dir = out_dir / "checkpoints"

    assert train_log.exists(), "train_log.csv missing"
    assert summary_csv.exists(), "summary.csv missing"
    assert diag_csv.exists(), "diag_summary.csv missing"
    assert c3_csv.exists(), "c3_matrix.csv missing"

    with open(train_log, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    assert len(train_rows) == 3, f"Expected 3 rows in train_log.csv, got {len(train_rows)}"

    for step in cfg["save_checkpoints_at"]:
        ckpt_file = ckpt_dir / f"ckpt_step_{int(step):04d}.pt"
        assert ckpt_file.exists(), f"Missing checkpoint {ckpt_file}"

    with open(diag_csv, "r", encoding="utf-8") as f:
        diag_rows = list(csv.DictReader(f))
    assert len(diag_rows) == 3, f"Expected 3 rows in diag_summary.csv, got {len(diag_rows)}"

    with open(c3_csv, "r", encoding="utf-8") as f:
        c3_rows = list(csv.DictReader(f))
    assert len(c3_rows) == 6, f"Expected 6 pairs in c3_matrix.csv, got {len(c3_rows)}"

    print("\n=== Local L0 End-to-End Verification Succeeded ===")
    print(f"train_log.csv rows: {len(train_rows)}")
    print(f"diag_summary.csv rows: {len(diag_rows)}")
    print(f"c3_matrix.csv rows: {len(c3_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="llm/configs/l0/local_verify.yaml")
    args = parser.parse_args()
    verify_local_l0(args.config)
