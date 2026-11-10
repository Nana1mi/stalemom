"""GSM8K greedy evaluation helper supporting both vLLM and HuggingFace backends (SPEC E5 §6.1)."""

import numpy as np
import torch

from llm.data import GSM8KItem
from llm.reward import compute_exact_match_reward


def evaluate_gsm8k_vllm(
    llm_engine,
    eval_pool: list[GSM8KItem],
    max_new_tokens: int = 512,
) -> dict[str, float]:
    """Run greedy decoding (temperature=0, max_new_tokens=512) on GSM8K test pool via vLLM."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=1,
    )
    prompts = [item.prompt_text for item in eval_pool]
    outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=False)

    rewards = []
    resp_lens = []
    for item, out in zip(eval_pool, outputs):
        comp = out.outputs[0]
        r = compute_exact_match_reward(comp.text, item.ground_truth)
        rewards.append(r)
        resp_lens.append(len(comp.token_ids))

    return {
        "eval_acc": float(np.mean(rewards)) if rewards else 0.0,
        "eval_resp_len_mean": float(np.mean(resp_lens)) if resp_lens else 0.0,
    }


@torch.no_grad()
def evaluate_gsm8k_hf(
    model,
    tokenizer,
    eval_pool: list[GSM8KItem],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Run greedy decoding (do_sample=False) on GSM8K test pool via HuggingFace `model.generate`."""
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rewards = []
    resp_lens = []
    for start in range(0, len(eval_pool), batch_size):
        batch_items = eval_pool[start : start + batch_size]
        for item in batch_items:
            enc = tokenizer(item.prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            prompt_len = input_ids.shape[1]
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            resp_ids = gen_ids[0, prompt_len:]
            text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            rewards.append(compute_exact_match_reward(text, item.ground_truth))
            resp_lens.append(int(resp_ids.numel()))

    if was_training:
        model.train()

    return {
        "eval_acc": float(np.mean(rewards)) if rewards else 0.0,
        "eval_resp_len_mean": float(np.mean(resp_lens)) if resp_lens else 0.0,
    }
