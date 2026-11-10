"""GSM8K 500-problem greedy evaluation helper (SPEC E5 §6.1)."""

import numpy as np
from llm.data import GSM8KItem
from llm.reward import compute_exact_match_reward


def evaluate_gsm8k_vllm(
    llm_engine,
    eval_pool: list[GSM8KItem],
    max_new_tokens: int = 512,
) -> dict[str, float]:
    """Run greedy decoding (temperature=0, max_new_tokens=512) on the 500-problem GSM8K test pool."""
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
