"""GSM8K dataset loading and prompt formatting for Qwen/Qwen3-0.6B (SPEC E5 §6.1)."""

from dataclasses import dataclass
import numpy as np

from llm.reward import extract_gsm8k_ground_truth


SYSTEM_PROMPT = (
    "Solve the following math word problem step by step. "
    "Put your final numerical answer inside \\boxed{}."
)


@dataclass
class GSM8KItem:
    idx: int
    question: str
    ground_truth: str
    prompt_text: str


def format_qwen3_prompt(question: str, tokenizer=None) -> str:
    """Format a GSM8K question using Qwen3 chat template with enable_thinking=False (§6.1)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_synthetic_math_pools(
    tokenizer=None,
    train_pool_size: int = 1024,
    eval_pool_size: int = 500,
    seed: int = 42,
) -> tuple[list[GSM8KItem], list[GSM8KItem]]:
    """Build deterministic arithmetic word problem pools for local offline debugging."""
    rng = np.random.default_rng(seed)

    def _make_pool(n: int, offset: int) -> list[GSM8KItem]:
        items = []
        pairs = [(1, 2, "3"), (2, 1, "3"), (3, 0, "3"), (1, 3, "4")]
        for i in range(n):
            a, b, ans = pairs[int(rng.integers(0, len(pairs)))]
            q = f"Problem #{offset + i}: What is {a} + {b}? Put your final answer in \\boxed{{}}."
            items.append(
                GSM8KItem(
                    idx=offset + i,
                    question=q,
                    ground_truth=ans,
                    prompt_text=format_qwen3_prompt(q, tokenizer),
                )
            )
        return items

    return _make_pool(train_pool_size, 0), _make_pool(eval_pool_size, 10000)


def load_gsm8k_pools(
    tokenizer=None,
    train_pool_size: int = 1024,
    eval_pool_size: int = 500,
    seed: int = 42,
    dataset_name: str = "openai/gsm8k",
) -> tuple[list[GSM8KItem], list[GSM8KItem]]:
    """Load fixed GSM8K train pool (1024 problems) and test pool (500 problems) (§6.1).

    Supports `dataset_name="synthetic"` for fast local end-to-end debugging without network download.
    """
    if dataset_name == "synthetic":
        return build_synthetic_math_pools(tokenizer, train_pool_size, eval_pool_size, seed)

    from datasets import load_dataset

    ds = load_dataset(dataset_name, "main")
    train_ds = ds["train"]
    test_ds = ds["test"]

    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_ds), size=min(train_pool_size, len(train_ds)), replace=False)
    eval_indices = rng.choice(len(test_ds), size=min(eval_pool_size, len(test_ds)), replace=False)

    train_pool = []
    for i in train_indices:
        row = train_ds[int(i)]
        gt = extract_gsm8k_ground_truth(row["answer"]) or "0"
        train_pool.append(
            GSM8KItem(
                idx=int(i),
                question=row["question"],
                ground_truth=gt,
                prompt_text=format_qwen3_prompt(row["question"], tokenizer),
            )
        )

    eval_pool = []
    for i in eval_indices:
        row = test_ds[int(i)]
        gt = extract_gsm8k_ground_truth(row["answer"]) or "0"
        eval_pool.append(
            GSM8KItem(
                idx=int(i),
                question=row["question"],
                ground_truth=gt,
                prompt_text=format_qwen3_prompt(row["question"], tokenizer),
            )
        )

    return train_pool, eval_pool
