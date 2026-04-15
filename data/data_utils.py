"""
data/data_utils.py
==================
Dataset loading and preprocessing utilities for the llm-alignment project.

All three training stages (SFT, Reward Model, DPO) consume the same underlying
Anthropic/hh-rlhf dataset but require it formatted differently:

      format_for_sft   format_for_rm   format_for_dpo
           |                │                │
      Single string    Pair of strings  Triplet dict
      (chosen only)    (chosen, rejected) (prompt, chosen, rejected)
"""

from __future__ import annotations

from typing import Optional, cast

from datasets import Dataset, load_dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_hh_rlhf(
    split: str = "train",
    subset: str = "harmless-base",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load the Anthropic/hh-rlhf dataset from the HuggingFace Hub.

    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    dataset = cast(Dataset, load_dataset(
        "Anthropic/hh-rlhf",
        data_dir=subset,
        split=split,
        cache_dir=cache_dir,
        verification_mode="no_checks",
    ))

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return dataset


def format_for_sft(example: dict) -> dict:
    """
    Reformat a raw HH-RLHF record for Supervised Fine-Tuning.

    """
    return {"text": example["chosen"].strip() + "<|endoftext|>"}



def format_for_reward_model(example: dict) -> dict:
    """
    Reformat a raw HH-RLHF record for Reward Model training.
    """
    chosen_str   = example["chosen"].strip()
    rejected_str = example["rejected"].strip()

    return {"chosen": chosen_str, "rejected": rejected_str}


def format_for_dpo(example: dict) -> Optional[dict]:
    """
    Reformat a raw HH-RLHF record for DPO training.

    """
    SEPARATOR = "\n\nAssistant:"

    chosen_parts   = example["chosen"].rsplit(SEPARATOR, maxsplit=1)
    rejected_parts = example["rejected"].rsplit(SEPARATOR, maxsplit=1)

    # Malformed example: separator not found in one or both strings — skip it.
    # Call dataset.filter(lambda x: x is not None) after mapping to drop these.
    if len(chosen_parts) != 2 or len(rejected_parts) != 2:
        return None

    chosen_prompt, chosen_response     = chosen_parts
    _,             rejected_response   = rejected_parts

    return {
        "prompt":   chosen_prompt + SEPARATOR,
        "chosen":   " " + chosen_response.strip(),
        "rejected": " " + rejected_response.strip(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_prompt_and_response(text: str) -> tuple[str, str]:
    """
    Split a full HH-RLHF conversation string into (prompt, response).
    """
    SEPARATOR = "\n\nAssistant:"

    split_parts = text.rsplit(SEPARATOR, maxsplit=1)

    if len(split_parts) != 2:
        raise ValueError(
            f"Separator '{SEPARATOR!r}' not found in text: {text[:100]!r}"
        )

    human_turns, final_response = split_parts

    return human_turns + SEPARATOR, final_response.strip()
