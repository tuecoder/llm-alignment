"""
rlhf/train_reward_model.py
==========================
Phase 2 of the RLHF pipeline: Reward Model Training.

The reward model (RM) learns to assign a higher scalar score to the chosen
(preferred) response than to the rejected (dispreferred) response, using the
Bradley-Terry ranking loss
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from data.data_utils import format_for_reward_model, load_hh_rlhf
from rlhf.reward_model import RewardModel
from utils.scheduler import get_scheduler
from utils.trainer_utils import AverageMeter, get_parameter_groups, log_metrics, save_checkpoint

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args() -> dict:
    """
    Parse CLI arguments and merge with the YAML config.

    """
    parser = argparse.ArgumentParser(description="Phase 1: Train Reward model")
    parser.add_argument("-c", "--config", default="configs/rlhf_config.yaml",
                        help="path to YAML config file")
    
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    return cfg

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict], tokenizer, max_length: int) -> dict:
    """
    Tokenise and batch chosen/rejected pairs for the reward model.

    Each example in ``batch`` has keys ``"chosen"`` and ``"rejected"``.
    We tokenise both, producing four tensors per batch:

        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask

    """
    # TODO: Extract all chosen and rejected strings from the batch list.
    chosen_text   = [x["chosen"]   for x in batch]
    rejected_text = [x["rejected"] for x in batch]

    chosen_enc = tokenizer(
        chosen_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids":        chosen_enc["input_ids"],
        "chosen_attention_mask":   chosen_enc["attention_mask"],
        "rejected_input_ids":      rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }


def build_dataloaders(cfg: dict, tokenizer) -> tuple[DataLoader, DataLoader]:
    """
    Build train and eval DataLoaders for reward model training.

    """
    batch_size      = cfg["reward_model"]["per_device_train_batch_size"]
    eval_batch_size = cfg["reward_model"]["per_device_eval_batch_size"]
    max_length      = cfg["model"]["max_length"]

    train_raw = load_hh_rlhf(
        split="train",
        subset=cfg["data"]["dataset_subset"],
        max_samples=cfg["data"]["max_train_samples"],
    )
    eval_raw = load_hh_rlhf(
        split="test",
        subset=cfg["data"]["dataset_subset"],
        max_samples=cfg["data"]["max_eval_samples"],
    )

    # format_for_reward_model returns {"chosen": str, "rejected": str}.
    # No pre-tokenization needed — collate_fn handles tokenization per batch.
    train_ds = train_raw.map(format_for_reward_model)
    eval_ds  = eval_raw.map(format_for_reward_model)

    # Wrap collate_fn to close over tokenizer and max_length.
    _collate = lambda batch: collate_fn(batch, tokenizer, max_length)

    return (
        DataLoader(train_ds, batch_size=batch_size,
                   shuffle=True,  collate_fn=_collate),
        DataLoader(eval_ds,  batch_size=eval_batch_size,
                   shuffle=False, collate_fn=_collate),
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Bradley-Terry pairwise ranking loss.

    """
    diff = chosen_rewards - rejected_rewards

    loss = -F.logsigmoid(diff).mean()
    
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: RewardModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one full pass over the training DataLoader.

    """
    model.train()

    loss_meter = AverageMeter("train_loss")
    chosen_meter = AverageMeter("chosen_reward")
    rejected_meter = AverageMeter("rejected_reward")

    for step, batch in enumerate(tqdm(loader, desc = f"Epoch:{epoch}")):
        batch = {k:v.to(device) for k,v in batch.items()}
        batch_size = batch["chosen_input_ids"].shape[0]

        r_chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        r_rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

        loss = ranking_loss(r_chosen, r_rejected)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_meter.update(loss.item(), n=batch_size)
        chosen_meter.update(r_chosen.mean().item(), n=batch_size)
        rejected_meter.update(r_rejected.mean().item(), n=batch_size)

        global_step = epoch * len(loader) + step
        log_metrics({
            "train_loss":    loss.item(),
            "chosen_reward": r_chosen.mean().item(),
            "rejected_reward": r_rejected.mean().item(),
        }, step=global_step)

    return {"train_loss": loss_meter.avg, "mean_chosen": chosen_meter.avg,
                  "mean_rejected": rejected_meter.avg,
                  "mean_reward_gap": chosen_meter.avg - rejected_meter.avg}


@torch.no_grad()
def evaluate(
    model: RewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the reward model on the eval DataLoader.

    """
    model.eval()

    loss_meter     = AverageMeter("eval_loss")
    chosen_meter   = AverageMeter("eval_chosen_reward")
    rejected_meter = AverageMeter("eval_rejected_reward")
    correct, total = 0, 0

    for eval_batch in loader:
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        batch_size = eval_batch["chosen_input_ids"].shape[0]
        
        r_chosen   = model(eval_batch["chosen_input_ids"],   eval_batch["chosen_attention_mask"])
        r_rejected = model(eval_batch["rejected_input_ids"], eval_batch["rejected_attention_mask"])
        
        loss = ranking_loss(r_chosen, r_rejected)
        loss_meter.update(loss.item(), n=batch_size)
        chosen_meter.update(r_chosen.mean().item(), n=batch_size)
        rejected_meter.update(r_rejected.mean().item(), n=batch_size)

        correct += (r_chosen > r_rejected).sum().item()
        total   += batch_size
    
    return {
    "eval_loss":       loss_meter.avg,
    "eval_accuracy":   correct / total if total > 0 else 0.0,
    "mean_chosen":     chosen_meter.avg,
    "mean_rejected":   rejected_meter.avg,
    "mean_reward_gap": chosen_meter.avg - rejected_meter.avg,
    }


def train(cfg: dict) -> None:
    """
    Full reward model training pipeline.

    """
    output_dir = cfg["reward_model"]["output_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["reward_model"]["sft_checkpoint"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_loader, eval_loader = build_dataloaders(cfg, tokenizer)

    model = RewardModel(
        model_name_or_path=cfg["reward_model"]["sft_checkpoint"],
        freeze_backbone=cfg["reward_model"]["freeze_backbone"]
    ).to(device)

    print(f"Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    param_groups = get_parameter_groups(model, cfg["reward_model"]["weight_decay"])
    optimizer = AdamW(param_groups, lr=cfg["reward_model"]["learning_rate"])


    total_steps = len(train_loader) * cfg["reward_model"]["num_train_epochs"]
    warmup_steps = int(total_steps * cfg["reward_model"]["warmup_ratio"])
    scheduler = get_scheduler("linear", optimizer, warmup_steps, total_steps)

    best_accuracy = 0.0
    for epoch in range(cfg["reward_model"]["num_train_epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        scheduler, device, epoch)
        eval_metrics  = evaluate(model, eval_loader, device)
        log_metrics({**train_metrics, **eval_metrics}, step=epoch)
        if eval_metrics["eval_accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["eval_accuracy"]
            save_checkpoint(model, output_dir,
                            epoch, eval_metrics, is_best=True)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print(f"Model saved to {output_dir}")




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
