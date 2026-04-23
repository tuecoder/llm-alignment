"""
rlhf/train_sft.py
=================
Phase 1 of the RLHF pipeline: Supervised Fine-Tuning (SFT).

Goal: adapt the base GPT-2 model to the HH-RLHF conversation format by
performing standard causal language modelling on the *chosen* responses.

Usage:
    python rlhf/train_sft.py --config configs/rlhf_config.yaml

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from tqdm import tqdm

from data.data_utils import format_for_sft, load_hh_rlhf
from utils.scheduler import get_scheduler
from utils.trainer_utils import (
    AverageMeter,
    get_parameter_groups,
    log_metrics,
    save_checkpoint,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args() -> dict:
    """
    Parse CLI arguments and merge them with the YAML config file.

    """
    parser = argparse.ArgumentParser(description="Phase 1: SFT training")
    parser.add_argument("-c", "--config", default="configs/rlhf_config.yaml",
                        help="path to YAML config file")
    args, overrides = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)


    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_datasets(cfg: dict) -> tuple:
    """
    Load and format the HH-RLHF dataset for SFT.

    """
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

    train_ds = train_raw.map(format_for_sft, remove_columns=["chosen", "rejected"])
    eval_ds  = eval_raw.map(format_for_sft,  remove_columns=["chosen", "rejected"])

    return (train_ds, eval_ds)


def build_dataloaders(
    train_ds,
    eval_ds,
    tokenizer,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Tokenise the text datasets and wrap them in PyTorch DataLoaders.

    Each example has a single ``"text"`` field.  We tokenise it and produce:
        - ``input_ids``      : token IDs, padded / truncated to ``max_length``
        - ``attention_mask`` : 1 for real tokens, 0 for padding
        - ``labels``         : copy of ``input_ids`` with pad positions set to -100
                               (the causal LM cross-entropy loss ignores -100)
    """
    def tokenise(examples):
        enc = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        # Copy input_ids to labels; mask pad positions with -100
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in enc["input_ids"]
        ]
        enc["labels"] = labels
        return enc

    train_tok = train_ds.map(tokenise, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tokenise,  batched=True, remove_columns=["text"])

    train_tok.set_format("torch")
    eval_tok.set_format("torch")

    return (
        DataLoader(train_tok, batch_size=batch_size,
                   shuffle=True, collate_fn=default_data_collator),
        DataLoader(eval_tok,  batch_size=eval_batch_size,
                   shuffle=False, collate_fn=default_data_collator),
    )


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def build_tokenizer(cfg: dict):
    """
    Load and configure the tokeniser for GPT-2.

    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: dict) -> None:
    """
    Run the full SFT training pipeline with a raw PyTorch loop.
    """
    mlflow_cfg = cfg["sft"].get("mlflow", {})

    project_root = Path(__file__).resolve().parent.parent
    raw_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    if raw_uri.startswith("sqlite:///"):
        rel = raw_uri[len("sqlite:///"):]
        abs_db = (project_root / rel).resolve()
        tracking_uri = f"sqlite:///{abs_db}"
    else:
        tracking_uri = str((project_root / raw_uri).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "sft-training"))
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"Launch UI with      : mlflow ui --backend-store-uri \"{tracking_uri}\"")

    with mlflow.start_run(run_name=mlflow_cfg.get("run_name", "sft")):
        mlflow.log_params({
            "base_model":                  cfg["model"]["base_model"],
            "max_length":                  cfg["model"]["max_length"],
            "learning_rate":               cfg["sft"]["learning_rate"],
            "num_train_epochs":            cfg["sft"]["num_train_epochs"],
            "per_device_train_batch_size": cfg["sft"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": cfg["sft"]["gradient_accumulation_steps"],
            "effective_batch_size":        cfg["sft"]["per_device_train_batch_size"]
                                           * cfg["sft"]["gradient_accumulation_steps"],
            "warmup_ratio":                cfg["sft"]["warmup_ratio"],
            "weight_decay":                cfg["sft"]["weight_decay"],
            "fp16":                        cfg["sft"].get("fp16", False),
            "lr_scheduler_type":           cfg["sft"].get("lr_scheduler_type", "cosine"),
        })

        _train_inner(cfg)


def _train_inner(cfg: dict) -> None:
    fp16             = cfg["sft"].get("fp16", False)
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_accum_steps = cfg["sft"]["gradient_accumulation_steps"]
    logging_steps    = cfg["sft"]["logging_steps"]
    eval_steps       = cfg["sft"]["eval_steps"]
    output_dir       = cfg["sft"]["output_dir"]

    tokenizer = build_tokenizer(cfg)
    train_ds, eval_ds = build_datasets(cfg)
    train_loader, eval_loader = build_dataloaders(
        train_ds, eval_ds, tokenizer,
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["sft"]["per_device_train_batch_size"],
        eval_batch_size=cfg["sft"]["per_device_eval_batch_size"],
    )

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"]).to(device)

    param_groups = get_parameter_groups(model, cfg["sft"]["weight_decay"])
    optimizer = AdamW(param_groups, lr=cfg["sft"]["learning_rate"])

    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps  = steps_per_epoch * cfg["sft"]["num_train_epochs"]
    warmup_steps = int(total_steps * cfg["sft"]["warmup_ratio"])
    scheduler = get_scheduler("cosine", optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if fp16 else None

    best_eval_loss = float("inf")
    global_step = 0

    for epoch in range(cfg["sft"]["num_train_epochs"]):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Only the forward pass goes inside autocast; backward must be outside.
            with autocast(enabled=fp16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    log_metrics(
                        {
                            "train_loss": loss.item() * grad_accum_steps,
                            "lr":         scheduler.get_last_lr()[0],
                        },
                        global_step, prefix="train",
                    )

                if global_step % eval_steps == 0:
                    model.eval()
                    eval_meter = AverageMeter("eval_loss")
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                            with autocast(enabled=fp16):
                                eval_out = model(**eval_batch)
                            eval_meter.update(
                                eval_out.loss.item(),
                                n=eval_batch["input_ids"].shape[0],
                            )
                    is_best = eval_meter.avg < best_eval_loss
                    if is_best:
                        best_eval_loss = eval_meter.avg
                    save_checkpoint(
                        model, output_dir, global_step,
                        {"eval_loss": eval_meter.avg},
                        tokenizer=tokenizer, is_best=is_best,
                        save_total_limit=cfg["sft"]["save_total_limit"],
                    )
                    log_metrics({"eval_loss": eval_meter.avg}, global_step, prefix="eval")
                    model.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
