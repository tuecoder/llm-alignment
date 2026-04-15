"""
utils/trainer_utils.py
======================
Shared training infrastructure used by all training scripts.

Provides:
    AverageMeter         — running mean tracker for loss / metric logging
    get_parameter_groups — splits model params into weight-decay / no-decay groups
    save_checkpoint      — saves model + tokenizer + metadata to disk
    load_checkpoint      — restores model weights and returns saved metadata
    log_metrics          — logs to W&B (if initialised) and stdout
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------


class AverageMeter:
    """
    Tracks a running mean of a scalar metric (e.g. loss, accuracy).

    Usage:
        meter = AverageMeter("train_loss")
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=len(batch))
        print(meter.avg)
        meter.reset()
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._sum: float = 0.0
        self._count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add ``val`` averaged over ``n`` samples to the running total."""
        self._sum += val * n
        self._count += n

    def reset(self) -> None:
        """Reset the meter to zero."""
        self._sum = 0.0
        self._count = 0

    @property
    def avg(self) -> float:
        """Current running average. Returns 0.0 if no updates yet."""
        return self._sum / self._count if self._count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.6f}, count={self._count})"


# ---------------------------------------------------------------------------
# Optimizer parameter groups
# ---------------------------------------------------------------------------


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Split model parameters into two groups for AdamW:
        1. Parameters subject to weight decay  (nd-dim tensors, excluding biases)
        2. Parameters exempt from weight decay (biases, LayerNorm weights, embeddings)

    Applying weight decay to biases and normalisation layers hurts convergence
    and is not done in standard practice (e.g. GPT-2, BERT fine-tuning).

    Args:
        model: The model whose parameters to group.
        weight_decay: L2 regularisation coefficient for the decay group.

    Returns:
        List of two dicts compatible with ``torch.optim.AdamW(params=...)``.

    Example:
        >>> param_groups = get_parameter_groups(model, weight_decay=0.01)
        >>> optimizer = torch.optim.AdamW(param_groups, lr=2e-5)
    """
    no_decay_keywords = ("bias", "layer_norm", "layernorm", "ln_")

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1-D tensors (biases, LayerNorm scale/shift) and any name matching
        # the no-decay keywords are excluded from weight decay.
        if param.ndim == 1 or any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    output_dir: str,
    epoch: int,
    metrics: dict,
    tokenizer=None,
    is_best: bool = False,
    save_total_limit: Optional[int] = 2,
) -> str:
    """
    Save a model checkpoint plus metadata to ``output_dir/checkpoint-<epoch>/``.

    Handles ``save_total_limit`` by deleting the oldest checkpoint(s) when the
    limit is exceeded.

    Args:
        model: The model to save.
        output_dir: Root directory for all checkpoints.
        epoch: Current epoch index (used to name the checkpoint folder).
        metrics: Dict of scalar metrics to store in ``metadata.json``.
        tokenizer: Optional tokenizer; if provided it is also saved.
        is_best: If True, additionally copy to ``output_dir/best/``.
        save_total_limit: Maximum number of checkpoints to keep.  Oldest
            checkpoints (by epoch number) are deleted first.  ``None`` keeps all.

    Returns:
        Path to the saved checkpoint directory.
    """
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    # Save metadata (epoch + metrics)
    metadata = {"epoch": epoch, **metrics}
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    # Copy to best/ if this is the best checkpoint so far
    if is_best:
        best_dir = os.path.join(output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
        with open(os.path.join(best_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(best_dir)

    # Enforce save_total_limit
    if save_total_limit is not None:
        existing = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1]),
        )
        while len(existing) > save_total_limit:
            oldest = existing.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)

    return ckpt_dir


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
) -> dict:
    """
    Load model weights from a checkpoint directory (or .bin file).

    Args:
        model: The model to load weights into (in-place).
        checkpoint_path: Path to a checkpoint directory (containing
            ``pytorch_model.bin``) or directly to a ``.bin`` file.
        strict: Whether to enforce that all keys in the state dict match
            the model exactly.  Default True.

    Returns:
        Metadata dict loaded from ``metadata.json`` (empty dict if not found).
    """
    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        meta_path = os.path.join(checkpoint_path, "metadata.json")
    else:
        weights_path = checkpoint_path
        meta_path = ""

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)

    metadata: dict = {}
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return metadata


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_metrics(
    metrics: dict,
    step: int,
    prefix: str = "",
) -> None:
    """
    Log a dict of scalar metrics to W&B (if initialised) and print to stdout.

    Args:
        metrics: Dict mapping metric name → scalar value.
        step: Global training step (used as x-axis in W&B).
        prefix: Optional string prepended to each key, e.g. ``"train/"`` or
            ``"eval/"``.  A ``"/"`` separator is added automatically if the
            prefix does not already end with one.
    """
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}

    # W&B logging (silent if wandb is not initialised)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(prefixed, step=step)
    except ImportError:
        pass

    # MLflow logging (silent if mlflow is not installed or no active run)
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_metrics(prefixed, step=step)
    except ImportError:
        pass

    # Stdout logging
    parts = [f"step={step}"]
    parts += [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
              for k, v in prefixed.items()]
    print("  ".join(parts))
