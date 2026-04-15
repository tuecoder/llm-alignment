"""
utils/scheduler.py
==================
Learning-rate schedulers with linear warmup.

Drop-in replacement for ``transformers.get_scheduler`` — no HuggingFace
dependency required.  All schedulers are built on ``torch.optim.lr_scheduler.LambdaLR``
so they compose freely with any PyTorch optimizer.

Supported types
---------------
"linear"  : Linear warmup then linear decay to 0 at ``num_training_steps``.
"cosine"  : Linear warmup then cosine decay to 0 (half-cosine, no restart).

Usage
-----
    scheduler = get_scheduler("cosine", optimizer,
                              num_warmup_steps=100,
                              num_training_steps=1000)
    # ... inside training loop ...
    scheduler.step()
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup from 0 → 1 over ``num_warmup_steps``, then linear decay
    from 1 → 0 over the remaining steps.

    Args:
        optimizer: The optimizer whose LR this scheduler will control.
        num_warmup_steps: Number of steps to ramp the LR from 0 to its peak.
        num_training_steps: Total number of training steps (warmup + decay).
        last_epoch: The index of the last epoch. Default -1 (fresh start).

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup from 0 → 1 over ``num_warmup_steps``, then half-cosine
    decay from 1 → 0 over the remaining steps.

    Args:
        optimizer: The optimizer whose LR this scheduler will control.
        num_warmup_steps: Warmup steps.
        num_training_steps: Total training steps.
        num_cycles: Number of cosine cycles. Default 0.5 (half-cosine, no
            restart). Use values > 0.5 for cosine-with-restarts.
        last_epoch: Default -1 (fresh start).

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Factory that dispatches to the correct scheduler by name.

    Args:
        name: Scheduler type. One of ``"linear"`` or ``"cosine"``.
        optimizer: The optimizer to wrap.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        A configured ``LambdaLR`` scheduler.

    Raises:
        ValueError: If ``name`` is not a supported scheduler type.

    Example:
        >>> # Compute warmup_steps from warmup_ratio
        >>> total_steps = len(train_loader) * num_epochs // grad_accum_steps
        >>> warmup_steps = int(total_steps * warmup_ratio)
        >>> scheduler = get_scheduler("cosine", optimizer, warmup_steps, total_steps)
    """
    schedulers = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
    }
    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler {name!r}. Choose from: {list(schedulers.keys())}"
        )
    return schedulers[name](optimizer, num_warmup_steps, num_training_steps)
