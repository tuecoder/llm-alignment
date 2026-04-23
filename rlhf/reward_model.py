"""
rlhf/reward_model.py
====================
Reward model architecture for Phase 2 of the RLHF pipeline.

"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RewardModel(nn.Module):
    """
    A reward model that wraps a GPT-2 backbone with a linear scalar head.

    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.config.n_embd

        self.scalar_head = nn.Linear(self.hidden_size, 1, bias = False)
        nn.init.normal_(self.scalar_head.weight, std = 1 / self.hidden_size)

        if freeze_backbone:
            for params in self.backbone.parameters():
                params.requires_grad = False


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a scalar reward for each sequence in the batch.

        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if attention_mask is None:
            # No padding — last token is always the final position
            hidden = outputs.last_hidden_state[:, -1, :]
        else:
            # Right-padded batch: find the last real token per sequence
            last_idx = (attention_mask.sum(dim=1) - 1).long()  # (batch,)
            batch_range = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden = outputs.last_hidden_state[batch_range, last_idx]  # (batch, hidden)

        rewards = self.scalar_head(hidden).squeeze(-1)         # (batch,)
        return rewards

    @property
    def num_trainable_parameters(self) -> int:
        """Return the count of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    rm = RewardModel()
    ids  = torch.randint(0, 50257, (2, 32))
    mask = torch.ones(2, 32)
    out  = rm(ids, mask)
    assert out.shape == (2,)  

