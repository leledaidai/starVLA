from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A minimal Linear + LoRA wrapper that preserves `weight` / `bias` keys."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(self.weight.new_empty((self.rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.rank)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        lora = F.linear(self.dropout(inputs), self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scaling
        return base + lora


def freeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def replace_modules_by_suffix(
    module: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Replace Linear children whose final path segment matches a configured suffix."""

    suffixes = tuple(str(name) for name in target_suffixes)
    replaced = 0

    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in suffixes:
            setattr(
                module,
                child_name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            replaced += 1
            continue
        replaced += replace_modules_by_suffix(
            child,
            target_suffixes=suffixes,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    return replaced

