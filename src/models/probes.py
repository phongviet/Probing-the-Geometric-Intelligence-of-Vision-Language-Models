"""
src/models/probes.py
--------------------
Probe architectures for Geometric Intelligence Quotient (GIQ) tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Standard linear probe (logistic regression).
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.linear(x)


class MLPProbe(nn.Module):
    """
    Multi-Layer Perceptron probe (Linear -> ReLU -> Linear).
    Tests for non-linear separability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseProbe(nn.Module):
    """
    Dense probe for surface normal estimation.
    Operates on patch tokens using a 1x1 convolution (linear per-patch).
    """

    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        # 1x1 Conv is equivalent to a linear layer applied to each spatial location independently.
        # We implement it as a Linear layer for flexibility with flat or spatial inputs.
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch tokens [B, N_patches, D] or [B, H, W, D]
        Returns:
            [B, N_patches, 3] or [B, H, W, 3]
        """
        return self.proj(x)
