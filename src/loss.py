"""Loss functions for Neural Operator training.

Implements the Relative Lp loss (normalized L2 norm) as introduced in
Li et al. (ICLR 2021).
"""

import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """Relative L2 Loss (Relative Lp norm with p=2).

    Computes:
        L_rel(y_pred, y_true) = ||y_pred - y_true||_2 / (||y_true||_2 + eps)
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        """
        Args:
            eps: Small constant to prevent division by zero.
            reduction: 'mean' to average across batch, 'sum' to sum, or 'none'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]

        # Flatten spatial dimensions
        diff = (y_pred - y_true).reshape(batch_size, -1)
        true = y_true.reshape(batch_size, -1)

        diff_norm = torch.norm(diff, p=2, dim=1)
        true_norm = torch.norm(true, p=2, dim=1) + self.eps

        rel_error = diff_norm / true_norm

        if self.reduction == "mean":
            return torch.mean(rel_error)
        elif self.reduction == "sum":
            return torch.sum(rel_error)
        elif self.reduction == "none":
            return rel_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class LpLoss(nn.Module):
    """General Lp loss supporting both relative and absolute norms."""

    def __init__(self, p: int = 2, relative: bool = True, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.p = p
        self.relative = relative
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        diff = (y_pred - y_true).reshape(batch_size, -1)

        diff_norm = torch.norm(diff, p=self.p, dim=1)

        if self.relative:
            true = y_true.reshape(batch_size, -1)
            true_norm = torch.norm(true, p=self.p, dim=1) + self.eps
            loss = diff_norm / true_norm
        else:
            loss = diff_norm

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss
