import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightMLP(nn.Module):
    """Small MLP that maps learnable bin embeddings to scalar log-weights."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_activation: str = "softplus",
        min_weight: float = 1e-6,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_activation = out_activation
        self.min_weight = float(min_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        if self.out_activation == "softplus":
            out = F.softplus(out)
        elif self.out_activation == "exp":
            out = torch.exp(out)
        elif self.out_activation in ("none", None):
            pass
        else:
            raise ValueError(f"Unknown out_activation: {self.out_activation}")
        if self.min_weight > 0:
            out = out + self.min_weight
        return out
