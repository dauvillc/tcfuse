"""Position-wise feed-forward block used inside each transformer block."""

from __future__ import annotations

from torch import Tensor, nn


class FeedForward(nn.Module):
    """Two-layer MLP with GELU, applied independently at every token position.

    Args:
        embed_dim: Token embedding dimension D (input and output size).
        mlp_ratio: Hidden-layer width as a multiple of ``embed_dim``.
        dropout: Dropout probability applied after each linear layer.
    """

    def __init__(self, *, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = round(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to every token, shape (B, L, D) -> (B, L, D)."""
        return self.net(x)
