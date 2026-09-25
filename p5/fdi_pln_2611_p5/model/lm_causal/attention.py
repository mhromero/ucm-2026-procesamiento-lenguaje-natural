"""Multi-head self-attention with optional causal masking."""

import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for the precomputed causal mask.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(self, x, causal=True):
        """Apply self-attention to the input sequence.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            causal: If True, apply a causal mask so positions cannot attend
                to future tokens.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        _, n_tokens, _ = x.shape
        q, k, v = self.qkv(x).tensor_split(3, dim=-1)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            scores = scores.masked_fill(
                self.mask[:n_tokens, :n_tokens] == float("-inf"), float("-inf")
            )
        scores = softmax(scores, dim=-1)
        scores = self.dropout(scores)
        z = scores @ v
        z = z.transpose(1, 2).flatten(-2)
        return self.out(z)

    def split_heads(self, x):
        """Reshape a tensor for multi-head attention.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
        """
        x = x.unflatten(-1, (self.n_heads, self.head_dim))
        return x.transpose(1, 2)
