import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    """Auto-atención multi-cabezal con máscara causal opcional."""

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
        x = x.unflatten(-1, (self.n_heads, self.head_dim))
        return x.transpose(1, 2)
