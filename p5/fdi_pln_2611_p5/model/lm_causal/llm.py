"""Decoder-only causal language model built on a Transformer backbone."""

import torch
import torch.nn as nn

from fdi_pln_2611_p5.model.lm_causal.attention import Attention
from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm attention and feed-forward layers.

    Architecture: layer norm → attention → residual, then layer norm → FFN →
    residual.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for the attention mask.
        dropout: Dropout probability.
    """

    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(
            d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Run the block on a sequence of hidden states.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            causal: Whether attention uses a causal mask.

        Returns:
            Updated hidden states with the same shape as ``x``.
        """
        x = x + self.attention(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class LLM(nn.Module):
    """Causal language model based on a decoder-only Transformer.

    Architecture: token embedding + positional embedding → stacked
    ``TransformerBlock`` layers with causal attention → final FFN → vocabulary
    projection. For NER, ``encode_tokens(causal=False)`` returns contextual
    representations without the vocabulary head.

    Args:
        tokenizer: BPE tokenizer defining the vocabulary.
        d_model: Hidden dimension size.
        n_blocks: Number of Transformer blocks.
        n_heads: Number of attention heads per block.
        window_size: Maximum context length and sliding-window size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        d_model: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        window_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.vocab_size = len(tokenizer.tok2id)

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(window_size, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=window_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.final_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vocab_projection = nn.Linear(d_model, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.d_model = d_model

    def encode_tokens(
        self, token_ids: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        """Return contextual token representations without the vocabulary head.

        Args:
            token_ids: Tensor of shape ``(batch, seq_len)``.
            causal: Whether attention uses a causal mask.

        Returns:
            Hidden states of shape ``(batch, seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len`` exceeds ``window_size``.
        """
        _, seq_len = token_ids.shape
        if seq_len > self.window_size:
            raise ValueError(
                f"seq_len={seq_len} supera window_size={self.window_size}."
            )

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, causal=causal)
        return self.final_ffn(x)

    def text_to_tokens(self, text: str) -> list[int]:
        """Tokenize text with the model's BPE tokenizer.

        Args:
            text: Input string.

        Returns:
            List of token ids.
        """
        return self.tokenizer.encode(text)

    def build_windows(self, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build sliding-window training examples from a token sequence.

        Args:
            token_ids: Full tokenized text.

        Returns:
            Tuple ``(x_windows, y_targets)`` of input and target tensors.

        Raises:
            ValueError: If the tokenized text is not longer than ``window_size``.
        """
        if len(token_ids) <= self.window_size:
            raise ValueError("El texto tokenizado debe ser mayor que window_size.")

        x_windows = []
        y_targets = []
        for i in range(len(token_ids) - self.window_size):
            x_windows.append(token_ids[i : i + self.window_size])
            y_targets.append(token_ids[i + 1 : i + self.window_size + 1])

        x = torch.tensor(x_windows, dtype=torch.long)
        y = torch.tensor(y_targets, dtype=torch.long)
        return x, y

    def forward(self, token_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Project contextual representations to vocabulary logits.

        Args:
            token_ids: Tensor of shape ``(batch, seq_len)``.
            causal: Whether attention uses a causal mask.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.encode_tokens(token_ids, causal=causal)
        return self.vocab_projection(x)

    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one causal LM training step and return the loss value.

        Args:
            x_batch: Input token windows.
            y_batch: Target token windows (shifted by one position).
            optimizer: Optimizer used for the parameter update.

        Returns:
            Scalar loss value for the batch.
        """
        self.train()
        optimizer.zero_grad()

        logits = self.forward(x_batch, causal=True)
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), y_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def generate(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0
    ) -> str:
        """Autoregressively generate text from a prompt.

        Args:
            prompt: Seed text; converted to lowercase before tokenization.
            max_new_tokens: Maximum number of tokens to append.
            temperature: Sampling temperature; must be positive.

        Returns:
            Prompt followed by newly generated text.

        Raises:
            ValueError: If ``temperature`` is not positive or the prompt is empty.
        """
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature debe ser mayor que 0.")
        model_device = next(self.parameters()).device
        prompt = prompt.lower()

        generated = self.text_to_tokens(prompt)
        if not generated:
            raise ValueError("El prompt no puede ser vacio.")
        new_token_ids = []

        for _ in range(max_new_tokens):
            context = generated[-self.window_size :]
            x = torch.tensor([context], dtype=torch.long, device=model_device)
            logits = self.forward(x, causal=True)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)
            new_token_ids.append(next_token_id)

        return prompt + self.tokenizer.decode(new_token_ids)
