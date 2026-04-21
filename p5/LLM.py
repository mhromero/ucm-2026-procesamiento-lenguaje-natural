import torch
import torch.nn as nn

from p5.attention import Attention
from p5.BPETokenizer import BPETokenizer


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class LLM(nn.Module):
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

    def text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def build_windows(self, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
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
        _, seq_len = token_ids.shape
        if seq_len > self.window_size:
            raise ValueError(f"seq_len={seq_len} supera window_size={self.window_size}.")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x, causal=causal)

        x = self.final_ffn(x)
        logits = self.vocab_projection(x)
        return logits

    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        optimizer.zero_grad()

        logits = self.forward(x_batch, causal=True)
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), y_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
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