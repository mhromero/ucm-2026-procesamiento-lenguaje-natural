import json
from collections import Counter
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

class BPETokenizer:

    def __init__(self, text: str, vocab_size: int = 5000):
        chars = sorted(set(text))
        self.tok2id = {c: i for i, c in enumerate(chars)}
        self.id2tok = {i: c for i, c in enumerate(chars)}

        tokens = [self.tok2id[c] for c in text]
        self.merges = []
        n = len(self.tok2id)
        max_merges = vocab_size - n

        with Progress(
            TextColumn("[bold blue]Entrenando BPE"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} merges"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("bpe", total=max_merges)
            for _ in range(max_merges):
                pairs = Counter(zip(tokens, tokens[1:]))
                if not pairs:
                    break
                best = pairs.most_common(1)[0][0]

                new_id = len(self.tok2id)
                new_tok = self.id2tok[best[0]] + self.id2tok[best[1]]
                self.tok2id[new_tok] = new_id
                self.id2tok[new_id] = new_tok
                self.merges.append((best, new_id))

                tokens = self._apply_merge(tokens, best[0], best[1], new_id)
                progress.advance(task)

    @staticmethod
    def _apply_merge(tokens: list[int], a: int, b: int, new_id: int) -> list[int]:
        "Reemplaza los tokens a y b por el nuevo token new_id"
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text: str) -> list[int]:
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.id2tok.get(t, "?") for t in tokens])
        
    def save(self, path: str):
        save_path = Path(path)
        payload = {
            "tok2id": self.tok2id,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges],
        }
        save_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls.__new__(cls)
        tokenizer.tok2id = payload["tok2id"]
        tokenizer.id2tok = {idx: tok for tok, idx in tokenizer.tok2id.items()}
        tokenizer.merges = [((a, b), new_id) for a, b, new_id in payload["merges"]]
        return tokenizer

    def get_tokens(self) -> list[str]:
        return list(self.tok2id.keys())
    
