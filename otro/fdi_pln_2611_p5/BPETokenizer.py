import json
from collections import Counter
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from fdi_pln_2611_p5.labels import label_to_id, merge_subword_labels


class BPETokenizer:
    """Tokenizador Byte Pair Encoding entrenado desde cero sobre el corpus.

    Parte de caracteres individuales y fusiona iterativamente los pares más
    frecuentes hasta alcanzar vocab_size. Soporta codificación con etiquetas
    NER por carácter (encode_with_labels) para propagar etiquetas al nivel de
    subpalabra mediante merge_subword_labels.
    """

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

    def padding_token_id(self) -> int:
        """Id de relleno en ventanas NER (espacio BPE, no el carácter índice 0 genérico)."""
        space_ids = self.encode(" ")
        return space_ids[0] if space_ids else 0

    def encode(self, text: str) -> list[int]:
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def encode_with_labels(
        self, text: str, char_labels: list[int]
    ) -> tuple[list[int], list[int]]:
        if len(text) != len(char_labels):
            raise ValueError("text y char_labels deben tener la misma longitud.")

        tokens = [self.tok2id.get(c, 0) for c in text]
        labels = list(char_labels)
        for (a, b), new_id in self.merges:
            merged_tokens: list[int] = []
            merged_labels: list[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged_tokens.append(new_id)
                    merged_labels.append(merge_subword_labels(labels[i], labels[i + 1]))
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    merged_labels.append(labels[i])
                    i += 1
            tokens, labels = merged_tokens, merged_labels
        return tokens, labels

    def decode_tokens(self, tokens: list[int]) -> list[str]:
        return [self.id2tok.get(token_id, "?") for token_id in tokens]

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
