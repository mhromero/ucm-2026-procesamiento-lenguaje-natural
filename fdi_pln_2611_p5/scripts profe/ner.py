"""Modelo NER, dataset y alineamiento palabra -> BPE.

El flujo de datos:

  palabras anotadas (word-level, BIO)
        |
        |  align_to_bpe()       <- asigna B-/I- a cada sub-token
        v
  sub-tokens + etiquetas BIO
        |
        |  NERDataset + collate_ner
        v
  batches (ids, labels) listos para cross_entropy
        |
        v
  NERLLM (Transformer + cabeza lineal por token)
"""

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

from p5.transformer import Transformer

# Etiquetas NER en esquema BIO
LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def align_to_bpe(words, word_labels, tokenizer):
    """Alinea etiquetas BIO de palabras a los sub-tokens de BPE.

    Como el tokenizador BPE puede partir una palabra en varios sub-tokens,
    hay que decidir que etiqueta dar a cada trozo. Regla: la B- se queda
    en el primer sub-token y los siguientes son I-; asi la cabeza marca
    la frontera de la entidad a nivel palabra.

      palabra 'alice' con etiqueta B-PER, BPE la parte en ['al', 'ice']
         -> al: B-PER, ice: I-PER
      palabra 'cheshire' con I-PER, BPE en ['ch', 'es', 'h', 'ire']
         -> todos I-PER
      palabra 'wonderland' con B-LOC, BPE en ['won', 'der', 'land']
         -> won: B-LOC, der: I-LOC, land: I-LOC
      palabras O -> todos sus sub-tokens O
      espacios entre palabras -> O

    Devuelve (token_ids, token_labels) con etiquetas como strings.
    """
    token_ids = []
    token_labels = []
    space_ids = tokenizer.encode(" ")
    for i, (word, label) in enumerate(zip(words, word_labels)):
        if i > 0:
            token_ids.extend(space_ids)
            token_labels.extend(["O"] * len(space_ids))
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        if label.startswith("B-"):
            inside = "I-" + label[2:]
            token_labels.append(label)
            token_labels.extend([inside] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))
    return token_ids, token_labels


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra -> sub-tokens BPE para una frase.

    Util para ver como el tokenizador parte cada palabra y donde aterriza
    cada etiqueta BIO: la B- se queda en el primer sub-token, el resto son I-.
    """
    print(f"  frase: {' '.join(words)}")
    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        if label.startswith("B-"):
            inside = "I-" + label[2:]
            labs = [label] + [inside] * (len(ids) - 1)
        else:
            labs = [label] * len(ids)
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER.

    Extiende Transformer añadiendo una cabeza lineal que asigna una etiqueta
    BIO a cada token. Usa atención bidireccional (causal=False): para etiquetar
    un token podemos mirar el contexto a derecha e izquierda.

    Los pesos del backbone se deben inicializar desde un CausalLLM pre-entrenado
    con load_state_dict(strict=False), que ignora las diferencias en las cabezas
    (lm_head vs ner_head) y transfiere solo el backbone compartido.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        num_labels,
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # El transformer ya tiene una representación suficientemente rica,
        # no tenemos más que proyectarla al espacio de etiquetas
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None):
        hidden = super().forward(input_ids, causal=False)
        logits = self.ner_head(hidden)
        loss = None
        if labels is not None:
            # cross_entropy espera logits 2D: para cada elemento, una
            # probabilidad por etiqueta.
            # Aplanamos batch y secuencia y tratamos cada token como una muestra
            # independiente:
            #   logits  (n_batches, n_tokens, num_labels) -> (n_batches*n_tokens, num_labels)
            #   labels  (n_batches, n_tokens)             -> (n_batches*n_tokens,)
            # Las posiciones de padding llevan -100 e ignore_index las descarta.
            flat_logits = logits.flatten(0, 1)
            flat_labels = labels.flatten()
            loss = cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def predict_entities(self, words, tokenizer):
        """Predice etiquetas BIO sobre una lista de **palabras**.

        Codifica la frase con `align_to_bpe` (etiquetas ficticias O), corre el
        modelo y agrupa sub-tokens B-X / I-X consecutivos en entidades.

        Devuelve las entidades nombradas ya compuestas [(texto, tipo), ...].
        """
        self.eval()
        ids, _ = align_to_bpe(words, ["O"] * len(words), tokenizer)
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor([ids], device=device))
        pred_labels = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        entities = []
        i = 0
        while i < len(ids):
            if pred_labels[i] != "O":
                kind = pred_labels[i].split("-")[1]
                j = i + 1
                while j < len(ids) and pred_labels[j] == f"I-{kind}":
                    j += 1
                text = tokenizer.decode(ids[i:j]).strip()
                if text:
                    entities.append((text, kind))
                i = j
            else:
                i += 1
        return entities


class NERDataset(Dataset):
    """Dataset de NER: aplica `align_to_bpe` a cada frase y convierte a tensores.

    `ner_data` es una lista de pares (words, labels), donde words es la lista
    de palabras de una frase y labels las etiquetas BIO alineadas. Es el
    formato que produce `load_ner_data` al leer el TSV en formato CoNLL.
    """

    def __init__(self, ner_data, tokenizer, max_len=128):
        self.samples = []
        for words, labels in ner_data:
            ids, labs = align_to_bpe(words, labels, tokenizer)
            ids = ids[:max_len]
            labs = labs[:max_len]
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor([LABEL2ID[l] for l in labs], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo maximo del batch. Las posiciones de padding usan -100
    en las etiquetas para que cross_entropy las ignore (no son tokens reales)."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
    return padded_x, padded_y
