from __future__ import annotations

import numpy as np
import ollama

MODEL = "nomic-embed-text"
BATCH = 32


def _embed_batch(textos: list[str]) -> list[np.ndarray | None]:
    response = ollama.embed(model=MODEL, input=textos)
    result = []
    for emb in response.embeddings:
        vec = np.array(emb, dtype=np.float32)
        norma = np.linalg.norm(vec)
        result.append(None if norma == 0 else vec / norma)
    return result


def calcular_embeddings(parrafos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ids = []
    embeddings = []
    n = len(parrafos)
    for i in range(0, n, BATCH):
        batch = parrafos[i : i + BATCH]
        textos = [p.get("text", p.get("texto", "")) for p in batch]
        vecs = _embed_batch(textos)
        for p, vec in zip(batch, vecs):
            if vec is not None:
                ids.append(p["index"])
                embeddings.append(vec)
        print(f"  {min(i + BATCH, n)}/{n} chunks procesados...")
    return np.array(embeddings, dtype=np.float32), np.array(ids, dtype=np.int32)


def guardar_embeddings(
    path_emb: str, path_ids: str, embeddings: np.ndarray, ids: np.ndarray
) -> None:
    np.save(path_emb, embeddings)
    np.save(path_ids, ids)


def cargar_embeddings(path_emb: str, path_ids: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(path_emb), np.load(path_ids)


def buscar_semantica(
    query: str, embeddings: np.ndarray, ids: np.ndarray, top_k: int = 20
) -> list[tuple[int, float]]:
    vecs = _embed_batch([query])
    vec = vecs[0]
    if vec is None:
        return []
    scores = embeddings @ vec
    indices = np.argsort(scores)[::-1][:top_k]
    return [(int(ids[i]), float(scores[i])) for i in indices]
