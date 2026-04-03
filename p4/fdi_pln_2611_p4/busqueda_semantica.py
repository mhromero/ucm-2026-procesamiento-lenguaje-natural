from __future__ import annotations

import numpy as np
import spacy

_nlp_md: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp_md
    if _nlp_md is None:
        _nlp_md = spacy.load("es_core_news_md", disable=["parser", "ner"])
    return _nlp_md


def _vectorizar(texto: str, nlp: spacy.language.Language) -> np.ndarray | None:
    doc = nlp(texto)
    vectores = [t.vector for t in doc if t.is_alpha and not t.is_stop and t.has_vector]
    if not vectores:
        return None
    vec = np.mean(vectores, axis=0)
    norma = np.linalg.norm(vec)
    if norma == 0:
        return None
    return vec / norma


def calcular_embeddings(
    parrafos: list[dict], nlp: spacy.language.Language
) -> tuple[np.ndarray, np.ndarray]:
    ids = []
    embeddings = []
    for p in parrafos:
        texto = p.get("text", p.get("texto", ""))
        vec = _vectorizar(texto, nlp)
        if vec is not None:
            ids.append(p["index"])
            embeddings.append(vec)
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
    nlp = _get_nlp()
    vec = _vectorizar(query, nlp)
    if vec is None:
        return []
    scores = embeddings @ vec
    indices = np.argsort(scores)[::-1][:top_k]
    return [(int(ids[i]), float(scores[i])) for i in indices]
