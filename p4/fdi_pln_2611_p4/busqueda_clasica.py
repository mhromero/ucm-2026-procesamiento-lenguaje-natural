from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from typing import Any

import spacy

_nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])


def sin_tildes(texto: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def cargar_json(ruta: str) -> Any:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def buscar_frase(
    query: str, indice: dict[str, list]
) -> tuple[list[tuple[int, float, int]], set[str]]:
    doc = _nlp(query.strip())
    content_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
    if not content_tokens:
        return [], set()

    claves: list[str] = []
    for token in content_tokens:
        for clave in [
            token.lemma_.lower(),
            token.text.lower(),
            sin_tildes(token.text.lower()),
        ]:
            if clave in indice:
                claves.append(clave)
                break

    if not claves:
        return [], set()

    scores: dict[int, float] = defaultdict(float)
    hits: dict[int, set] = defaultdict(set)
    for clave in claves:
        for pid, tfidf in indice[clave]:
            scores[pid] += tfidf
            hits[pid].add(clave)

    resultados = [(pid, scores[pid], len(hits[pid])) for pid in scores]
    resultados.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return resultados, set(claves)


def destacar(texto: str, claves: set[str]) -> str:
    doc = _nlp(texto)
    partes = []
    ultimo = 0
    for token in doc:
        if token.is_alpha and (
            token.lemma_.lower() in claves
            or token.text.lower() in claves
            or sin_tildes(token.text.lower()) in claves
        ):
            partes.append(texto[ultimo : token.idx])
            partes.append(f"[b yellow]{token.text}[/]")
            ultimo = token.idx + len(token.text)
    partes.append(texto[ultimo:])
    return "".join(partes)
