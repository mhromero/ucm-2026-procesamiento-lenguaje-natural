"""Indexa un HTML en chunks de frases y crea un vocabulario invertido con TF-IDF.

Estrategia de chunking: ventana deslizante de frases.
  - Se extraen todas las frases del texto manteniendo su contexto de heading.
  - Se agrupan en ventanas de WINDOW frases con un paso de STEP (solapamiento de
    WINDOW-STEP frases entre chunks consecutivos).

Formato del vocabulario: { term: [[chunk_id, tfidf], ...] } ordenado por TF-IDF desc.
  - TF(t, d)  = ocurrencias(t, d) / tokens_no_stop(d)
  - IDF(t)    = log(N / df(t))   donde N = nº chunks, df = nº chunks con t
  - TF-IDF    = TF * IDF
"""

from __future__ import annotations

import re
from collections import defaultdict
from math import log
from pathlib import Path
from typing import Dict, List

import spacy
from bs4 import BeautifulSoup, NavigableString, Tag

CHAPTER_RE = re.compile(
    r"^\s*(cap[ií]tulo|primera parte|segunda parte|tercera parte|cuarta parte)\b",
    re.IGNORECASE,
)

WINDOW = 5
STEP = 3

# Cargamos con senter habilitado para segmentar frases correctamente.
nlp = spacy.load("es_core_news_sm", disable=["ner"])


def text_from_tag(tag: Tag) -> str:
    return " ".join(tag.stripped_strings)


def update_heading_path(path: Dict[int, str], level: int, text: str) -> None:
    path[level] = text
    for lower in range(level + 1, 7):
        path.pop(lower, None)


def heading_list(path: Dict[int, str]) -> List[str]:
    return [path[level] for level in sorted(path)]


def _extract_sentences(
    input_html: Path,
) -> list[tuple[str, list[str]]]:
    """Devuelve lista de (frase, headings) para todo el documento."""
    html = input_html.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    current_headings: Dict[int, str] = {}
    sentences: list[tuple[str, list[str]]] = []

    body = soup.body or soup
    for node in body.descendants:
        if isinstance(node, NavigableString) or not isinstance(node, Tag):
            continue

        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = text_from_tag(node)
            if title:
                update_heading_path(current_headings, int(name[1]), title)
            continue

        if name != "p":
            continue

        text = text_from_tag(node)
        if not text:
            continue

        if CHAPTER_RE.match(text):
            update_heading_path(current_headings, 6, text)

        headings = heading_list(current_headings)
        doc = nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append((sent_text, headings))

    return sentences


def _build_chunks(
    sentences: list[tuple[str, list[str]]],
    window: int = WINDOW,
    step: int = STEP,
) -> list[dict]:
    """Agrupa frases en ventanas deslizantes."""
    chunks = []
    n = len(sentences)
    i = 0
    while i < n:
        window_sents = sentences[i : i + window]
        text = " ".join(s for s, _ in window_sents)
        headings = window_sents[0][1]
        chunks.append(
            {
                "index": len(chunks),
                "headings": headings,
                "text": text,
                "sent_start": i,
                "sent_end": i + len(window_sents) - 1,
                "n_sents_total": n,
            }
        )
        i += step
    return chunks


def index_html(input_html: Path) -> tuple[list[dict], dict[str, list]]:
    sentences = _extract_sentences(input_html)
    chunks = _build_chunks(sentences)

    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    chunk_lengths: Dict[int, int] = {}

    for chunk in chunks:
        cid = chunk["index"]
        doc = nlp(chunk["text"])
        valid_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
        chunk_lengths[cid] = max(len(valid_tokens), 1)
        for token in valid_tokens:
            counts[token.lemma_.lower()][cid] += 1
            counts[token.text.lower()][cid] += 1

    N = len(chunks)
    vocabulary: Dict[str, list] = {}
    for term, doc_counts in sorted(counts.items()):
        df = len(doc_counts)
        idf = log(N / df)
        entries = [
            [cid, round(count / chunk_lengths[cid] * idf, 6)]
            for cid, count in doc_counts.items()
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        vocabulary[term] = entries

    return chunks, vocabulary
