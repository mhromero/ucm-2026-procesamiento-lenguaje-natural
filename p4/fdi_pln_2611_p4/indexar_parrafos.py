"""Indexa párrafos HTML y crea un vocabulario invertido con puntuaciones TF-IDF.

Formato del vocabulario: { term: [[párrafo_id, tfidf], ...] } ordenado por TF-IDF desc.
  - TF(t, d)  = ocurrencias(t, d) / tokens_no_stop(d)
  - IDF(t)    = log(N / df(t))   donde N = nº párrafos, df = nº párrafos con t
  - TF-IDF    = TF * IDF
"""

from __future__ import annotations

import argparse
import json
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

nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])


def text_from_tag(tag: Tag) -> str:
    return " ".join(tag.stripped_strings)


def update_heading_path(path: Dict[int, str], level: int, text: str) -> None:
    path[level] = text
    for lower in range(level + 1, 7):
        path.pop(lower, None)


def heading_list(path: Dict[int, str]) -> List[str]:
    return [path[level] for level in sorted(path)]


def index_html(input_html: Path) -> tuple[list[dict], dict[str, list]]:
    html = input_html.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    current_headings: Dict[int, str] = {}
    paragraphs: List[dict] = []

    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    par_lengths: Dict[int, int] = {}

    body = soup.body or soup
    current_paragraph_index = 0

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

        paragraphs.append(
            {
                "index": current_paragraph_index,
                "headings": heading_list(current_headings),
                "text": text,
            }
        )

        doc = nlp(text)
        valid_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
        par_lengths[current_paragraph_index] = max(len(valid_tokens), 1)

        for token in valid_tokens:
            counts[token.lemma_.lower()][current_paragraph_index] += 1
            counts[token.text.lower()][current_paragraph_index] += 1

        current_paragraph_index += 1

    N = len(paragraphs)
    vocabulary: Dict[str, list] = {}
    for term, doc_counts in sorted(counts.items()):
        df = len(doc_counts)
        idf = log(N / df)
        entries = [
            [pid, round(count / par_lengths[pid] * idf, 6)]
            for pid, count in doc_counts.items()
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        vocabulary[term] = entries

    return paragraphs, vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera índice de párrafos y vocabulario invertido con TF-IDF."
    )
    parser.add_argument("input_html", type=Path)
    parser.add_argument(
        "--out-parrafos", type=Path, default=Path("parrafos_index.json")
    )
    parser.add_argument(
        "--out-vocabulario", type=Path, default=Path("vocabulario_index.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paragraphs, vocabulary = index_html(args.input_html)

    args.out_parrafos.write_text(
        json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.out_vocabulario.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Párrafos indexados: {len(paragraphs)}\n"
        f"Vocabulario único: {len(vocabulary)}\n"
        f"Archivo párrafos: {args.out_parrafos}\n"
        f"Archivo vocabulario: {args.out_vocabulario}"
    )


if __name__ == "__main__":
    main()
