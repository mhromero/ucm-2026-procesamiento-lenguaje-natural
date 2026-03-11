#!/usr/bin/env python3
"""Indexa párrafos HTML y crea un vocabulario invertido por índice de párrafo.

Uso:
    python3 indexar_parrafos.py input.html
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from bs4 import BeautifulSoup, NavigableString, Tag


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)
CHAPTER_RE = re.compile(
    r"^\s*(cap[ií]tulo|primera parte|segunda parte|tercera parte|cuarta parte)\b",
    re.IGNORECASE,
)


def normalize_word(word: str) -> str:
    """Normaliza tokens para el índice de vocabulario."""
    token = unicodedata.normalize("NFKC", word).lower()
    return token.strip("_")


def text_from_tag(tag: Tag) -> str:
    """Extrae texto legible de una etiqueta HTML."""
    return " ".join(tag.stripped_strings)


def update_heading_path(path: Dict[int, str], level: int, text: str) -> None:
    """Actualiza la jerarquía de títulos (h1..h6)."""
    path[level] = text
    for lower in range(level + 1, 7):
        path.pop(lower, None)


def heading_list(path: Dict[int, str]) -> List[str]:
    return [path[level] for level in sorted(path)]


def index_html(input_html: Path) -> tuple[list[dict], dict[str, list[int]]]:
    html = input_html.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    current_headings: Dict[int, str] = {}
    paragraphs: List[dict] = []
    vocabulary: Dict[str, Set[int]] = defaultdict(set)

    body = soup.body or soup
    current_paragraph_index = 0

    for node in body.descendants:
        if isinstance(node, NavigableString) or not isinstance(node, Tag):
            continue

        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = text_from_tag(node)
            if title:
                level = int(name[1])
                update_heading_path(current_headings, level, title)
            continue

        if name != "p":
            continue

        text = text_from_tag(node)
        if not text:
            continue

        if CHAPTER_RE.match(text):
            update_heading_path(current_headings, 6, text)

        entry = {
            "index": current_paragraph_index,
            "headings": heading_list(current_headings),
            "text": text,
        }
        paragraphs.append(entry)

        for raw_word in WORD_RE.findall(text):
            word = normalize_word(raw_word)
            if word:
                vocabulary[word].add(current_paragraph_index)

        current_paragraph_index += 1

    vocabulary_json = {word: sorted(indices) for word, indices in sorted(vocabulary.items())}
    return paragraphs, vocabulary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dos JSON: índice de párrafos y vocabulario invertido "
            "(palabra -> índices de párrafo)."
        )
    )
    parser.add_argument("input_html", type=Path, help="Ruta del HTML de entrada.")
    parser.add_argument(
        "--out-parrafos",
        type=Path,
        default=Path("parrafos_index.json"),
        help="Salida JSON para el índice de párrafos.",
    )
    parser.add_argument(
        "--out-vocabulario",
        type=Path,
        default=Path("vocabulario_index.json"),
        help="Salida JSON para el índice de vocabulario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paragraphs, vocabulary = index_html(args.input_html)

    args.out_parrafos.write_text(
        json.dumps(paragraphs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out_vocabulario.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Párrafos indexados: {len(paragraphs)}\n"
        f"Vocabulario único: {len(vocabulary)}\n"
        f"Archivo párrafos: {args.out_parrafos}\n"
        f"Archivo vocabulario: {args.out_vocabulario}"
    )


if __name__ == "__main__":
    main()
