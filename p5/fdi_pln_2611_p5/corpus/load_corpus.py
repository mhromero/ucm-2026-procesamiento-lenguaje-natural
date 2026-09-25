"""Corpus loading, concatenation, and optional Harry Potter truncation."""

from __future__ import annotations

from pathlib import Path

from fdi_pln_2611_p5.config import resolve_data_dir
from fdi_pln_2611_p5.paths import require_dir

# First-chapter markers for each Harry Potter book (lowercase; extra corpus is lowercased).
_HARRY_POTTER_BOOK_MARKERS = (
    "the worst birthday",
    "owl post",
    "the riddle house",
    "dudley demented",
    "the other minister",
    "the dark lord ascending",
)


def concatenar_archivos_txt(data_dir: str | Path) -> str:
    """Concatenate all ``.txt`` files in a directory into one string.

    Files are read in sorted filename order and joined with newline separators.

    Args:
        data_dir: Directory containing ``*.txt`` files.

    Returns:
        Combined corpus text.
    """
    data_path = require_dir(data_dir, label="corpus directory")
    textos = [
        archivo.read_text(encoding="utf-8")
        for archivo in sorted(data_path.glob("*.txt"))
    ]
    return "\n".join(textos)


def _harry_potter_book_starts(text: str) -> list[int]:
    """Find start indices of the seven books in a concatenated corpus.

    Args:
        text: Lowercased concatenated Harry Potter corpus text.

    Returns:
        Sorted list of book start character indices, including index ``0``.

    Raises:
        ValueError: If a book marker string cannot be found in ``text``.
    """
    starts = [0]
    search_from = 0
    for marker in _HARRY_POTTER_BOOK_MARKERS:
        idx = text.find(marker, search_from)
        if idx < 0:
            raise ValueError(
                f"Marcador de libro Harry Potter no encontrado: {marker!r}"
            )
        starts.append(idx)
        search_from = idx + 1
    return starts


def build_extra_train_corpus(corpus_cfg: dict) -> str:
    """Build optional Harry Potter training text from corpus configuration.

    When ``extra_max_books`` is ``0``, returns an empty string so training uses
    only the Alice corpus.

    Args:
        corpus_cfg: Corpus section of the application config.

    Returns:
        Lowercased extra training text, optionally truncated by book count.
    """
    max_books = corpus_cfg.get("extra_max_books")
    if max_books == 0:
        return ""
    extra = concatenar_archivos_txt(
        resolve_data_dir(corpus_cfg["extra_data_dir"])
    ).lower()
    if max_books is None:
        return extra
    return truncar_corpus_extra(extra, max_books)


def truncar_corpus_extra(text: str, max_books: int | None) -> str:
    """Truncate extra corpus text to the first ``max_books`` Harry Potter books.

    Args:
        text: Lowercased concatenated Harry Potter corpus text.
        max_books: Number of books to keep; ``None`` or ``>= 7`` keeps the full text.

    Returns:
        Truncated corpus text.

    Raises:
        ValueError: If ``max_books`` is less than 1.
    """
    if max_books is None or max_books >= 7:
        return text
    if max_books < 1:
        raise ValueError("max_books debe ser >= 1 (usa 0 en corpus para solo Alice)")
    starts = _harry_potter_book_starts(text)
    if max_books >= len(starts):
        return text
    return text[: starts[max_books]]
