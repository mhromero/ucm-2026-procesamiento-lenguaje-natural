from __future__ import annotations

from pathlib import Path

from fdi_pln_2611_p5.config import package_path

# Primer capítulo de cada libro (minúsculas; el corpus extra se pasa a .lower()).
_HARRY_POTTER_BOOK_MARKERS = (
    "the worst birthday",
    "owl post",
    "the riddle house",
    "dudley demented",
    "the other minister",
    "the dark lord ascending",
)


def concatenar_archivos_txt(data_dir: str | Path) -> str:
    data_path = Path(data_dir)
    textos = [
        archivo.read_text(encoding="utf-8")
        for archivo in sorted(data_path.glob("*.txt"))
    ]
    return "\n".join(textos)


def _harry_potter_book_starts(text: str) -> list[int]:
    """Índices de inicio de los 7 libros en el .txt concatenado."""
    starts = [0]
    search_from = 0
    for marker in _HARRY_POTTER_BOOK_MARKERS:
        idx = text.find(marker, search_from)
        if idx < 0:
            raise ValueError(f"Marcador de libro Harry Potter no encontrado: {marker!r}")
        starts.append(idx)
        search_from = idx + 1
    return starts


def build_extra_train_corpus(corpus_cfg: dict) -> str:
    """Texto HP para entrenamiento. ``extra_max_books=0`` → cadena vacía (solo Alice)."""
    max_books = corpus_cfg.get("extra_max_books")
    if max_books == 0:
        return ""
    extra = concatenar_archivos_txt(package_path(corpus_cfg["extra_data_dir"])).lower()
    if max_books is None:
        return extra
    return truncar_corpus_extra(extra, max_books)


def truncar_corpus_extra(text: str, max_books: int | None) -> str:
    """Recorta el corpus extra a los primeros ``max_books`` (p. ej. 4 ≈ mitad de 7)."""
    if max_books is None or max_books >= 7:
        return text
    if max_books < 1:
        raise ValueError("max_books debe ser >= 1 (usa 0 en corpus para solo Alice)")
    starts = _harry_potter_book_starts(text)
    if max_books >= len(starts):
        return text
    return text[: starts[max_books]]
