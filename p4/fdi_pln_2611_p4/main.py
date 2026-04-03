from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HTML_PATH = DATA_DIR / "2000-h.htm"
PARRAFOS_PATH = DATA_DIR / "parrafos_index.json"
INDICE_PATH = DATA_DIR / "vocabulario_index.json"


def _regenerar_indices() -> None:
    from .indexar_parrafos import index_html

    print("Generando índices (esto puede tardar unos segundos)...")
    paragraphs, vocabulary = index_html(HTML_PATH)
    PARRAFOS_PATH.write_text(
        json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    INDICE_PATH.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Índices generados: {len(paragraphs)} párrafos, {len(vocabulary)} términos.")


def main() -> None:
    if not PARRAFOS_PATH.exists() or not INDICE_PATH.exists():
        _regenerar_indices()

    from .buscador_textual import Buscador

    app = Buscador(str(INDICE_PATH), str(PARRAFOS_PATH))
    app.run()


if __name__ == "__main__":
    main()
