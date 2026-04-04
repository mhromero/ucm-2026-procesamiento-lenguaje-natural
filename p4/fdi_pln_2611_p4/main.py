from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HTML_PATH = DATA_DIR / "2000-h.htm"
PARRAFOS_PATH = DATA_DIR / "parrafos_index.json"
INDICE_PATH = DATA_DIR / "vocabulario_index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_PATH = DATA_DIR / "embeddings_ids.npy"


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


def _regenerar_embeddings() -> None:
    from .busqueda_semantica import calcular_embeddings, guardar_embeddings

    print("Generando embeddings con ollama (puede tardar varios minutos)...")
    parrafos = json.loads(PARRAFOS_PATH.read_text(encoding="utf-8"))
    embeddings, ids = calcular_embeddings(parrafos)
    guardar_embeddings(str(EMBEDDINGS_PATH), str(EMBEDDINGS_IDS_PATH), embeddings, ids)
    print(f"Embeddings generados: {len(ids)} chunks.")


def main() -> None:
    if not PARRAFOS_PATH.exists() or not INDICE_PATH.exists():
        _regenerar_indices()

    if not EMBEDDINGS_PATH.exists() or not EMBEDDINGS_IDS_PATH.exists():
        _regenerar_embeddings()

    from .buscador_textual import Buscador

    app = Buscador(
        str(INDICE_PATH),
        str(PARRAFOS_PATH),
        str(EMBEDDINGS_PATH),
        str(EMBEDDINGS_IDS_PATH),
    )
    app.run()


if __name__ == "__main__":
    main()
