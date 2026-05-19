from __future__ import annotations

from pathlib import Path


def concatenar_archivos_txt(data_dir: str | Path) -> str:
    data_path = Path(data_dir)
    textos = [
        archivo.read_text(encoding="utf-8")
        for archivo in sorted(data_path.glob("*.txt"))
    ]
    return "\n".join(textos)
