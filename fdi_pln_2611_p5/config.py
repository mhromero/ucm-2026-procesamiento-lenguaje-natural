"""Package configuration loading and path resolution utilities."""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.json"


def package_path(relative: str) -> Path:
    """Resolve a path relative to the package root.

    Args:
        relative: Path segment relative to the package directory.

    Returns:
        Absolute path under the package root.
    """
    return PACKAGE_DIR / relative


def load_config(path: Path | None = None) -> dict:
    """Load the JSON configuration file.

    Args:
        path: Optional config file path. Defaults to ``config.json`` in the package.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_tokenizer_path(
    checkpoint_payload: dict | None,
    config: dict,
    *,
    weights_path: Path | None = None,
    weights_dir: Path | None = None,
) -> Path:
    """Resolve the BPE tokenizer file path.

      Search order: checkpoint metadata (if present), package default, then paths
    adjacent to the weights file or results directory.

      Args:
          checkpoint_payload: Optional checkpoint metadata containing
              ``tokenizer_path``.
          config: Application configuration dictionary.
          weights_path: Optional path to model weights; its parent is used as a
              search root.
          weights_dir: Optional directory to search for bundled tokenizer files.

      Returns:
          Path to an existing ``bpe_tokenizer.json`` file.

      Raises:
          FileNotFoundError: If no candidate tokenizer file exists.
    """
    candidates: list[Path] = []
    if checkpoint_payload and checkpoint_payload.get("tokenizer_path"):
        stored = Path(checkpoint_payload["tokenizer_path"])
        candidates.append(stored)
        # Absolute cluster path: .../fdi_pln_2611_p5/data/bpe_tokenizer.json
        parts = stored.parts
        if "fdi_pln_2611_p5" in parts:
            idx = parts.index("fdi_pln_2611_p5")
            candidates.append(package_path(Path(*parts[idx + 1 :]).as_posix()))

    candidates.append(package_path(config["tokenizer"]["cache_path"]))

    base = weights_dir
    if base is None and weights_path is not None:
        base = Path(weights_path).resolve().parent
    if base is not None:
        candidates.extend(
            [
                base / "data/bpe_tokenizer.json",
                base / "fdi_pln_2611_p5/data/bpe_tokenizer.json",
                base / "resultados_pixel/data/bpe_tokenizer.json",
            ]
        )

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No se encontró bpe_tokenizer.json. Probado: {tried}. "
        "Descarga con: ./cluster_pixel/fetch_results_pixel.sh"
    )
