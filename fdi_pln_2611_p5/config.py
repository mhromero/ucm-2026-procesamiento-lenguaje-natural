"""Package configuration loading and path resolution utilities."""

from __future__ import annotations

import json
from pathlib import Path

from fdi_pln_2611_p5.paths import require_file

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.json"


def package_path(relative: str) -> Path:
    """Resolve a path relative to the installed package root (bundled assets).

    Args:
        relative: Path segment relative to the package directory.

    Returns:
        Absolute path under the package root.
    """
    return PACKAGE_DIR / relative


def path_in_cwd(relative: str | Path) -> Path:
    """Resolve a path relative to the current working directory.

    Used for CLI outputs (weights, reports, caches) so artifacts land where the
    user runs the command, not under ``site-packages``.

    Args:
        relative: Path segment (or absolute path) to resolve.

    Returns:
        Resolved absolute path.
    """
    p = Path(relative)
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def asset_path(relative: str) -> Path:
    """Resolve a bundled asset, preferring a copy in the working directory.

    Args:
        relative: Path relative to the project/package layout (e.g. ``data/corpus``).

    Returns:
        Existing path in the working directory, or the bundled package path.
    """
    local = path_in_cwd(relative)
    if local.exists():
        return local
    return package_path(relative)


def resolve_data_dir(relative: str) -> Path:
    """Resolve a corpus directory (cwd override, else bundled package data).

    Args:
        relative: Corpus directory from config (e.g. ``data/corpus``).

    Returns:
        Absolute path to an existing directory.

    Raises:
        FileNotFoundError: If the directory exists neither in cwd nor in the package.
    """
    local = path_in_cwd(relative)
    if local.is_dir():
        return local
    bundled = package_path(relative)
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        f"Corpus directory not found: {relative}\n  → {local}\n  → {bundled}"
    )


def load_config(path: Path | None = None) -> dict:
    """Load the JSON configuration file.

    Args:
        path: Optional config file path. Defaults to ``config.json`` in the package.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = require_file(path or DEFAULT_CONFIG_PATH, label="config file")
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
            rel = Path(*parts[idx + 1 :]).as_posix()
            candidates.append(asset_path(rel))
            candidates.append(package_path(rel))

    cache_rel = config["tokenizer"]["cache_path"]
    candidates.append(path_in_cwd(cache_rel))
    candidates.append(asset_path(cache_rel))
    candidates.append(package_path(cache_rel))

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
        f"BPE tokenizer not found (bpe_tokenizer.json).\n"
        f"  Tried: {tried}\n"
        f"  Hint: run ./cluster_pixel/fetch_results_pixel.sh"
    )
