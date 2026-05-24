"""Path existence checks with formatted English error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


class PathNotFoundError(FileNotFoundError):
    """Raised when a required file or directory path does not exist."""

    def __init__(
        self,
        *,
        kind: Literal["file", "directory"],
        label: str,
        path: Path | str,
    ) -> None:
        self.kind = kind
        self.label = label
        self.path = Path(path).expanduser().resolve()
        noun = "File" if kind == "file" else "Directory"
        super().__init__(f"{noun} not found: {label}\n  → {self.path}")


def require_file(path: Path | str, *, label: str) -> Path:
    """Resolve ``path`` and ensure it points to an existing regular file.

    Args:
        path: File path to validate.
        label: Short English description (e.g. ``"config file"``).

    Returns:
        Resolved absolute path.

    Raises:
        PathNotFoundError: If the path is missing or not a file.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PathNotFoundError(kind="file", label=label, path=resolved)
    return resolved


def require_dir(path: Path | str, *, label: str) -> Path:
    """Resolve ``path`` and ensure it points to an existing directory.

    Args:
        path: Directory path to validate.
        label: Short English description.

    Returns:
        Resolved absolute path.

    Raises:
        PathNotFoundError: If the path is missing or not a directory.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise PathNotFoundError(kind="directory", label=label, path=resolved)
    return resolved


def require_optional_file(path: Path | str | None, *, label: str) -> Path | None:
    """Like :func:`require_file`, but returns ``None`` when ``path`` is ``None``."""
    if path is None:
        return None
    return require_file(path, label=label)
