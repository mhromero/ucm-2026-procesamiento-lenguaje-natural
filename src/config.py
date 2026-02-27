"""
Configuración del bot: se carga desde config.json (mismo directorio que este módulo).
"""

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


_c = _load_config()


def _no_trailing_slash(url: str) -> str:
    return url.rstrip("/")


API_BASE = _no_trailing_slash(os.getenv("FDI_PLN__BUTLER_ADDRESS", _c["api_base"]))
OLLAMA_URL = os.getenv("FDI_PLN__OLLAMA_URL", _c["ollama_url"])
MODEL = os.getenv("FDI_PLN__MODEL", _c["model"])
GOLD_RESOURCE_NAME = _c["gold_resource_name"]
MAILBOX_ENDPOINT = API_BASE + _c["mailbox_endpoint"]
LETTER_ENDPOINT = API_BASE + _c["letter_endpoint"]
PACKAGE_ENDPOINT = API_BASE + _c["package_endpoint"]
ALIAS = os.getenv("FDI_PLN__ALIAS", _c.get("alias", ""))
SINGLE_PLAYER_MODE = os.getenv("FDI_PLN__MODO_MONOPUESTO", str(_c.get("modo_monopuesto", False))).lower() in ("true", "1", "yes")
LETTERS_BEFORE_REBROADCAST = int(_c.get("letters_before_rebroadcast", 5))
OFFERS_PER_PERSON = int(_c.get("offers_per_person", 2))
