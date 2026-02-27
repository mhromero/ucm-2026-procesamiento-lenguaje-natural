"""
Configuración del bot y CLI con Click.
Prioridad: 1) variables de entorno, 2) argumentos Click, 3) config.json.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable

import click

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_c: dict = {}
_cli_overrides: dict = {}


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def init_cli_overrides(overrides: dict) -> None:
    """Registra valores procedentes de argumentos Click. Debe llamarse antes de acceder a la config."""
    _cli_overrides.update({k: v for k, v in overrides.items() if v is not None})


def _resolve(
    key: str,
    env_var: str,
    converter: Callable[[Any], Any] = str,
) -> Any:
    """Resuelve valor: 1) env, 2) cli, 3) config.json."""
    global _c
    if not _c:
        _c = _load_config()

    env_val = os.getenv(env_var)
    if env_val is not None:
        return converter(env_val)
    if key in _cli_overrides:
        return converter(_cli_overrides[key])
    if key not in _c:
        raise KeyError(f"config.json debe incluir la clave '{key}'")
    return converter(_c[key])


def _no_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def _resolve_bool(val: Any) -> bool:
    return str(val).lower() in ("true", "1", "yes")


# Nombres exportados (lazy via __getattr__)
_EXPORTS = frozenset({
    "API_BASE", "OLLAMA_URL", "MODEL", "GOLD_RESOURCE_NAME",
    "MAILBOX_ENDPOINT", "LETTER_ENDPOINT", "PACKAGE_ENDPOINT",
    "ALIAS", "SINGLE_PLAYER_MODE", "LETTERS_BEFORE_REBROADCAST", "OFFERS_PER_PERSON",
})


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "API_BASE":
        return _no_trailing_slash(_resolve("api_base", "FDI_PLN__BUTLER_ADDRESS"))

    if name == "OLLAMA_URL":
        return _resolve("ollama_url", "FDI_PLN__OLLAMA_URL")

    if name == "MODEL":
        return _resolve("model", "FDI_PLN__MODEL")

    if name == "GOLD_RESOURCE_NAME":
        return _resolve("gold_resource_name", "FDI_PLN__GOLD_RESOURCE_NAME")

    if name == "ALIAS":
        return _resolve("alias", "FDI_PLN__ALIAS")

    if name == "SINGLE_PLAYER_MODE":
        return _resolve("modo_monopuesto", "FDI_PLN__MODO_MONOPUESTO", _resolve_bool)

    if name == "LETTERS_BEFORE_REBROADCAST":
        return _resolve("letters_before_rebroadcast", "FDI_PLN__LETTERS_BEFORE_REBROADCAST", int)

    if name == "OFFERS_PER_PERSON":
        return _resolve("offers_per_person", "FDI_PLN__OFFERS_PER_PERSON", int)

    base = __getattr__("API_BASE")
    if name == "MAILBOX_ENDPOINT":
        path = _resolve("mailbox_endpoint", "FDI_PLN__MAILBOX_ENDPOINT")
        return base + path
    if name == "LETTER_ENDPOINT":
        path = _resolve("letter_endpoint", "FDI_PLN__LETTER_ENDPOINT")
        return base + path
    if name == "PACKAGE_ENDPOINT":
        path = _resolve("package_endpoint", "FDI_PLN__PACKAGE_ENDPOINT")
        return base + path

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- CLI ---

@click.command()
@click.option(
    "--api-base",
    help="URL base de la API (ej. http://127.0.0.1:7719). Env: FDI_PLN__BUTLER_ADDRESS.",
)
@click.option(
    "--ollama-url",
    help="URL de la API de Ollama. Env: FDI_PLN__OLLAMA_URL.",
)
@click.option(
    "--model",
    help="Modelo de Ollama (ej. mistral:7b). Env: FDI_PLN__MODEL.",
)
@click.option(
    "--gold-resource-name",
    help="Nombre del recurso oro. Env: FDI_PLN__GOLD_RESOURCE_NAME.",
)
@click.option(
    "--mailbox-endpoint",
    help="Endpoint del buzón (ej. /buzon). Env: FDI_PLN__MAILBOX_ENDPOINT.",
)
@click.option(
    "--letter-endpoint",
    help="Endpoint de cartas (ej. /carta). Env: FDI_PLN__LETTER_ENDPOINT.",
)
@click.option(
    "--package-endpoint",
    help="Endpoint de paquetes (ej. /paquete). Env: FDI_PLN__PACKAGE_ENDPOINT.",
)
@click.option(
    "--alias",
    help="Alias del agente. Env: FDI_PLN__ALIAS.",
)
@click.option(
    "--modo-monopuesto/--no-modo-monopuesto",
    "modo_monopuesto",
    default=None,
    help="Activar modo monopuesto. Env: FDI_PLN__MODO_MONOPUESTO (true/1/yes).",
)
@click.option(
    "--letters-before-rebroadcast",
    type=int,
    help="Cartas procesadas antes de reenviar ofertas. Env: FDI_PLN__LETTERS_BEFORE_REBROADCAST.",
)
@click.option(
    "--offers-per-person",
    type=int,
    help="Ofertas aleatorias por persona. Env: FDI_PLN__OFFERS_PER_PERSON.",
)
def cli(
    api_base: str | None,
    ollama_url: str | None,
    model: str | None,
    gold_resource_name: str | None,
    mailbox_endpoint: str | None,
    letter_endpoint: str | None,
    package_endpoint: str | None,
    alias: str | None,
    modo_monopuesto: bool | None,
    letters_before_rebroadcast: int | None,
    offers_per_person: int | None,
) -> None:
    """Ejecuta el bot de negociación. Prioridad: env > args Click > config.json."""
    init_cli_overrides({
        "api_base": api_base,
        "ollama_url": ollama_url,
        "model": model,
        "gold_resource_name": gold_resource_name,
        "mailbox_endpoint": mailbox_endpoint,
        "letter_endpoint": letter_endpoint,
        "package_endpoint": package_endpoint,
        "alias": alias,
        "modo_monopuesto": modo_monopuesto,
        "letters_before_rebroadcast": letters_before_rebroadcast,
        "offers_per_person": offers_per_person,
    })
    from .app import main
    main()
