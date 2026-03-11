"""
Cliente HTTP para Ollama: generación de texto con LLM local.

Permite opcionalmente forzar salida JSON con un schema en `format`.
"""

from typing import Any

import requests

from .config import MODEL, OLLAMA_URL
from . import logs


def ollama(prompt: str, format: dict[str, Any] | None = None) -> str:
    """
    Llama al modelo Ollama. Si se pasa `format` (JSON Schema), la respuesta
    se fuerza a cumplir ese esquema (JSON Schema-guided generation).
    """
    payload: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if format is not None:
        payload["format"] = format
    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=500,
        )
        r.raise_for_status()
        return r.json()["response"]

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
    ):
        logs.print_error("Timeout al llamar a Ollama (se superó el tiempo de espera)")
        raise

    except requests.exceptions.HTTPError:
        logs.print_error(f"HTTP OLLAMA: {r.status_code}\n{r.text}")
        raise

    except requests.exceptions.ConnectionError:
        logs.print_error("Ollama no está corriendo (ollama serve)")
        raise
