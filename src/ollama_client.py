"""
Cliente para Ollama (generación con LLM local).
Soporta JSON Schema en `format` para forzar salida estructurada.
"""

from typing import Any, Dict, Optional

import requests

from .config import MODEL, OLLAMA_URL


def ollama(prompt: str, format: Optional[Dict[str, Any]] = None) -> str:
    """
    Llama al modelo Ollama. Si se pasa `format` (JSON Schema), la respuesta
    se fuerza a cumplir ese esquema (JSON Schema–guided generation).
    """
    payload: Dict[str, Any] = {
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
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"]

    except requests.exceptions.HTTPError as e:
        print("ERROR HTTP OLLAMA:", r.status_code)
        print(r.text)
        raise

    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama no está corriendo (ollama serve)")
        raise
