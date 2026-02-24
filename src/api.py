"""
Llamadas a la API externa: info, gente, cartas y paquetes.
"""

import requests
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from .config import (
    API_BASE,
    LETTER_ENDPOINT,
    MAILBOX_ENDPOINT,
    PACKAGE_ENDPOINT,
    ALIAS,
)

REQUEST_TIMEOUT = 10


def get_info() -> Dict[str, Any]:
    r = requests.get(f"{API_BASE}/info", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_people() -> Any:
    r = requests.get(f"{API_BASE}/gente", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def set_alias(nombre: str) -> Any:
    """Configura nuestro alias en el servidor (POST /alias/{nombre})."""
    r = requests.post(f"{API_BASE}/alias/{nombre}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def remove_myself(info: Dict[str, Any], people: list) -> list:
    """
    Devuelve la lista de agentes sin incluirnos a nosotros mismos.
    Soporta people como lista de strings o lista de dicts {"alias": ...}.
    """
    myself = info.get("Alias") or info.get("alias")
    filtered = []
    for p in people:
        if isinstance(p, str):
            alias = p
        elif isinstance(p, dict):
            alias = p.get("alias") or p.get("Alias")
        else:
            continue
        if alias and alias != myself:
            filtered.append(alias)
    return filtered


def send_letter(to_alias: str, subject: str, body: str) -> Any:
    """
    Envía una carta a otro agente con la estructura definida en la API:
    {
      "remi": "string",
      "dest": "string",
      "asunto": "string",
      "cuerpo": "string",
      "id": "string",
      "fecha": "string"
    }
    """
    payload = {
        "remi": ALIAS or "",
        "dest": to_alias,
        "asunto": subject,
        "cuerpo": body,
        "id": str(uuid4()),
        "fecha": datetime.utcnow().isoformat(),
    }
    r = requests.post(LETTER_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_mailbox() -> Any:
    """Obtiene las cartas del buzón."""
    r = requests.get(MAILBOX_ENDPOINT, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def delete_letter(uid: str) -> Any:
    """Elimina una carta del buzón (DELETE /mail/{uid})."""
    r = requests.delete(f"{API_BASE}/mail/{uid}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def send_package(to_alias: str, resources: Dict[str, int]) -> Any:
    """
    Envía un paquete de recursos a otro agente siguiendo la sintaxis
    documentada para POST /paquete/{dest}.

    Ejemplo de llamada resultante:
      POST /paquete/{dest}
      cuerpo JSON:
      {
        "madera": 4,
        "oro": 2
      }
    """
    # La API espera el alias del destinatario en el path y directamente
    # un objeto con los recursos en el cuerpo.
    r = requests.post(
        f"{PACKAGE_ENDPOINT}/{to_alias}",
        json=resources,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
