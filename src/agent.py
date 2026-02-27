"""
Agente LLM: interpreta cartas y decide si aceptar/rechazar ofertas.

Usa Ollama para extraer estructura JSON de las cartas (oferta/confirmación)
y para evaluar si una oferta cumple nuestras condiciones de intercambio.
"""

import json
import re
from typing import Any

import requests

from .config import GOLD_RESOURCE_NAME, MAX_INTENTOS_OFERTA
from . import logs
from .ollama_client import ollama


def _parse_json_response(response: str) -> dict[str, Any] | None:
    """
    Intenta parsear JSON aunque el modelo añada texto antes/después.
    Devuelve dict o None si no encuentra ningún objeto JSON válido.
    """
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(response):
        if ch != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(response[idx:])
            if isinstance(candidate, dict):
                return candidate
        except json.JSONDecodeError:
            continue
    return None


def _gold_only(needs: dict[str, Any], surplus: dict[str, int], inventory: dict[str, int]) -> bool:
    """True si no hemos alcanzado objetivo, no tenemos surplus y tenemos al menos 1 oro."""
    if len(needs) == 0 or surplus:
        return False
    return inventory.get(GOLD_RESOURCE_NAME, 0) >= 1


def _text_has_explicit_quantity(text: str) -> bool:
    """True si el texto contiene cantidades numéricas explícitas."""
    return bool(re.search(r"\b\d+\b", text or ""))


def _normalize_amounts_if_ambiguous(
    parsed: dict[str, Any],
    letter_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Si la carta no trae números explícitos, evita cantidades inventadas por el LLM
    normalizando cada recurso detectado a 1 unidad.
    """
    text = f"{letter_data.get('asunto', '')} {letter_data.get('cuerpo', '')}"
    if _text_has_explicit_quantity(text):
        return parsed

    for key in ("oferta", "pide", "recursos_recibidos"):
        block = parsed.get(key) or {}
        if isinstance(block, dict):
            normalized: dict[str, int] = {}
            for resource, amount in block.items():
                if resource is None:
                    continue
                try:
                    amount_num = int(amount)
                except (TypeError, ValueError):
                    amount_num = 1
                if amount_num > 0:
                    normalized[str(resource)] = 1
            parsed[key] = normalized
    return parsed


def _normalize_resource_map(value: Any) -> dict[str, int]:
    """
    Normaliza múltiples variantes de recursos al formato canónico:
    {"queso": 1, "aceite": 2}
    """
    out: dict[str, int] = {}

    def _add(resource: Any, amount: Any = 1) -> None:
        if not isinstance(resource, str):
            return
        name = resource.strip()
        if not name:
            return
        try:
            qty = int(amount)
        except (TypeError, ValueError):
            qty = 1
        if qty <= 0:
            return
        out[name] = out.get(name, 0) + qty

    if isinstance(value, dict):
        # Caso: {"recurso":"queso","cantidad":2}
        if "recurso" in value:
            _add(value.get("recurso"), value.get("cantidad", 1))
            return out

        # Caso canónico o semi-canónico: {"queso":1} / {"queso":"2"} / {"queso":"mucho"}
        for k, v in value.items():
            if k == "cantidad":
                continue
            if isinstance(v, dict) and "cantidad" in v:
                _add(k, v.get("cantidad", 1))
                continue
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            try:
                qty = int(v)
            except (TypeError, ValueError):
                # En formato canónico, texto no numérico se ignora.
                continue
            _add(k, qty)
        return out

    # Caso lista: [{"recurso":"queso","cantidad":2}, ...]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if "recurso" in item:
                    _add(item.get("recurso"), item.get("cantidad", 1))
                else:
                    for k, v in item.items():
                        if v is None:
                            continue
                        if isinstance(v, str) and not v.strip():
                            continue
                        try:
                            qty = int(v)
                        except (TypeError, ValueError):
                            continue
                        _add(k, qty)
    return out


def _normalize_parsed_letter_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Fuerza campos de carta al esquema esperado por el resto del flujo.
    """
    parsed["oferta"] = _normalize_resource_map(parsed.get("oferta"))
    parsed["pide"] = _normalize_resource_map(parsed.get("pide"))
    parsed["recursos_recibidos"] = _normalize_resource_map(parsed.get("recursos_recibidos"))
    return parsed


def _sanitize_decision_against_received_offer(
    decision_data: dict[str, Any],
    received_offer: dict[str, Any],
) -> dict[str, Any]:
    """
    Acota la decisión del LLM a los recursos/cantidades detectados en la carta.
    Evita que el segundo análisis "invente" recursos o cantidades mayores.
    """
    def _to_positive_int_map(value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in value.items():
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n > 0:
                out[str(k)] = n
        return out

    allowed_offer = _to_positive_int_map(received_offer.get("oferta"))
    allowed_request = _to_positive_int_map(received_offer.get("pide"))

    def _clamp_section(section_key: str, allowed: dict[str, int]) -> dict[str, int]:
        current = _to_positive_int_map(decision_data.get(section_key))
        clamped: dict[str, int] = {}
        for resource, amount in current.items():
            if resource not in allowed:
                continue
            clamped[resource] = min(amount, allowed[resource])
        return clamped

    decision_data["oferta"] = _clamp_section("oferta", allowed_offer)
    decision_data["pide"] = _clamp_section("pide", allowed_request)
    return decision_data


def parse_letter(
    letter_data: dict[str, Any],
    needs: dict[str, Any],
    surplus: dict[str, int],
) -> dict[str, Any]:
    """
    Usa Ollama para interpretar una carta y devolver un JSON con
    tipo (oferta|confirmacion|otro), oferta, pide, recursos_recibidos.
    """
    prompt = f"""
Eres un asistente que ayuda a interpretar cartas de intercambio de recursos
entre agentes en un juego.

Tu tarea es LEER la carta y devolver un JSON estructurado con esta forma:

{{
  "tipo": "oferta" | "confirmacion" | "otro",
  "oferta": {{
    "recurso": cantidad entero
  }},
  "pide": {{
    "recurso": cantidad entero
  }},
  "recursos_recibidos": {{
    "recurso": cantidad entero
  }}
}}

Donde:
- "tipo" = "oferta" si la carta propone un intercambio (yo te doy X, tú me das Y).
- "tipo" = "confirmacion" si la carta dice que ya nos han enviado recursos.
- "tipo" = "otro" si no encaja claramente en ninguno de los casos.
- "oferta" describe lo que EL OTRO agente nos ofrece.
- "pide" describe lo que EL OTRO agente quiere que le enviemos.
- "recursos_recibidos" son los recursos que el agente afirma que YA nos ha enviado.

IMPORTANTE:
- Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional.
- Si algún campo no está claro en la carta, devuélvelo como un objeto vacío {{}}.

OFRECEMOS:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

NECESITAMOS:
{json.dumps(needs, ensure_ascii=False, indent=2)}

CARTA RECIBIDA:
{json.dumps(letter_data, ensure_ascii=False, indent=2)}
"""
    try:
        response = ollama(prompt)
    except (
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
    ):
        logs.print_error("No se pudo analizar la carta (timeout/conexión/modelo no encontrado); se usa fallback.")
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}

    logs.print_llm_response(response)

    data = _parse_json_response(response)
    if data is not None:
        data = _normalize_parsed_letter_schema(data)
        return _normalize_amounts_if_ambiguous(data, letter_data)
    else:
        logs.print_error("Ollama no devolvió JSON válido al analizar carta")
        logs.print_llm_response(response)
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}


def analyze_offer(
    offer: dict[str, Any],
    needs: dict[str, Any],
    surplus: dict[str, int],
    inventory: dict[str, int],
) -> dict[str, Any]:
    """
    Usa Ollama para decidir si aceptar o rechazar una oferta.
    Devuelve un JSON con decision (aceptada|rechazada), oferta y pide.
    Si needs está vacío (objetivo cumplido), acepta ofertas que nos den oro a cambio de surplus.
    Si solo tenemos oro (surplus vacío, tenemos needs y >= 1 oro), acepta ofertas de 1 recurso a cambio de 1 oro.
    """
    objective_met = len(needs) == 0
    gold_only = _gold_only(needs, surplus, inventory)

    if objective_met:
        rules = f"""
- Ya hemos cumplido el objetivo de recursos. Ahora queremos MAXIMIZAR ORO.
- "decision" = "aceptada" si:
    a) nos ofrecen {GOLD_RESOURCE_NAME} (oro)
    b) nos piden recursos que tenemos en surplus (PODEMOS OFRECER)
    c) podemos dar las cantidades que pide
    d) es razonable (p. ej. 1:1 o menos recursos que recibimos de oro)
- "decision" = "rechazada" si no nos ofrecen oro o piden algo que no tenemos en surplus.
"""
    elif gold_only:
        rules = f"""
- Solo tenemos {GOLD_RESOURCE_NAME} para ofrecer y no hemos alcanzado el objetivo.
- "decision" = "aceptada" si:
    a) nos ofrecen 1 unidad de CUALQUIER recurso (no necesariamente de NECESITAMOS)
    b) nos piden exactamente 1 {GOLD_RESOURCE_NAME} (y tenemos al menos 1)
- "decision" = "rechazada" si no nos ofrecen ningún recurso o no piden 1 oro.
"""
    else:
        rules = """
- "decision" = "aceptada" si se cumplen TODAS las condiciones siguientes:
    a) los recursos que ofrece los necesitamos para cumplir el objetivo
    b) no pide recursos que necesitemos para nuestro objetivo
    c) podemos dar los recursos que pide
    d) se envían como máximo el mismo número de recursos que se reciben, salvo que con la oferta completemos el objetivo al 100%

- "decision" = "rechazada" si no se cumple alguna de las condiciones anteriores
"""
    prompt = f"""
Eres un asistente que toma decisiones sobre ofertas recibidas.

Tu tarea es LEER la oferta y devolver un JSON estructurado con esta forma:

{{
  "decision": "aceptada" | "rechazada",
  "oferta": {{
    "recurso": cantidad entero
  }},
  "pide": {{
    "recurso": cantidad entero
  }},

}}

Donde:
{rules}
- "oferta" describe lo que EL OTRO agente nos ofrece.
- "pide" describe lo que EL OTRO agente quiere que le enviemos.
- No hace falta aceptar la oferta al completo, se puede aceptar parcialmente solo los recursos que nos interesen y que cumplan las condiciones anteriores.
IMPORTANTE:
- Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional.

NECESITAMOS:
{json.dumps(needs, ensure_ascii=False, indent=2)}

PODEMOS OFRECER:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

OFERTA:
{json.dumps(offer, ensure_ascii=False, indent=2)}

"""
    max_attempts = MAX_INTENTOS_OFERTA
    for attempt in range(max_attempts):
        try:
            response = ollama(prompt)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
        ):
            if attempt < max_attempts - 1:
                logs.print_retry(f"Intento {attempt + 1}/{max_attempts}: Error con Ollama, reintentando...")
            else:
                logs.print_error("No se pudo contactar Ollama (timeout/404/modelo); se rechaza la oferta por defecto.")
                return {"decision": "rechazada", "oferta": {}, "pide": {}}
        data = _parse_json_response(response)
        if data is not None:
            return _sanitize_decision_against_received_offer(data, offer)
        else:
            if attempt < max_attempts - 1:
                logs.print_retry(f"Intento {attempt + 1}/{max_attempts}: JSON inválido, reintentando...")
            else:
                logs.print_error(f"Ollama no devolvió JSON válido al analizar oferta (tras {max_attempts} intentos)")
                logs.print_llm_response(response)
                return {"decision": "rechazada", "oferta": {}, "pide": {}}
