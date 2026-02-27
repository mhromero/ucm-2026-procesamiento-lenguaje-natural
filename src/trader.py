import json
from typing import Any, Dict

from . import api
from .agent import _gold_only, analyze_offer
from .config import GOLD_RESOURCE_NAME
from .letters import build_trade_confirmation_letter


def evaluate_offer(
    analysis: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
    inventory: Dict[str, int],
) -> Dict[str, Any]:
    """
    Procesa una oferta: decide si se acepta y comprueba todas las condiciones
    (no enviar oro, no enviar lo que necesitamos, tener stock suficiente).

    Estructura devuelta:
    {
      "aceptada": bool,
      "motivo": str,
      "oferta": Dict[str, int],
      "pide": Dict[str, int],
      "recursos_a_enviar": Dict[str, int] | {}
    }
    """
    offer = analysis.get("oferta") or {}
    requested = analysis.get("pide") or {}

    if not offer or not requested:
        return {
            "aceptada": False,
            "motivo": "Oferta sin datos claros (oferta/pide vacíos).",
            "oferta": offer,
            "pide": requested,
            "recursos_a_enviar": {},
        }

    decision = analyze_offer(analysis, needs, surplus, inventory)

    if decision.get("decision") != "aceptada":
        return {
            "aceptada": False,
            "motivo": "Oferta rechazada según análisis del LLM.",
            "oferta": decision.get("oferta") or offer,
            "pide": decision.get("pide") or requested,
            "recursos_a_enviar": {},
        }

    decided_offer = decision.get("oferta") or offer
    decided_request = decision.get("pide") or requested

    try:
        resources_to_send = {k: int(v) for k, v in (decided_request or {}).items()}
    except (TypeError, ValueError):
        return {
            "aceptada": False,
            "motivo": "No se pudo interpretar las cantidades de recursos a enviar.",
            "oferta": decided_offer,
            "pide": decided_request,
            "recursos_a_enviar": {},
        }

    # Comprobaciones de condiciones antes de aceptar
    gold_only = _gold_only(needs, surplus, inventory)
    gold_to_send = resources_to_send.get(GOLD_RESOURCE_NAME, 0)
    if gold_to_send > 0:
        if not gold_only or gold_to_send > 1:
            return {
                "aceptada": False,
                "motivo": "No enviamos oro (o solo permitimos 1 oro cuando solo tenemos oro para cambiar).",
                "oferta": decided_offer,
                "pide": decided_request,
                "recursos_a_enviar": {},
            }
        if inventory.get(GOLD_RESOURCE_NAME, 0) < 1:
            return {
                "aceptada": False,
                "motivo": "No tenemos suficiente oro para enviar.",
                "oferta": decided_offer,
                "pide": decided_request,
                "recursos_a_enviar": {},
            }

    for resource, amount in resources_to_send.items():
        if needs.get(resource, 0) > 0:
            return {
                "aceptada": False,
                "motivo": f"No enviamos recurso que necesitamos para el objetivo: {resource}.",
                "oferta": decided_offer,
                "pide": decided_request,
                "recursos_a_enviar": {},
            }
        if inventory.get(resource, 0) < amount:
            return {
                "aceptada": False,
                "motivo": f"No tenemos suficientes '{resource}' (tenemos {inventory.get(resource, 0)}, piden {amount}).",
                "oferta": decided_offer,
                "pide": decided_request,
                "recursos_a_enviar": {},
            }

    return {
        "aceptada": True,
        "motivo": "Oferta aceptada.",
        "oferta": decided_offer,
        "pide": decided_request,
        "recursos_a_enviar": resources_to_send,
    }


def evaluate_confirmation(
    analysis: Dict[str, Any],
    inventory: Dict[str, int],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
) -> Dict[str, Any]:
    """
    Procesa una confirmación de envío: extrae qué nos han enviado y qué piden
    a cambio, y comprueba las condiciones (no enviar oro, no enviar lo que
    necesitamos, tener stock suficiente) antes de autorizar el envío.

    Estructura devuelta:
    {
      "tiene_recursos_recibidos": bool,
      "es_regalo": bool,
      "puede_enviar": bool,
      "motivo": str,
      "recursos_recibidos": Dict[str, int],
      "pide": Dict[str, int],
      "recursos_a_enviar": Dict[str, int] | {}
    }
    """
    resources_received = analysis.get("recursos_recibidos") or {}
    requested = analysis.get("pide") or {}

    if not resources_received:
        return {
            "tiene_recursos_recibidos": False,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": "Confirmación sin recursos recibidos claros.",
            "recursos_recibidos": {},
            "pide": requested,
            "recursos_a_enviar": {},
        }

    if not requested:
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": True,
            "puede_enviar": False,
            "motivo": "Confirmación sin recursos a enviar a cambio, se asume regalo.",
            "recursos_recibidos": resources_received,
            "pide": {},
            "recursos_a_enviar": {},
        }

    try:
        resources_to_send = {k: int(v) for k, v in requested.items()}
    except (TypeError, ValueError):
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": "No se pudo interpretar las cantidades de recursos a enviar.",
            "recursos_recibidos": resources_received,
            "pide": requested,
            "recursos_a_enviar": {},
        }

    # Comprobaciones de condiciones antes de autorizar el envío
    gold_only = _gold_only(needs, surplus, inventory)
    gold_to_send = resources_to_send.get(GOLD_RESOURCE_NAME, 0)
    if gold_to_send > 0:
        if not gold_only or gold_to_send > 1:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": "No enviamos oro en confirmación (o solo 1 oro cuando solo tenemos oro para cambiar).",
                "recursos_recibidos": resources_received,
                "pide": requested,
                "recursos_a_enviar": {},
            }
        if inventory.get(GOLD_RESOURCE_NAME, 0) < 1:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": "No tenemos suficiente oro para enviar.",
                "recursos_recibidos": resources_received,
                "pide": requested,
                "recursos_a_enviar": {},
            }

    for resource, amount in resources_to_send.items():
        if needs.get(resource, 0) > 0:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": f"No enviamos recurso que necesitamos para el objetivo: {resource}.",
                "recursos_recibidos": resources_received,
                "pide": requested,
                "recursos_a_enviar": {},
            }
        if inventory.get(resource, 0) < amount:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": f"No tenemos suficientes '{resource}' para enviar (tenemos {inventory.get(resource, 0)}, piden {amount}).",
                "recursos_recibidos": resources_received,
                "pide": requested,
                "recursos_a_enviar": {},
            }

    return {
        "tiene_recursos_recibidos": True,
        "es_regalo": False,
        "puede_enviar": True,
        "motivo": "Confirmación con recursos a enviar a cambio.",
        "recursos_recibidos": resources_received,
        "pide": requested,
        "recursos_a_enviar": resources_to_send,
    }


def handle_offer(
    sender: str,
    analysis: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
    inventory: Dict[str, int],
) -> bool:
    """
    Procesa una oferta: decide, comprueba condiciones, envía paquete y carta
    de confirmación si se acepta. Devuelve True si nuestros recursos cambiaron.
    """
    result = evaluate_offer(analysis, needs, surplus, inventory)
    print("Decisión sobre la oferta:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result.get("aceptada"):
        print(f"Oferta rechazada: {result.get('motivo')}")
        return False

    offer = result.get("oferta") or {}
    resources_to_send = result.get("recursos_a_enviar") or {}
    if not resources_to_send:
        print(
            "Oferta aceptada pero sin recursos a enviar (resultado vacío), no se realiza envío."
        )
        return False

    try:
        print(f"Aceptando oferta de {sender}. Enviando paquete: {resources_to_send}")
        api.send_package(sender, resources_to_send)
    except Exception as e:
        print(f"ERROR enviando paquete de oferta a {sender}: {e}")
        return False

    try:
        confirmation_letter = build_trade_confirmation_letter(
            resources_sent=resources_to_send,
            resources_expected=offer,
        )
        print(f"→ Enviando carta de confirmación de oferta aceptada a {sender}...")
        api.send_letter(
            sender, "Confirmación de oferta aceptada", confirmation_letter
        )
    except Exception as e:
        print(f"ERROR enviando carta de confirmación a {sender}: {e}")

    return True


def handle_confirmation(
    sender: str,
    analysis: Dict[str, Any],
    inventory: Dict[str, int],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
) -> bool:
    """
    Procesa una confirmación: decide, comprueba condiciones, envía paquete
    y carta de confirmación si aplica. Devuelve True si nuestros recursos cambiaron.
    """
    result = evaluate_confirmation(analysis, inventory, needs, surplus)
    print("Decisión sobre la confirmación:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result.get("tiene_recursos_recibidos"):
        print(f"No se procesan recursos: {result.get('motivo')}")
        return False

    resources_received = result.get("recursos_recibidos") or {}
    resources_to_send = result.get("recursos_a_enviar") or {}

    if result.get("es_regalo"):
        print(
            "Se interpreta la confirmación como regalo, no se envían recursos a cambio."
        )
        return True

    if not result.get("puede_enviar") or not resources_to_send:
        print(
            f"No se envía paquete de confirmación: {result.get('motivo', 'sin recursos a enviar')}."
        )
        return False

    try:
        print(
            f"Confirmación correcta de {sender}. Enviando paquete de vuelta: {resources_to_send}"
        )
        api.send_package(sender, resources_to_send)
    except Exception as e:
        print(f"ERROR enviando paquete de confirmación a {sender}: {e}")
        return False

    try:
        confirmation_letter = build_trade_confirmation_letter(
            resources_sent=resources_to_send,
            resources_expected=resources_received,
        )
        print(f"→ Enviando carta de confirmación de envío de recursos a {sender}...")
        api.send_letter(
            sender, "Confirmación de envío de recursos", confirmation_letter
        )
    except Exception as e:
        print(
            f"ERROR enviando carta de confirmación (confirmación recibida) a {sender}: {e}"
        )

    return True
