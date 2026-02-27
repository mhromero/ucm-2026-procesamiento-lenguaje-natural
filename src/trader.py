"""
Lógica de intercambio: evaluación de ofertas/confirmaciones y envío de paquetes.

Comprueba condiciones (no enviar oro innecesariamente, tener stock) antes
de aceptar y ejecuta el envío de paquetes y cartas de confirmación.
"""

from typing import Any

from . import api
from . import logs
from .agent import _gold_only, analyze_offer
from .config import GOLD_RESOURCE_NAME
from .letters import build_trade_confirmation_letter


def _validate_resources_to_send(
    resources_to_send: dict[str, int],
    needs: dict[str, Any],
    surplus: dict[str, int],
    inventory: dict[str, int],
    gold_only: bool,
) -> str | None:
    """
    Comprueba si podemos enviar los recursos (no oro salvo gold_only, no needs, stock).
    Devuelve None si es válido, o motivo de rechazo si no.
    """
    gold_to_send = resources_to_send.get(GOLD_RESOURCE_NAME, 0)
    if gold_to_send > 0:
        if not gold_only or gold_to_send > 1:
            return "No enviamos oro (o solo permitimos 1 oro cuando solo tenemos oro para cambiar)."
        if inventory.get(GOLD_RESOURCE_NAME, 0) < 1:
            return "No tenemos suficiente oro para enviar."

    for resource, amount in resources_to_send.items():
        if needs.get(resource, 0) > 0:
            return f"No enviamos recurso que necesitamos para el objetivo: {resource}."
        if inventory.get(resource, 0) < amount:
            return f"No tenemos suficientes '{resource}' (tenemos {inventory.get(resource, 0)}, piden {amount})."
    return None


def evaluate_offer(
    analysis: dict[str, Any],
    needs: dict[str, Any],
    surplus: dict[str, int],
    inventory: dict[str, int],
) -> dict[str, Any]:
    """
    Procesa una oferta: decide si se acepta y comprueba todas las condiciones
    (no enviar oro, no enviar lo que necesitamos, tener stock suficiente).

    Estructura devuelta:
    {
      "aceptada": bool,
      "motivo": str,
      "oferta": dict[str, int],
      "pide": dict[str, int],
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

    gold_only = _gold_only(needs, surplus, inventory)
    motivo = _validate_resources_to_send(
        resources_to_send, needs, surplus, inventory, gold_only
    )
    if motivo:
        return {
            "aceptada": False,
            "motivo": motivo,
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
    analysis: dict[str, Any],
    inventory: dict[str, int],
    needs: dict[str, Any],
    surplus: dict[str, int],
) -> dict[str, Any]:
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
      "recursos_recibidos": dict[str, int],
      "pide": dict[str, int],
      "recursos_a_enviar": dict[str, int] | {}
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

    gold_only = _gold_only(needs, surplus, inventory)
    motivo = _validate_resources_to_send(
        resources_to_send, needs, surplus, inventory, gold_only
    )
    if motivo:
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": motivo,
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
    analysis: dict[str, Any],
    needs: dict[str, Any],
    surplus: dict[str, int],
    inventory: dict[str, int],
) -> bool:
    """
    Procesa una oferta: decide, comprueba condiciones, envía paquete y carta
    de confirmación si se acepta. Devuelve True si nuestros recursos cambiaron.
    """
    result = evaluate_offer(analysis, needs, surplus, inventory)
    logs.print_decision("Decisión sobre la oferta", result)

    if not result.get("aceptada"):
        logs.print_bot(f"Oferta rechazada: {result.get('motivo')}", warning=True)
        return False

    offer = result.get("oferta") or {}
    resources_to_send = result.get("recursos_a_enviar") or {}
    if not resources_to_send:
        logs.print_bot(
            "Oferta aceptada pero sin recursos a enviar (resultado vacío), no se realiza envío.",
            warning=True,
        )
        return False

    try:
        logs.print_bot(
            f"Aceptando oferta de {sender}. Enviando paquete: {resources_to_send}",
            success=True,
        )
        api.send_package(sender, resources_to_send)
    except Exception as e:
        logs.print_error(f"Enviando paquete de oferta a {sender}: {e}")
        return False

    try:
        confirmation_letter = build_trade_confirmation_letter(
            resources_sent=resources_to_send,
            resources_expected=offer,
        )
        logs.print_bot_dim(
            f"→ Enviando carta de confirmación de oferta aceptada a {sender}..."
        )
        api.send_letter(sender, "Confirmación de oferta aceptada", confirmation_letter)
    except Exception as e:
        logs.print_error(f"Enviando carta de confirmación a {sender}: {e}")

    return True


def handle_confirmation(
    sender: str,
    analysis: dict[str, Any],
    inventory: dict[str, int],
    needs: dict[str, Any],
    surplus: dict[str, int],
) -> bool:
    """
    Procesa una confirmación: decide, comprueba condiciones, envía paquete
    y carta de confirmación si aplica. Devuelve True si nuestros recursos cambiaron.
    """
    result = evaluate_confirmation(analysis, inventory, needs, surplus)
    logs.print_decision("Decisión sobre la confirmación", result)

    if not result.get("tiene_recursos_recibidos"):
        logs.print_bot(f"No se procesan recursos: {result.get('motivo')}", warning=True)
        return False

    resources_received = result.get("recursos_recibidos") or {}
    resources_to_send = result.get("recursos_a_enviar") or {}

    if result.get("es_regalo"):
        logs.print_bot(
            "Se interpreta la confirmación como regalo, no se envían recursos a cambio.",
        )
        return True

    if not result.get("puede_enviar") or not resources_to_send:
        logs.print_bot(
            f"No se envía paquete de confirmación: {result.get('motivo', 'sin recursos a enviar')}.",
            warning=True,
        )
        return False

    try:
        logs.print_bot(
            f"Confirmación correcta de {sender}. Enviando paquete de vuelta: {resources_to_send}",
            success=True,
        )
        api.send_package(sender, resources_to_send)
    except Exception as e:
        logs.print_error(f"Enviando paquete de confirmación a {sender}: {e}")
        return False

    try:
        confirmation_letter = build_trade_confirmation_letter(
            resources_sent=resources_to_send,
            resources_expected=resources_received,
        )
        logs.print_bot_dim(
            f"→ Enviando carta de confirmación de envío de recursos a {sender}..."
        )
        api.send_letter(
            sender, "Confirmación de envío de recursos", confirmation_letter
        )
    except Exception as e:
        logs.print_error(
            f"Enviando carta de confirmación (confirmación recibida) a {sender}: {e}"
        )

    return True
