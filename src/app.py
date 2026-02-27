"""
Lógica principal del bot: flujo de negociación (main) y flujo legacy.
"""

import json
import time
import requests

from . import api
from .config import ALIAS, LETTERS_BEFORE_REBROADCAST, OFFERS_PER_PERSON, REMITENTE_SISTEMA
from .game_state import State
from .agent import parse_letter
from .letters import broadcast_offers
from . import logs
from .logs import (
    print_section,
    print_kv,
    print_status_letter,
    print_raw_letter,
    print_llm,
    print_error,
    print_bot,
    print_bot_dim,
    print_mailbox,
)
from .trader import handle_confirmation, handle_offer


def _process_letter(letter_id: str, content: dict, state: State) -> str | None:
    """
    Procesa una carta: analiza, gestiona oferta/confirmación. Retorna tipo de carta o None.
    No borra la carta del buzón (lo hace el llamador).
    """
    sender = content.get("remi", "??")

    if sender == state.alias:
        return None

    print_section(f"CARTA RECIBIDA de {sender}")
    print_kv("ID", letter_id)
    print_kv("Remitente", sender)
    print_kv("Asunto", content.get("asunto", ""))
    print_kv("Fecha", content.get("fecha", ""))
    print_raw_letter(content)

    analysis = parse_letter(content, state.needs, state.surplus)
    print_section("ANÁLISIS LLM DE LA CARTA")
    print_llm(analysis)

    letter_type = analysis.get("tipo", "otro")
    state.update()

    if letter_type == "oferta":
        sender_val = content.get("remi")
        if not sender_val:
            print_bot("Oferta sin remitente claro, se ignora.", warning=True)
        else:
            print_kv("Acción", f"Gestionando OFERTA de {sender_val}", color=logs.GREEN)
            handle_offer(sender_val, analysis, state.needs, state.surplus, state.inventory)
    elif letter_type == "confirmacion":
        sender_val = content.get("remi")
        if not sender_val:
            print_bot("Confirmación sin remitente claro, se ignora.", warning=True)
        else:
            print_kv("Acción", f"Gestionando CONFIRMACIÓN de {sender_val}", color=logs.GREEN)
            handle_confirmation(sender_val, analysis, state.inventory, state.needs, state.surplus)

    return letter_type


def main() -> None:
    """
    Flujo de negociación:
    1) Leer /info y construir estado (alias, inventario, objetivo, buzón).
    2) Enviar a todos una carta preescrita con lo que tenemos y necesitamos.
    3) Leer buzón del estado, analizar cada carta y actuar (ofertas/confirmaciones).
    4) Si cambian nuestros recursos, reenviar carta de estado actualizada.
    """
    print_section("INICIO DEL BOT")

    # Configuramos nuestro alias según la configuración (doc: POST /alias/{nombre})
    if ALIAS:
        try:
            print_kv("Alias configurado", ALIAS)
            api.set_alias(ALIAS)
        except Exception as e:
            print_error(f"No se pudo configurar el alias '{ALIAS}': {e}")

    print_kv("Acción", "Obteniendo nuestros recursos (/info)")

    state = State(alias="", inventory={}, target={}, needs={}, surplus={}, mailbox={})
    while True:
        try:
            state.update()
            break
        except requests.exceptions.RequestException as e:
            print_error(f"Error leyendo /info al arrancar: {e}")
            print_bot("Reintentando en 5 s...", warning=True)
            time.sleep(5)

    print_section("ESTADO INICIAL")
    print_kv("Alias", state.alias)
    print_kv("Inventario inicial", json.dumps(state.inventory, ensure_ascii=False))
    print_kv("Objetivo de recursos", json.dumps(state.target, ensure_ascii=False))

    print_section("AGENTES")
    print_kv("Acción", "Obteniendo agentes (/gente)")
    people = api.remove_myself({"Alias": state.alias}, api.get_people())
    print_kv("Otros agentes", [p.get("alias", p) for p in people])

    print_section("NECESIDADES Y EXCEDENTES")
    print_kv("Necesitamos", json.dumps(state.needs, ensure_ascii=False))
    print_kv(
        "Podemos ofrecer (incluido oro, aunque luego lo filtraremos al enviar)",
        json.dumps(state.surplus, ensure_ascii=False),
    )

    # En lugar de una carta gigante, mandamos "mini cartas" 1 a 1.
    # Si ya cumplimos objetivo: ofrecemos surplus a cambio de oro (maximizar oro).
    # Si no: combinamos cada recurso que necesitamos con cada recurso que nos sobra.
    print_section("CARTAS DE OFERTA SIMPLES A ENVIAR")
    broadcast_offers(people, state, offers_per_person=OFFERS_PER_PERSON)

    # 1) Leer buzón una vez (ya está en state.mailbox); luego bucle 2–4
    # Cuando alcanzamos objetivo, no salimos: enviamos ofertas surplus→oro y seguimos maximizando oro.
    print_section("BUZÓN INICIAL")
    print_kv("Acción", "Leyendo cartas del buzón")
    print_mailbox(state.mailbox)

    letters_processed = 0
    while True:
        # 2) Ordenar cartas por fecha (más antiguas primero)
        sorted_letters = sorted(
            state.mailbox.items(),
            key=lambda item: item[1].get("fecha", ""),
        )

        # 3) Procesar de más antigua a más nueva y eliminar del buzón
        for letter_id, content in sorted_letters:
            if content.get("remi") == state.alias:
                api.delete_letter(letter_id)
                continue
            if content.get("remi") == REMITENTE_SISTEMA:
                print_bot_dim(f"[BOT] Ignorando carta de {REMITENTE_SISTEMA} (no se procesa)")
                api.delete_letter(letter_id)
                continue

            letter_type = _process_letter(letter_id, content, state)
            if letter_type is not None:
                letters_processed += 1

            print_bot_dim(f"[BOT] Eliminando carta del buzón (id={letter_id})")
            api.delete_letter(letter_id)

        # Cada N cartas analizadas: ofertas extra
        if letters_processed >= LETTERS_BEFORE_REBROADCAST:
            letters_processed = 0
            reason = f"{LETTERS_BEFORE_REBROADCAST} cartas analizadas. "
            broadcast_offers(people, state, offers_per_person=OFFERS_PER_PERSON, reason=reason)

        # 4) No hay cartas (o ya se procesaron): esperar 5 s y volver a leer buzón
        print_section("BUZÓN VACÍO")
        print_bot(
            "Sin cartas en buzón. Esperando 5 s y releyendo buzón...",
            warning=True,
        )
        time.sleep(5)
        try:
            state.update()
        except requests.exceptions.RequestException as e:
            print_error(f"Error refrescando estado: {e}")
            continue
        print_mailbox(state.mailbox)
