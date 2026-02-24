"""
Lógica principal del bot: flujo de negociación (main) y flujo legacy.
"""

import json
import random
import time
import requests

from . import api
from .config import ALIAS, GOLD_RESOURCE_NAME
from .game_state import State
from .letters import (
    analizar_carta,
    build_status_letter,
    build_simple_offer_letter,
    build_surplus_for_gold_letter,
)
from . import logs
from .logs import (
    print_section,
    print_kv,
    print_carta_estado,
    print_carta_cruda,
    print_llm,
    print_error,
    print_bot,
    print_bot_dim,
    print_buzon,
)
from .trader import handle_offer, handle_confirmation


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

    state = State(alias="", inventario={}, objetivo={}, needs={}, surplus={}, buzon={})
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
    print_kv("Inventario inicial", json.dumps(state.inventario, ensure_ascii=False))
    print_kv("Objetivo de recursos", json.dumps(state.objetivo, ensure_ascii=False))

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

    if state.has_reached_objective() and state.surplus:
        print_bot("Objetivo alcanzado. Enviando una oferta surplus→oro aleatoria por persona.", success=True)
        surplus_list = list(state.surplus.keys())
        for p in people:
            alias = p.get("alias") or p.get("Alias") if isinstance(p, dict) else p
            if not alias or alias == state.alias:
                continue
            recurso_sobrante = random.choice(surplus_list)
            cuerpo = build_surplus_for_gold_letter(recurso_sobrante, GOLD_RESOURCE_NAME)
            asunto = f"Oferta: 1 {recurso_sobrante} por 1 {GOLD_RESOURCE_NAME}"
            try:
                print_kv(
                    "Enviando oferta surplus→oro a",
                    f"{alias} -> {asunto}",
                    color=logs.GREEN,
                )
                api.send_letter(alias, asunto, cuerpo)
            except Exception as e:
                print_error(f"al enviar oferta a {p}: {e}")
    else:
        # Caso normal: dos ofertas aleatorias (necesario↔sobrante) por persona
        pares_oferta = [
            (n, s) for n in state.needs.keys() for s in state.surplus.keys()
        ]
        if pares_oferta:
            print_bot("Enviando 2 ofertas aleatorias por persona.", success=True)
            for p in people:
                alias = p.get("alias") or p.get("Alias") if isinstance(p, dict) else p
                if not alias or alias == state.alias:
                    continue
                elegidos = random.choices(pares_oferta, k=min(2, len(pares_oferta)))
                for recurso_necesario, recurso_sobrante in elegidos:
                    cuerpo = build_simple_offer_letter(
                        recurso_necesario=recurso_necesario,
                        recurso_sobrante=recurso_sobrante,
                    )
                    asunto = f"Oferta: 1 {recurso_necesario} por 1 {recurso_sobrante}"
                    try:
                        print_kv(
                            "Enviando mini oferta a",
                            f"{alias} -> {asunto}",
                            color=logs.GREEN,
                        )
                        api.send_letter(alias, asunto, cuerpo)
                    except Exception as e:
                        print_error(f"al enviar mini oferta a {p}: {e}")

    # 1) Leer buzón una vez (ya está en state.buzon); luego bucle 2–4
    # Cuando alcanzamos objetivo, no salimos: enviamos ofertas surplus→oro y seguimos maximizando oro.
    print_section("BUZÓN INICIAL")
    print_kv("Acción", "Leyendo cartas del buzón")
    print_buzon(state.buzon)

    cartas_analizadas = 0
    while True:
        # 2) Ordenar cartas por fecha (más antiguas primero)
        sorted_letters = sorted(
            state.buzon.items(),
            key=lambda item: item[1].get("fecha", ""),
        )

        # 3) Procesar de más antigua a más nueva y eliminar del buzón
        for id_carta, content in sorted_letters:
            remitente = content.get("remi", "??")
            asunto = content.get("asunto", "")
            fecha = content.get("fecha", "")

            if remitente == state.alias:
                api.delete_letter(id_carta)
                continue

            print_section(f"CARTA RECIBIDA de {remitente}")
            print_kv("ID", id_carta)
            print_kv("Remitente", remitente)
            print_kv("Asunto", asunto)
            print_kv("Fecha", fecha)
            print_carta_cruda(content)

            analisis = analizar_carta(content, state.needs, state.surplus)
            print_section("ANÁLISIS LLM DE LA CARTA")
            print_llm(analisis)
            cartas_analizadas += 1

            tipo = analisis.get("tipo", "otro")

            state.update()

            if tipo == "oferta":
                remitente = content.get("remi")
                if not remitente:
                    print_bot("Oferta sin remitente claro, se ignora.", warning=True)
                else:
                    print_kv(
                        "Acción", f"Gestionando OFERTA de {remitente}", color=logs.GREEN
                    )
                    handle_offer(
                        remitente,
                        analisis,
                        state.needs,
                        state.surplus,
                        state.inventario,
                    )
            elif tipo == "confirmacion":
                remitente = content.get("remi")
                if not remitente:
                    print_bot(
                        "Confirmación sin remitente claro, se ignora.",
                        warning=True,
                    )
                else:
                    print_kv(
                        "Acción",
                        f"Gestionando CONFIRMACIÓN de {remitente}",
                        color=logs.GREEN,
                    )
                    handle_confirmation(
                        remitente, analisis, state.inventario, state.needs
                    )

            print_bot_dim(f"[BOT] Eliminando carta del buzón (id={id_carta})")
            api.delete_letter(id_carta)

        # Cada 5 cartas analizadas: ofertas extra (oro si objetivo cumplido, sino 2 aleatorias por persona)
        if cartas_analizadas >= 5:
            cartas_analizadas = 0
            if state.has_reached_objective() and state.surplus:
                surplus_list = list(state.surplus.keys())
                print_bot(
                    "5 cartas analizadas. Enviando una oferta surplus→oro aleatoria por persona.",
                    success=True,
                )
                for p in people:
                    alias = p.get("alias") or p.get("Alias") if isinstance(p, dict) else p
                    if not alias or alias == state.alias:
                        continue
                    recurso_sobrante = random.choice(surplus_list)
                    cuerpo = build_surplus_for_gold_letter(
                        recurso_sobrante, GOLD_RESOURCE_NAME
                    )
                    asunto = f"Oferta: 1 {recurso_sobrante} por 1 {GOLD_RESOURCE_NAME}"
                    try:
                        print_kv(
                            "Enviando oferta surplus→oro a",
                            f"{alias} -> {asunto}",
                            color=logs.GREEN,
                        )
                        api.send_letter(alias, asunto, cuerpo)
                    except Exception as e:
                        print_error(f"al enviar oferta surplus→oro a {alias}: {e}")
            elif state.needs and state.surplus:
                pares_oferta = [
                    (n, s) for n in state.needs.keys() for s in state.surplus.keys()
                ]
                print_bot(
                    "5 cartas analizadas. Enviando 2 ofertas aleatorias por persona.",
                    success=True,
                )
                for p in people:
                    alias = p.get("alias") or p.get("Alias") if isinstance(p, dict) else p
                    if not alias or alias == state.alias:
                        continue
                    elegidos = random.choices(pares_oferta, k=min(2, len(pares_oferta)))
                    for recurso_necesario, recurso_sobrante in elegidos:
                        cuerpo = build_simple_offer_letter(
                            recurso_necesario=recurso_necesario,
                            recurso_sobrante=recurso_sobrante,
                        )
                        asunto = f"Oferta: 1 {recurso_necesario} por 1 {recurso_sobrante}"
                        try:
                            print_kv(
                                "Enviando mini oferta a",
                                f"{alias} -> {asunto}",
                                color=logs.GREEN,
                            )
                            api.send_letter(alias, asunto, cuerpo)
                        except Exception as e:
                            print_error(f"al enviar mini oferta a {alias}: {e}")

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
        print_buzon(state.buzon)
