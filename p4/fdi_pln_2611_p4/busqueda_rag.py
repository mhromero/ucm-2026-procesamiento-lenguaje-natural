from __future__ import annotations

import ollama
import re

DEFAULT_MODEL = "llama3.2"
TOP_K_CADA = 5


def _extraer_chunks_citados(respuesta: str) -> list[int]:
    ids: list[int] = []
    vistos: set[int] = set()
    for match in re.findall(r"\[Chunk\s+(\d+)\]", respuesta):
        cid = int(match)
        if cid not in vistos:
            vistos.add(cid)
            ids.append(cid)
    return ids


def _resaltar_citas_chunk(texto: str) -> str:
    return re.sub(
        r"\[Chunk\s+(\d+)\]",
        r"[b cyan]Chunk \1[/]",
        texto,
    )


def _bloque_referencias(
    chunks: dict[int, dict],
    ids: list[int],
    scores_por_chunk: dict[int, dict[str, float]],
) -> str:
    fragmentos: list[str] = []
    for cid in ids:
        chunk = chunks.get(cid)
        if chunk is None:
            continue
        headings = " > ".join(chunk.get("headings", []))
        texto = chunk.get("text", "")
        cabecera = f"[b cyan]Chunk {cid}[/]"
        if headings:
            cabecera += f" [{headings}]"
        info_scores = scores_por_chunk.get(cid, {})
        origenes = []
        if "clasica" in info_scores:
            origenes.append(f"clásica (TF-IDF: {info_scores['clasica']:.4f})")
        if "semantica" in info_scores:
            origenes.append(f"semántica (similitud: {info_scores['semantica']:.4f})")
        origen_str = ", ".join(origenes) if origenes else "sin score disponible"
        fragmentos.append(f"{cabecera}\n[dim]Origen: {origen_str}[/]\n\n{texto}")
    if not fragmentos:
        return ""
    return "[b cyan]REFERENCIAS[/]\n\n" + "\n\n---\n\n".join(fragmentos)


def _construir_contexto(
    chunks: dict[int, dict],
    resultados_clasica: list[tuple[int, float, int]],
    resultados_semantica: list[tuple[int, float]],
    top_k: int = TOP_K_CADA,
) -> tuple[str, list[int], dict[int, dict[str, float]]]:
    ids_clasica = [pid for pid, _, _ in resultados_clasica[:top_k]]
    ids_semantica = [pid for pid, _ in resultados_semantica[:top_k]]
    scores_por_chunk: dict[int, dict[str, float]] = {}
    for pid, tfidf, _ in resultados_clasica[:top_k]:
        scores_por_chunk.setdefault(pid, {})["clasica"] = tfidf
    for pid, score in resultados_semantica[:top_k]:
        scores_por_chunk.setdefault(pid, {})["semantica"] = score

    # Unión sin duplicados, manteniendo orden: primero clásica, luego semántica
    ids_vistos: set[int] = set()
    ids_ordenados: list[int] = []
    for pid in ids_clasica + ids_semantica:
        if pid not in ids_vistos:
            ids_vistos.add(pid)
            ids_ordenados.append(pid)

    fragmentos = []
    for pid in ids_ordenados:
        chunk = chunks.get(pid)
        if chunk is None:
            continue
        headings = " > ".join(chunk.get("headings", []))
        texto = chunk.get("text", "")
        cabecera = f"[Chunk {pid}]"
        if headings:
            cabecera += f" [{headings}]"
        fragmentos.append(f"{cabecera}\n{texto}")

    return "\n\n---\n\n".join(fragmentos), ids_ordenados, scores_por_chunk


def buscar_rag(
    query: str,
    chunks: dict[int, dict],
    resultados_clasica: list[tuple[int, float, int]],
    resultados_semantica: list[tuple[int, float]],
    model: str = DEFAULT_MODEL,
    top_k_rag: int = TOP_K_CADA,
) -> str:
    contexto, ids_ordenados, scores_por_chunk = _construir_contexto(
        chunks, resultados_clasica, resultados_semantica, top_k=top_k_rag
    )
    if not contexto:
        return "No se encontraron fragmentos relevantes para responder la pregunta."

    prompt = (
        "Eres un asistente experto en literatura española. "
        "Responde la consulta basándote ÚNICAMENTE en los fragmentos del texto que se te proporcionan. "
        "La consulta puede ser una pregunta completa, una palabra suelta o un grupo corto de palabras. "
        "Si es palabra o frase corta, explica su significado o sentido en el contexto del texto, "
        "menciona cómo se usa en los fragmentos recuperados y aporta una respuesta breve y útil. "
        "No copies fragmentos completos ni cites párrafos largos literalmente; sintetiza con tus palabras. "
        "Si la respuesta no está en los fragmentos, dilo explícitamente. "
        "Responde en español.\n"
        "Cita siempre las evidencias con el formato [Chunk N], justo en la frase donde uses esa evidencia.\n"
        "No agrupes todas las citas al final: deben aparecer inline durante la explicación.\n"
        "No añadas una sección manual de referencias al final.\n\n"
        f"FRAGMENTOS DEL TEXTO:\n{contexto}\n\n"
        f"PREGUNTA: {query}\n\n"
        "RESPUESTA:"
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    respuesta = response.message.content.strip()
    ids_citados = _extraer_chunks_citados(respuesta)
    if not ids_citados:
        ids_citados = ids_ordenados[:3]
        refs_inline = ", ".join(f"[Chunk {pid}]" for pid in ids_citados)
        respuesta = f"{respuesta}\n\nFuentes usadas: {refs_inline}"

    respuesta = _resaltar_citas_chunk(respuesta)
    bloque_refs = _bloque_referencias(chunks, ids_citados, scores_por_chunk)
    if not bloque_refs:
        return respuesta
    separador = "[dim]────────────────────────────────────────[/]"
    return f"{respuesta}\n\n{separador}\n\n{bloque_refs}"
