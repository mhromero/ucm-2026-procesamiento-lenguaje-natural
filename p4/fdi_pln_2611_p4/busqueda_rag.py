from __future__ import annotations

import ollama

MODEL = "llama3.2"
TOP_K_CADA = 3


def _construir_contexto(
    chunks: list[dict],
    resultados_clasica: list[tuple[int, float, int]],
    resultados_semantica: list[tuple[int, float]],
    top_k: int = TOP_K_CADA,
) -> str:
    ids_clasica = [pid for pid, _, _ in resultados_clasica[:top_k]]
    ids_semantica = [pid for pid, _ in resultados_semantica[:top_k]]

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
        fragmentos.append(f"[{headings}]\n{texto}")

    return "\n\n---\n\n".join(fragmentos)


def buscar_rag(
    query: str,
    chunks: dict[int, dict],
    resultados_clasica: list[tuple[int, float, int]],
    resultados_semantica: list[tuple[int, float]],
) -> str:
    contexto = _construir_contexto(chunks, resultados_clasica, resultados_semantica)
    if not contexto:
        return "No se encontraron fragmentos relevantes para responder la pregunta."

    prompt = (
        "Eres un asistente experto en literatura española. "
        "Responde la pregunta basándote ÚNICAMENTE en los fragmentos del texto que se te proporcionan. "
        "Si la respuesta no está en los fragmentos, dilo explícitamente. "
        "Responde en español.\n\n"
        f"FRAGMENTOS DEL TEXTO:\n{contexto}\n\n"
        f"PREGUNTA: {query}\n\n"
        "RESPUESTA:"
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content
