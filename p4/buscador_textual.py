import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Input, Static

_nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])


def sin_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )


def buscar_frase(
    query: str, indice: dict[str, list]
) -> tuple[list[tuple[int, float, int]], set[str]]:
    """Busca una frase en el índice TF-IDF.

    Para cada token de contenido (sin stopwords) de la query, localiza su mejor
    clave en el índice (lema → forma literal → sin tildes) y acumula los scores
    por párrafo.

    Devuelve:
        resultados : lista de (párrafo_id, tfidf_total, n_términos_matched)
                     ordenada por (n_términos desc, tfidf desc)
        claves     : conjunto de claves encontradas, para resaltar en pantalla
    """
    doc = _nlp(query.strip())
    content_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
    if not content_tokens:
        return [], set()

    # Para cada token de contenido, buscar la mejor clave disponible en el índice
    claves: list[str] = []
    for token in content_tokens:
        for clave in [token.lemma_.lower(), token.text.lower(), sin_tildes(token.text.lower())]:
            if clave in indice:
                claves.append(clave)
                break

    if not claves:
        return [], set()

    # Acumular TF-IDF por párrafo y contar cuántas claves distintas matchean
    scores: dict[int, float] = defaultdict(float)
    hits: dict[int, set] = defaultdict(set)  # qué claves matchean en cada párrafo

    for clave in claves:
        for pid, tfidf in indice[clave]:
            scores[pid] += tfidf
            hits[pid].add(clave)

    resultados = [(pid, scores[pid], len(hits[pid])) for pid in scores]
    resultados.sort(key=lambda x: (x[2], x[1]), reverse=True)

    return resultados, set(claves)


def destacar(texto: str, claves: set[str]) -> str:
    """Resalta en amarillo las palabras cuyo lema o forma literal están en claves."""
    doc = _nlp(texto)
    partes = []
    ultimo = 0
    for token in doc:
        if token.is_alpha and (
            token.lemma_.lower() in claves or token.text.lower() in claves
        ):
            partes.append(texto[ultimo : token.idx])
            partes.append(f"[b yellow]{token.text}[/]")
            ultimo = token.idx + len(token.text)
    partes.append(texto[ultimo:])
    return "".join(partes)


def cargar_json(ruta: str) -> Any:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


class Buscador(App):
    BINDINGS = [
        Binding("left,p", "anterior", "← Anterior"),
        Binding("right,n", "siguiente", "→ Siguiente"),
        Binding("slash", "enfocar_busqueda", "/ Buscar"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #busqueda { margin: 1; }
    #estado { margin: 0 1; color: $text-muted; }
    #resultado {
        margin: 1;
        border: round #666;
        padding: 1;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, indice_path: str, parrafos_path: str) -> None:
        super().__init__()
        self.indice: dict[str, list] = cargar_json(indice_path)
        lista_parrafos: list[dict[str, Any]] = cargar_json(parrafos_path)
        self.parrafos: dict[int, dict[str, Any]] = {
            int(p["index"]): p for p in lista_parrafos if "index" in p
        }

        self._resultados: list[tuple[int, float, int]] = []  # (pid, tfidf, n_matched)
        self._claves: set[str] = set()
        self._pos: int = 0
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Buscar palabra o frase...", id="busqueda")
        yield Static("Introduce un término o frase.", id="estado")
        yield Static("", id="resultado")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._query = event.value.strip()
        self._resultados, self._claves = buscar_frase(self._query, self.indice)
        self._pos = 0
        self.query_one("#busqueda", Input).blur()
        self._mostrar()

    def action_enfocar_busqueda(self) -> None:
        self.query_one("#busqueda", Input).focus()

    def action_siguiente(self) -> None:
        if self._resultados and self._pos < len(self._resultados) - 1:
            self._pos += 1
            self._mostrar()

    def action_anterior(self) -> None:
        if self._resultados and self._pos > 0:
            self._pos -= 1
            self._mostrar()

    def _mostrar(self) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)

        if not self._query:
            estado.update("Introduce un término o frase.")
            contenedor.update("")
            return

        if not self._resultados:
            estado.update(f"Sin resultados para '{self._query}'.")
            contenedor.update("")
            return

        pid, tfidf, n_matched = self._resultados[self._pos]
        n_terminos = len(self._claves)
        estado.update(
            f"[b]{self._pos + 1}[/] / {len(self._resultados)}  "
            f"[dim]términos: {n_matched}/{n_terminos}  "
            f"TF-IDF: {tfidf:.4f}  (← → navegar)[/]"
        )

        parrafo = self.parrafos.get(pid)
        if parrafo is None:
            contenedor.update("[red]Párrafo no encontrado.[/]")
            return

        texto = parrafo.get("text", parrafo.get("texto", ""))
        parrafo_id = parrafo.get("index", parrafo.get("id"))
        headings = parrafo.get("headings", [])
        meta_str = " > ".join(headings) if headings else ""

        contenedor.update(
            f"[b cyan]Párrafo {parrafo_id}[/]\n[dim]{meta_str}[/]\n\n"
            + destacar(texto, self._claves)
        )


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python buscador_textual.py indice.json parrafos.json")
        sys.exit(1)

    app = Buscador(sys.argv[1], sys.argv[2])
    app.run()


if __name__ == "__main__":
    main()
