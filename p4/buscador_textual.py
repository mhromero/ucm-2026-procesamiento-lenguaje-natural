import json
import sys
import unicodedata
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


def candidatos(texto: str) -> list[str]:
    """Devuelve posibles claves de búsqueda en orden de preferencia:
    lema spaCy, palabra tal cual, palabra sin tildes."""
    palabra = texto.strip().lower()
    doc = _nlp(palabra)
    lema = next((t.lemma_.lower() for t in doc if t.is_alpha), palabra)
    vistos = []
    for clave in [lema, palabra, sin_tildes(palabra)]:
        if clave not in vistos:
            vistos.append(clave)
    return vistos


def destacar(texto: str, lema: str) -> str:
    """Envuelve en markup Rich las palabras cuyo lema coincide con el buscado."""
    doc = _nlp(texto)
    partes = []
    ultimo = 0
    for token in doc:
        if token.is_alpha and (token.lemma_.lower() == lema or token.text.lower() == lema):
            partes.append(texto[ultimo : token.idx])
            partes.append(f"[b yellow]{token.text}[/]")
            ultimo = token.idx + len(token.text)
    partes.append(texto[ultimo:])
    return "".join(partes)


def cargar_json(ruta: str) -> Any:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


class Resultado(Static):
    def __init__(self, parrafo: dict[str, Any], lema: str) -> None:
        super().__init__()
        self.parrafo = parrafo
        self.lema = lema

    def render(self) -> str:
        texto = self.parrafo.get("text", self.parrafo.get("texto", ""))
        parrafo_id = self.parrafo.get("index", self.parrafo.get("id"))
        headings = self.parrafo.get("headings")
        if isinstance(headings, list) and headings:
            meta_str = " > ".join(str(h) for h in headings)
        else:
            meta = self.parrafo.get("metadatos", {})
            meta_str = " | ".join(f"{k}:{v}" for k, v in meta.items()) if meta else ""

        texto_destacado = destacar(texto, self.lema)
        return f"[b cyan]Párrafo {parrafo_id}[/]\n[dim]{meta_str}[/]\n\n{texto_destacado}"


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
        self.indice: dict[str, list[int]] = cargar_json(indice_path)
        lista_parrafos: list[dict[str, Any]] = cargar_json(parrafos_path)
        self.parrafos: dict[int, dict[str, Any]] = {}
        for p in lista_parrafos:
            pid = p.get("index", p.get("id"))
            if pid is None:
                continue
            self.parrafos[int(pid)] = p

        self._ids: list[int] = []
        self._pos: int = 0
        self._lema: str = ""
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Buscar palabra...", id="busqueda")
        yield Static("Introduce un término.", id="estado")
        yield Static("", id="resultado")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._query = event.value.strip()
        self._ids = []
        self._lema = ""
        for clave in candidatos(self._query):
            ids = self.indice.get(clave, [])
            if ids:
                self._ids = ids
                self._lema = clave
                break
        self._pos = 0
        self.query_one("#busqueda", Input).blur()
        self._mostrar()

    def action_enfocar_busqueda(self) -> None:
        self.query_one("#busqueda", Input).focus()

    def action_siguiente(self) -> None:
        if self._ids and self._pos < len(self._ids) - 1:
            self._pos += 1
            self._mostrar()

    def action_anterior(self) -> None:
        if self._ids and self._pos > 0:
            self._pos -= 1
            self._mostrar()

    def _mostrar(self) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)

        if not self._query:
            estado.update("Escribe algo.")
            contenedor.update("")
            return

        if not self._ids:
            estado.update(f"Sin resultados para '{self._query}'.")
            contenedor.update("")
            return

        pid, tfidf = self._ids[self._pos]
        estado.update(
            f"Clave: [b]'{self._lema}'[/] · {self._pos + 1} / {len(self._ids)}  "
            f"[dim]TF-IDF: {tfidf:.4f}  (← → para navegar)[/]"
        )
        parrafo = self.parrafos.get(pid)
        if parrafo is None:
            contenedor.update("[red]Párrafo no encontrado.[/]")
            return

        texto = parrafo.get("text", parrafo.get("texto", ""))
        parrafo_id = parrafo.get("index", parrafo.get("id"))
        headings = parrafo.get("headings")
        if isinstance(headings, list) and headings:
            meta_str = " > ".join(str(h) for h in headings)
        else:
            meta = parrafo.get("metadatos", {})
            meta_str = " | ".join(f"{k}:{v}" for k, v in meta.items()) if meta else ""

        texto_destacado = destacar(texto, self._lema)
        contenedor.update(
            f"[b cyan]Párrafo {parrafo_id}[/]\n[dim]{meta_str}[/]\n\n{texto_destacado}"
        )


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python buscador_textual.py indice.json parrafos.json")
        sys.exit(1)

    app = Buscador(sys.argv[1], sys.argv[2])
    app.run()


if __name__ == "__main__":
    main()
