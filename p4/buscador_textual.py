from __future__ import annotations

import sys
from typing import Any, Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Static

from busqueda_clasica import buscar_frase, cargar_json, destacar

ModoBusqueda = Literal["clasica", "semantica", "rag"]


class Buscador(App):
    BINDINGS = [
        Binding("left,p", "anterior", "← Anterior"),
        Binding("right,n", "siguiente", "→ Siguiente"),
        Binding("slash", "enfocar_busqueda", "/ Buscar"),
        Binding("1", "modo_clasica", "1 Clásica"),
        Binding("2", "modo_semantica", "2 Semántica"),
        Binding("3", "modo_rag", "3 RAG"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #modos { margin: 0 1; color: $text-muted; }
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

        self._modo: ModoBusqueda = "clasica"
        self._resultados: list[tuple[int, float, int]] = []
        self._claves: set[str] = set()
        self._pos: int = 0
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Modos de búsqueda: [1] Clásica  [2] Semántica  [3] RAG", id="modos")
        yield Input(placeholder="Buscar palabra o frase...", id="busqueda")
        yield Static(
            "Modo actual: Clásica (1/2/3 para cambiar). Introduce un término o frase.",
            id="estado",
        )
        yield Static("", id="resultado")
        yield Footer()

    def _set_modo(self, modo: ModoBusqueda) -> None:
        self._modo = modo

        self._resultados = []
        self._claves = set()
        self._pos = 0
        self._mostrar()

    def action_modo_clasica(self) -> None:
        self._set_modo("clasica")

    def action_modo_semantica(self) -> None:
        self._set_modo("semantica")

    def action_modo_rag(self) -> None:
        self._set_modo("rag")

    def action_enfocar_busqueda(self) -> None:
        self.query_one("#busqueda", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._query = event.value.strip()
        self._pos = 0
        self.query_one("#busqueda", Input).blur()

        if self._modo == "clasica":
            self._resultados, self._claves = buscar_frase(self._query, self.indice)
        else:
            self._resultados = []
            self._claves = set()

        self._mostrar()

    def action_siguiente(self) -> None:
        if self._modo != "clasica":
            return
        if self._resultados and self._pos < len(self._resultados) - 1:
            self._pos += 1
            self._mostrar()

    def action_anterior(self) -> None:
        if self._modo != "clasica":
            return
        if self._resultados and self._pos > 0:
            self._pos -= 1
            self._mostrar()

    def _modo_label(self) -> str:
        if self._modo == "clasica":
            return "Clásica"
        if self._modo == "semantica":
            return "Semántica"
        return "RAG"

    def _mostrar(self) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        prefijo = f"Modo actual: {self._modo_label()} (1/2/3 para cambiar, / para buscar)"

        if not self._query:
            estado.update(f"{prefijo}. Introduce un término o frase.")
            contenedor.update("")
            return

        if self._modo != "clasica":
            estado.update(f"{prefijo}. '{self._query}'")
            contenedor.update(
                "[yellow]Este modo está pendiente de implementar.[/]\n"
                "Por ahora solo está implementada la búsqueda clásica."
            )
            return

        if not self._resultados:
            estado.update(f"{prefijo}. Sin resultados para '{self._query}'.")
            contenedor.update("")
            return

        pid, tfidf, n_matched = self._resultados[self._pos]
        n_terminos = len(self._claves)
        estado.update(
            f"{prefijo}. [b]{self._pos + 1}[/] / {len(self._resultados)}  "
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
