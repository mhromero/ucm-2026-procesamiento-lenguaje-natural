from __future__ import annotations

from typing import Any, Literal

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Static

from .busqueda_clasica import buscar_frase, cargar_json, destacar
from .busqueda_rag import buscar_rag
from .busqueda_semantica import buscar_semantica, cargar_embeddings

ModoBusqueda = Literal["clasica", "semantica", "rag"]


class Buscador(App):
    BINDINGS = [
        Binding("left,p", "anterior", "← Anterior"),
        Binding("right,n", "siguiente", "→ Siguiente"),
        Binding("slash", "enfocar_busqueda", "/ Buscar"),
        Binding("ctrl+1", "modo_clasica", "^1 Clásica"),
        Binding("ctrl+2", "modo_semantica", "^2 Semántica"),
        Binding("ctrl+3", "modo_rag", "^3 RAG"),
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

    def __init__(
        self,
        indice_path: str,
        parrafos_path: str,
        embeddings_path: str | None,
        embeddings_ids_path: str | None,
    ) -> None:
        super().__init__()
        self.indice: dict[str, list] = cargar_json(indice_path)
        lista_parrafos: list[dict[str, Any]] = cargar_json(parrafos_path)
        self.parrafos: dict[int, dict[str, Any]] = {
            int(p["index"]): p for p in lista_parrafos if "index" in p
        }

        self._embeddings_disponibles = embeddings_path is not None
        if self._embeddings_disponibles:
            self.embeddings, self.embeddings_ids = cargar_embeddings(
                embeddings_path, embeddings_ids_path
            )
        else:
            self.embeddings = None
            self.embeddings_ids = None

        self._modo: ModoBusqueda = "clasica"
        self._resultados_clasica: list[tuple[int, float, int]] = []
        self._resultados_semantica: list[tuple[int, float]] = []
        self._claves: set[str] = set()
        self._pos: int = 0
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "Modos de búsqueda: [^1] Clásica  [^2] Semántica  [^3] RAG", id="modos"
        )
        yield Input(placeholder="Buscar palabra o frase...", id="busqueda")
        yield Static(
            "Modo actual: Clásica. Introduce un término o frase.",
            id="estado",
        )
        yield Static("", id="resultado")
        yield Footer()

    def _set_modo(self, modo: ModoBusqueda) -> None:
        self._modo = modo
        self._resultados_clasica = []
        self._resultados_semantica = []
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
            self._resultados_clasica, self._claves = buscar_frase(
                self._query, self.indice
            )
            self._resultados_semantica = []
        elif self._modo == "semantica":
            if not self._embeddings_disponibles:
                self._mostrar()
                return
            self._resultados_semantica = buscar_semantica(
                self._query, self.embeddings, self.embeddings_ids
            )
            self._resultados_clasica = []
            self._claves = set()
        elif self._modo == "rag":
            if not self._embeddings_disponibles:
                self._mostrar()
                return
            self._resultados_clasica, self._claves = buscar_frase(
                self._query, self.indice
            )
            self._resultados_semantica = buscar_semantica(
                self._query, self.embeddings, self.embeddings_ids
            )
            self._mostrar_rag_cargando()
            self._lanzar_rag(self._query)
            return

        self._mostrar()

    def _mostrar_rag_cargando(self) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        estado.update("Modo actual: RAG. Consultando al LLM...")
        contenedor.update("[dim]Generando respuesta, por favor espera...[/]")

    def _lanzar_rag(self, query: str) -> None:
        def _worker() -> None:
            respuesta = buscar_rag(
                query,
                self.parrafos,
                self._resultados_clasica,
                self._resultados_semantica,
            )
            self.call_from_thread(self._mostrar_respuesta_rag, respuesta)

        threading.Thread(target=_worker, daemon=True).start()

    def _mostrar_respuesta_rag(self, respuesta: str) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        estado.update(f"Modo actual: RAG. '{self._query}'")
        contenedor.update(respuesta)

    def _n_resultados(self) -> int:
        if self._modo == "clasica":
            return len(self._resultados_clasica)
        if self._modo == "semantica":
            return len(self._resultados_semantica)
        return 0

    def action_siguiente(self) -> None:
        if self._modo == "rag":
            return
        if self._pos < self._n_resultados() - 1:
            self._pos += 1
            self._mostrar()

    def action_anterior(self) -> None:
        if self._modo == "rag":
            return
        if self._pos > 0:
            self._pos -= 1
            self._mostrar()

    def _modo_label(self) -> str:
        if self._modo == "clasica":
            return "Clásica"
        if self._modo == "semantica":
            return "Semántica"
        return "RAG"

    def _render_chunk(self, chunk: dict[str, Any], texto_renderizado: str) -> str:
        headings = chunk.get("headings", [])
        meta_str = " > ".join(headings) if headings else ""
        cid = chunk.get("index", "?")

        sent_start = chunk.get("sent_start", 0)
        sent_end = chunk.get("sent_end", 0)
        n_total = chunk.get("n_sents_total", 1)
        prefijo_texto = "[dim]…[/] " if sent_start > 0 else ""
        sufijo_texto = " [dim]…[/]" if sent_end < n_total - 1 else ""

        return (
            f"[b cyan]Chunk {cid}[/]\n[dim]{meta_str}[/]\n\n"
            + prefijo_texto
            + texto_renderizado
            + sufijo_texto
        )

    def _mostrar(self) -> None:
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        prefijo = f"Modo actual: {self._modo_label()}"

        if not self._query:
            estado.update(f"{prefijo}. Introduce un término o frase.")
            contenedor.update("")
            return

        if self._modo in ("semantica", "rag") and not self._embeddings_disponibles:
            estado.update(f"{prefijo}.")
            contenedor.update(
                "[red]ollama no está disponible.[/]\n"
                "Asegúrate de que ollama está en ejecución y vuelve a arrancar la aplicación:\n"
                "  ollama serve\n"
                "  ollama pull nomic-embed-text"
            )
            return

        if self._modo == "rag":
            estado.update(f"{prefijo}. '{self._query}'")
            contenedor.update("[yellow]Introduce una pregunta y pulsa Enter.[/]")
            return

        n = self._n_resultados()
        if n == 0:
            estado.update(f"{prefijo}. Sin resultados para '{self._query}'.")
            contenedor.update("")
            return

        if self._modo == "clasica":
            pid, tfidf, n_matched = self._resultados_clasica[self._pos]
            n_terminos = len(self._claves)
            estado.update(
                f"{prefijo}. [b]{self._pos + 1}[/] / {n}  "
                f"[dim]términos: {n_matched}/{n_terminos}  "
                f"TF-IDF: {tfidf:.4f}  (← → navegar)[/]"
            )
            parrafo = self.parrafos.get(pid)
            if parrafo is None:
                contenedor.update("[red]Chunk no encontrado.[/]")
                return
            contenedor.update(
                self._render_chunk(
                    parrafo, destacar(parrafo.get("text", ""), self._claves)
                )
            )

        elif self._modo == "semantica":
            pid, score = self._resultados_semantica[self._pos]
            estado.update(
                f"{prefijo}. [b]{self._pos + 1}[/] / {n}  "
                f"[dim]similitud: {score:.4f}  (← → navegar)[/]"
            )
            parrafo = self.parrafos.get(pid)
            if parrafo is None:
                contenedor.update("[red]Chunk no encontrado.[/]")
                return
            contenedor.update(self._render_chunk(parrafo, parrafo.get("text", "")))
