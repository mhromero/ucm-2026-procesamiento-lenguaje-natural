# Práctica 4 - El que lee mucho y anda mucho, ve mucho y sabe mucho
<code>**Procesamiento del Lenguaje Natural**</code>

## Integrantes

- María Romero Huertas
- Javier Martín Fuentes

## Índice

- [1. Descripción del proyecto](#1-descripción-del-proyecto)
- [2. Instrucciones de instalación y ejecución](#2-instrucciones-de-instalación-y-ejecución)
- [3. Preprocesamiento manual](#3-preprocesamiento-manual)
- [4. Modos de búsqueda](#4-modos-de-búsqueda)
- [5. Modelos](#5-modelos)
- [6. Estructura del repositorio](#6-estructura-del-repositorio)

## 1. Descripción del proyecto

Aplicación TUI para explorar un corpus en español basado en *Don Quijote de la Mancha*.  
El sistema combina recuperación clásica y semántica, y añade un modo RAG para responder consultas con referencias a fragmentos concretos (`Chunk N`) del texto.

## 2. Instrucciones de instalación y ejecución

### Requisitos previos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [ollama](https://ollama.com) instalado y en ejecución

### Instalación y arranque

```bash
# 1) Instalar dependencias
uv sync

# 2) Arrancar ollama (en otra terminal)
ollama serve

# 3) Descargar modelos base
ollama pull nomic-embed-text
ollama pull llama3.2

# 4) Ejecutar la aplicación
uv run fdi-pln-2611-p4
```

### Opciones CLI útiles

```bash
# Elegir modelo del modo RAG
uv run fdi-pln-2611-p4 --rag-model llama3.2

# Ajustar cuántos chunks usa RAG por búsqueda
uv run fdi-pln-2611-p4 --top-k-rag 5

# Ajustar cuántos resultados devuelve la búsqueda semántica
uv run fdi-pln-2611-p4 --top-k-semantica 20

# Forzar regeneración completa (índices + embeddings)
uv run fdi-pln-2611-p4 --regenerate
```

### Política de regeneración automática al arrancar

- Si existen índices y embeddings, no se recalcula nada.
- Si faltan embeddings, se recalculan embeddings.
- Si faltan índices, se recalculan índices y embeddings.
- `--regenerate` fuerza siempre la regeneración completa.

### Controles de la interfaz

| Tecla | Acción |
|---|---|
| `Ctrl+1` | Modo búsqueda clásica |
| `Ctrl+2` | Modo búsqueda semántica |
| `Ctrl+3` | Modo búsqueda RAG |
| `/` | Enfocar caja de búsqueda |
| `←` / `→` | Navegar resultados |
| `Enter` | Lanzar búsqueda |
| `q` | Salir |

También puedes seleccionar el modo de búsqueda haciendo click en los botones `Clásica`, `Semántica` y `RAG`.

## 3. Preprocesamiento manual

Antes de indexar, se limpió manualmente el HTML original para conservar solo el cuerpo narrativo de la obra.  
Se eliminaron elementos paratextuales (tasas, aprobaciones, privilegios, fe de erratas, dedicatorias, prólogos y composiciones poéticas iniciales/finales) para reducir ruido y mejorar la calidad de recuperación.

## 4. Modos de búsqueda

### 1) Búsqueda clásica (léxica)

- Tokenización y análisis con spaCy (`es_core_news_sm`).
- Filtrado de tokens no alfabéticos y stopwords.
- Coincidencia por lema, forma original y variante sin tildes.
- Ranking por cobertura de términos + suma de pesos TF-IDF por chunk.
- Presentación de score y términos emparejados en la TUI.

### 2) Búsqueda semántica (embeddings)

- Cada chunk textual se embebe con `nomic-embed-text`.
- Los vectores se normalizan y se comparan por similitud coseno.
- Se devuelve `top-k` por similitud (configurable con `--top-k-semantica`).
- En interfaz se muestra score de similitud y navegación por resultados.

### 3) Búsqueda RAG

- Recupera contexto combinando top resultados clásica + semántica.
- El parámetro `--top-k-rag` controla cuántos chunks de cada rama se usan.
- Construye prompt con fragmentos etiquetados por chunk.
- Consulta un LLM (modelo configurable con `--rag-model`).
- La respuesta se muestra con citas inline de chunks y bloque final de referencias.

### Generación de párrafos e índices

- El corpus se segmenta en frases y se agrupa con ventana deslizante:
  - `WINDOW = 5`
  - `STEP = 3` (con solapamiento).
- Se generan:
  - `parrafos_index.json` (chunks)
  - `vocabulario_index.json` (índice invertido TF-IDF).

### Generación de embeddings

- Se recorren los chunks en lotes (`BATCH = 32`).
- Se guarda:
  - `embeddings.npy` (vectores)
  - `embeddings_ids.npy` (IDs de chunks).
- La regeneración se integra en el flujo de arranque y en `--regenerate`.

## 5. Modelos

| Modelo | Uso | Instalación |
|---|---|---|
| `es_core_news_sm` (spaCy) | Lematización y segmentación de frases | Incluido en dependencias |
| `nomic-embed-text` (ollama) | Embeddings para búsqueda semántica | `ollama pull nomic-embed-text` |
| `llama3.2` (ollama) | Modelo por defecto para RAG | `ollama pull llama3.2` |

> El modelo de RAG es configurable en ejecución con `--rag-model`.

## 6. Estructura del repositorio

```text
fdi_pln_2611_p4/
├── data/
│   ├── 2000-h.htm             # Corpus fuente (Don Quijote, Project Gutenberg)
│   ├── parrafos_index.json    # Chunks extraídos del corpus
│   ├── vocabulario_index.json # Índice invertido con TF-IDF
│   ├── embeddings.npy         # Embeddings de chunks
│   └── embeddings_ids.npy     # IDs de chunks embebidos
├── main.py                    # CLI y orquestación de arranque/regeneración
├── buscador_textual.py        # Interfaz TUI (Textual)
├── busqueda_clasica.py        # Recuperación léxica (lemas + TF-IDF)
├── busqueda_semantica.py      # Recuperación por embeddings
├── busqueda_rag.py            # Construcción de contexto + llamada LLM
└── indexar_parrafos.py        # Chunking e indexación TF-IDF
```