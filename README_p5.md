# Práctica 5 — LM causal y NER (Alice in Wonderland)

## Integrantes

- María Romero Huertas
- Javier Martín Fuentes

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Instalación

```bash
uv sync
```

## Ejecutable

```bash
uv run fdi-pln-2611-p5 --help
```

## Comandos

| Comando | Descripción |
|---------|-------------|
| `train-causal` | Entrena el LM causal (`--weights`, opcional `--grid-search`) |
| `train-ner` | Entrena NER sobre el backbone (`--weights`, `--causal-weights`) |
| `generate` | Genera texto (`--weights`, `--prompt`) |
| `ner` | Lista entidades en un fichero (`--weights`, `FICHERO`) |
| `merge-annotations` | Fusiona JSON de anotadores y calcula κ de Cohen |
| `merge-etiquetados` | Fusiona `data/etiquetados/` (p1/parte1 + p2/parte2) |
| `annotation-report` | Informe HTML con métricas y gráficos |

### Ejemplos

```bash
# Exploración de hiperparámetros (9 combinaciones) + entrenamiento final
uv run fdi-pln-2611-p5 train-causal --weights p5_causal_2611.pth --grid-search

# Fusionar etiquetados + informe (recomendado tras completar anotación)
uv run fdi-pln-2611-p5 merge-etiquetados
uv run fdi-pln-2611-p5 annotation-report --skip-merge

# O en un solo paso (vuelve a fusionar):
uv run fdi-pln-2611-p5 annotation-report

# Entrenar NER (requiere anotaciones fusionadas con etiquetas)
uv run fdi-pln-2611-p5 train-ner --weights p5_ner_2611.pth --causal-weights p5_causal_2611.pth

# Inferencia
uv run fdi-pln-2611-p5 generate --weights p5_causal_2611.pth --prompt "Alice"
uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth fdi_pln_2611_p5/data/alice_in_wonderland.txt
```

## Etiquetado

Hay dos granularidades de plantilla:

| Nivel | Comando | Salida | Qué se etiqueta |
|-------|---------|--------|-----------------|
| **Palabra** | `generate-templates-word` | `data/alice_jsons_palabra/` | Cada palabra, espacio o signo |
| **Token BPE** | `generate-templates-token` | `data/alice_jsons_token/` | Cada subpalabra del tokenizador |

```bash
# Por palabra (no requiere BPE previo)
uv run fdi-pln-2611-p5 generate-templates-word

# Por token BPE (requiere data/bpe_tokenizer.json, p. ej. tras train-causal)
uv run fdi-pln-2611-p5 generate-templates-token
```

Scripts equivalentes: `generar_jsons_palabra.py` y `generar_jsons_token.py`.

**Nuevo lote (6 frases/JSON, ≥20 palabras)** — no toca `data/alice_jsons/`:

```bash
uv run fdi-pln-2611-p5 generate-templates-6frases
# o: uv run python fdi_pln_2611_p5/generar_jsons_6frases.py
```

Salida: `data/alice_jsons_6frases/` (13 JSON, 39 frases, cada frase en 2 anotadores).

1. Completar el campo `valor` en cada unidad (`clave`):
   - `o` — fuera de entidad
   - `pi` / `pc` — persona (inicio / continuación)
   - `li` / `lc` — lugar (inicio / continuación)
2. Fusionar (indica el directorio que hayáis etiquetado):

```bash
uv run fdi-pln-2611-p5 merge-annotations \
  --json-dir fdi_pln_2611_p5/data/alice_jsons_palabra \
  --assignments fdi_pln_2611_p5/data/alice_jsons_palabra/asignaciones.json
```

El entrenamiento NER convierte cualquier granularidad a etiquetas por carácter y luego a BPE.

Solo `json_08.json` (en `alice_jsons/`) está anotado de momento; conviene completar más JSON antes del entrenamiento NER.

## Entrega (campus)

- `fdi_pln_2611_p5-1.0-py3-none-any.whl` → `uv build`
- `p5_causal_2611.pth`, `p5_ner_2611.pth`
- Informe opcional con resultados de `--grid-search` (`data/grid_search_results.json`)

## Formato

```bash
uv format --check
```
