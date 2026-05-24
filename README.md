<div align="center">

# LM causal y NER

**Procesamiento del Lenguaje Natural** — *Alice in Wonderland* + Harry Potter

<p>
  <img src="https://img.shields.io/badge/Práctica-5-4338ca?style=flat-square" alt="Práctica 5" />
  <img src="https://img.shields.io/badge/Grupo-11-6366f1?style=flat-square" alt="Grupo 11" />
  <img src="https://img.shields.io/badge/UCM-2025%2F26-312e81?style=flat-square" alt="UCM 2025/26" />
</p>

<p>
  <strong>María Romero Huertas</strong> · <strong>Javier Martín Fuentes</strong>
</p>

<p>
  Transformer causal entrenado desde cero (BPE + atención) y cabezal NER fine-tuned sobre el mismo backbone.
</p>

</div>

## Índice

- [Resultados](#resultados)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Inicio rápido](#inicio-rápido)
- [Comandos CLI](#comandos-cli)
  - [Entrenamiento](#entrenamiento)
  - [Inferencia](#inferencia)
  - [Anotación](#anotación)
- [Informes por comando](#informes-por-comando)
- [Reproducibilidad](#reproducibilidad)
- [Estructura del paquete](#estructura-del-paquete)
- [Corpus y datos](#corpus-y-datos)
- [Hiperparámetros](#hiperparámetros)
- [Etiquetas NER](#etiquetas-ner)

---

## Resultados

Memoria del trabajo con descripción del sistema, grid search, experimentos de arquitectura, métricas NER, ejemplos de generación y conclusiones:

**[Informe general (HTML)](fdi_pln_2611_p5/informes/INFORME_GENERAL.html)**

Abre el fichero en el navegador o desde el explorador de archivos del IDE. Los informes técnicos generados automáticamente por cada comando se listan en [Informes por comando](#informes-por-comando).

---

## Requisitos e instalación

| Requisito | Versión |
|-----------|---------|
| Python | 3.12+ |
| Gestor | [uv](https://docs.astral.sh/uv/) |

```bash
git clone <repo-url>
cd fdi-pln2611
uv sync
```

Instalación desde wheel (entrega):

```bash
pip install fdi_pln_2611_p5-1.0-py3-none-any.whl
```

Todos los comandos del proyecto se invocan con:

```bash
uv run fdi-pln-2611-p5 <comando> [opciones]
```

La configuración por defecto está en `fdi_pln_2611_p5/config.json`. Puedes pasar otra con `--config <ruta>`.

---

## Inicio rápido

Pipeline típico de principio a fin:

```bash
# 1. Tokenizador BPE (también se entrena dentro de train-causal)
uv run fdi-pln-2611-p5 train-tokenizer

# 2. LM causal (por defecto: p5_causal_2611.pth)
uv run fdi-pln-2611-p5 train-causal

# 3. Fusionar anotaciones manuales
uv run fdi-pln-2611-p5 merge-annotations

# 4. NER sobre el backbone causal (por defecto: p5_ner_2611.pth + p5_causal_2611.pth)
uv run fdi-pln-2611-p5 train-ner

# 5. Inferencia
uv run fdi-pln-2611-p5 inference-generate          # prompt de ejemplo (config)
uv run fdi-pln-2611-p5 inference-ner               # fichero de ejemplo (sample_ner_test.txt)
```

---

## Comandos CLI

### Entrenamiento

| Comando | Descripción |
|---------|-------------|
| `train-tokenizer` | Entrena BPE y guarda `data/bpe_tokenizer.json` |
| `train-causal` | Entrena el LM causal (pesos por defecto: `p5_causal_2611.pth`) |
| `train-causal --weights <ruta.pth>` | Igual con ruta de salida personalizada |
| `train-causal --tokenizer <bpe.json>` | Usa un BPE preentrenado (p. ej. de un experimento) |
| `train-causal --grid-search` | Grid search 3×3 (lr × batch) y entrenamiento final con la mejor combinación |
| `train-ner` | Fine-tuning NER (por defecto: `p5_ner_2611.pth`, backbone `p5_causal_2611.pth`) |
| `train-ner --weights <ruta.pth> --causal-weights <ruta.pth>` | Rutas de salida y backbone personalizadas |
| `train-ner … --tokenizer <bpe.json>` | NER con BPE explícito (por defecto: el del checkpoint causal) |
| `run-experiments` | Ocho ablaciones (corpus, ventana, vocab, profundidad) + informe comparativo |

### Inferencia

| Comando | Descripción |
|---------|-------------|
| `inference-generate` | Generación autoregresiva (pesos por defecto: `p5_causal_2611.pth`) |
| `inference-generate --tokenizer <bpe.json>` | Generación con BPE explícito |
| `inference-ner <fichero.txt>` | Entidades en un fichero UTF-8 (por defecto: `sample_ner_test.txt`) |
| `inference-ner --tokenizer <bpe.json>` | NER con BPE explícito |

### Anotación

| Comando | Descripción |
|---------|-------------|
| `prepare-annotations` | Plantillas JSON vacías (por defecto: 5 frases/JSON, 2 anotadores/frase) |
| `merge-annotations` | Fusiona `json_XX.json` → `data/annotations/merged.json` (asignaciones automáticas) |

**Opciones habituales:** `--config`, `--tokenizer`, rutas de pesos, `--grid-search`, `--max-new-tokens`, `--temperature`.

---

## Informes por comando

| Comando | Artefacto principal | Informe HTML |
|---------|---------------------|--------------|
| — | Memoria completa | [`informes/INFORME_GENERAL.html`](fdi_pln_2611_p5/informes/INFORME_GENERAL.html) |
| `run-experiments` | `data/experiment_results.json` + `data/experiments/<id>/` | [`informes/informe_experimentos.html`](fdi_pln_2611_p5/informes/informe_experimentos.html) |
| `train-causal --grid-search` | `data/grid_search_results.json` | [`informes/informe_grid_search.html`](fdi_pln_2611_p5/informes/informe_grid_search.html) |
| `train-ner` | Pesos `.pth` + CSV de épocas | [`informes/ner_report.html`](fdi_pln_2611_p5/informes/ner_report.html) |
| `merge-annotations` | `data/annotations/merged.json` | Solo resumen en consola (κ, nº frases) |

El informe de acuerdo entre anotadores ([`informes/informe_etiquetado.html`](fdi_pln_2611_p5/informes/informe_etiquetado.html)) se genera con `generate_annotation_report()` en código; no se invoca aún desde `merge-annotations`.

---

## Reproducibilidad

Cada entrenamiento guarda junto a los artefactos:

| Archivo | Contenido |
|---------|-----------|
| `config.json` | Configuración **efectiva** del run (con overrides aplicados) |
| `training_config.json` | Metadatos + config completa (`run_type`, pérdidas, rutas, etc.) |
| `experiment_config.json` | Solo en `data/experiments/<id>/` (suite de 8 experimentos) |

Si pesos y config del paquete comparten directorio (`p5_*.pth` en la raíz), el informe de config usa el nombre `{stem}_training_config.json` para no sobrescribir el `config.json` principal del proyecto.

---

## Estructura del paquete

```
fdi_pln_2611_p5/
├── main.py                 # CLI (Typer)
├── config.py / config.json
├── model/
│   ├── lm_causal/          # Transformer, BPE, checkpoints causales
│   └── ner/                # Etiquetas, modelo NER, decodificación
├── corpus/                 # Carga y concatenación de textos
├── inference/              # Generación y extracción de entidades
├── annotations/            # Plantillas, merge, dataset NER, informes de etiquetado
├── training/               # Causal, NER, grid search, experimentos, run_config
├── data/
│   ├── corpus/             # Alice, Looking-Glass, Harry Potter (extra)
│   ├── annotations/        # merged.json
│   └── experiments/        # Caché por experimento (pesos, BPE, config)
└── informes/               # HTML generados y memoria
```

---

## Corpus y datos

| Uso | Fuente |
|-----|--------|
| Entrenamiento causal | Alice + *Through the Looking-Glass* + hasta 4 libros HP (`extra_max_books`) |
| Validación causal | *Alice in Wonderland* |
| NER | Frases de Alice anotadas manualmente (`merged.json`) |

Textos en `fdi_pln_2611_p5/data/corpus/`. Anotaciones en `data/alice_jsons/` (o la carpeta que indiques con `--json-dir`).

---

## Hiperparámetros

Valores por defecto en `config.json` (resumen):

<details>
<summary><strong>Tokenizador y arquitectura</strong></summary>

| Parámetro | Valor |
|-----------|-------|
| `vocab_size` | 300 |
| `d_model` | 128 |
| `n_blocks` | 4 |
| `n_heads` | 4 |
| `window_size` | 128 |
| `dropout` | 0.1 |

</details>

<details>
<summary><strong>Entrenamiento causal</strong></summary>

| Parámetro | Valor |
|-----------|-------|
| `epochs` | 10 |
| `batch_size` | 64 |
| `learning_rate` | 3e-4 |
| `seed` | 42 |

**Grid search:** lr ∈ {1e-4, 3e-4, 1e-3}, batch ∈ {32, 64, 128}, 3 épocas por combinación.

</details>

<details>
<summary><strong>Entrenamiento NER</strong></summary>

| Parámetro | Valor |
|-----------|-------|
| `epochs` | 15 |
| `batch_size` | 16 |
| `learning_rate` | 3e-4 |
| `val_ratio` | 0.2 (split estratificado) |
| `backbone_lr_factor` | 0.1 |
| `best_metric` | `macro_f1_non_o` |

</details>

---

## Etiquetas NER

Esquema BIO reducido (persona / lugar):

| Etiqueta | Significado |
|----------|-------------|
| `o` | Fuera de entidad |
| `pi` / `pc` | Inicio / continuación de **persona** |
| `li` / `lc` | Inicio / continuación de **lugar** |
