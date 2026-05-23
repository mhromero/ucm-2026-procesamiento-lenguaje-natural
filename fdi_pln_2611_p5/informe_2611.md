# Informe de la Práctica 5 — LM causal y NER (Alice in Wonderland)

**Asignatura**: Procesamiento del Lenguaje Natural — UCM 2025/26 
**Grupo**: 11  
**Integrantes**: María Romero Huertas, Javier Martín Fuentes

---

## 1. Descripción del sistema

<!-- Breve descripción de la arquitectura implementada: tokenizador BPE, Transformer causal, cabezal NER -->

---

## 2. Exploración de hiperparámetros del LM causal

### 2.1 Grid search: learning rate × batch size

<!-- Referencia: informes/informe_grid_search.html -->

**Configuración explorada**: 9 combinaciones (lr ∈ {0.0001, 0.0003, 0.001} × batch ∈ {32, 64, 128}), 3 épocas por run.

**Mejor combinación**: lr=0.0003, batch=128 (test_loss=4.49)

**Observaciones**:

<!-- ¿Qué patrón se observa con batch_size? ¿Y con lr? ¿Hay sobreajuste en alguna combinación? -->

### 2.2 Experimentos de arquitectura y corpus

<!-- Referencia: informes/informe_experimentos.html -->

Resumen de resultados (5 épocas, validación siempre en Alice):

| Experimento | val_loss |
|-------------|----------|
| Solo Alice (corpus_alice) | 2.856 |
| Alice + 4 libros HP (corpus_alice_hp4) | 4.921 |
| Ventana 64 (window_64) | 4.886 |
| Ventana 128 — referencia (window_128) | 4.921 |
| Vocabulario BPE = 200 (vocab_200) | 4.706 |
| Vocabulario BPE = 300 — referencia (vocab_300) | 4.921 |
| 2 bloques Transformer (depth_2) | 4.963 |
| 4 bloques — referencia (depth_4) | 4.921 |

#### Efecto del corpus

<!-- Solo Alice da val_loss mucho menor (2.86 vs 4.92). ¿Qué implica esto? ¿Overfitting de dominio? -->

#### Efecto del tamaño de ventana

<!-- window_64 (4.886) vs window_128 (4.921): diferencia pequeña. ¿Qué sugiere sobre el contexto necesario? -->

#### Efecto del vocabulario BPE

<!-- vocab_200 (4.706) vs vocab_300 (4.921): ¿por qué un vocabulario menor mejora la loss? Relación con el número de tokens/ventanas. -->

#### Efecto de la profundidad

<!-- depth_2 (4.963) vs depth_4 (4.921): diferencia marginal. ¿Qué dice sobre la capacidad del modelo para este corpus? -->

---

## 3. Entrenamiento NER

<!-- Referencia: informes/ner_report.html -->

**Configuración**: 15 épocas de fine-tuning, batch=16, lr=0.0003, validación estratificada 80/20.  
**Mejor epoch**: 10 (criterio: macro F1 sin clase "o")  
**Entity token accuracy**: 13.4%

**Observaciones**:

<!-- ¿Qué etiquetas se detectan mejor? ¿Personas o lugares? ¿Qué errores son más frecuentes según la matriz de confusión? -->

### 3.1 Evaluación cualitativa y limitaciones

<!-- Evaluar el NER con ejemplos concretos (comando: uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth --text "...") -->
<!-- Explicar por qué la entity token accuracy es tan baja (13.4%):
     - Dataset de anotaciones muy pequeño (pocas frases etiquetadas manualmente)
     - Desequilibrio extremo de clases: la mayoría de tokens son "o"
     - El modelo tiende a predecir "o" casi siempre → alta accuracy global pero pésima detección de entidades
     - La segmentación BPE dificulta la continuidad de etiquetas (pi/pc): un token "a" de "alice" puede recibir "pi" pero el siguiente subtoken "lice" queda como "o"
     - Fine-tuning sobre un backbone de dominio mixto (Alice+HP) con pocos datos NER
     - Las entidades de Alice son muy específicas (nombres propios poco frecuentes en el vocabulario BPE)
-->

---

## 4. Generación de texto

<!-- Ejemplos de texto generado con distintos prompts. Observaciones sobre coherencia, vocabulario, repeticiones. -->

---

## 5. Conclusiones generales

<!-- Qué hemos aprendido de implementar un LLM desde cero: tokenización, atención, entrenamiento causal, fine-tuning NER. Limitaciones del sistema. Posibles mejoras. -->
