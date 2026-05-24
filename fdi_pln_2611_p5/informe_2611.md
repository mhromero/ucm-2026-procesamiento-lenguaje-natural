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

Ejemplo de predicción sobre frases extraídas del corpus:

```
Entrada:
  alice was beginning to get very tired of sitting by her sister on the bank.
  the white rabbit ran close by her.
  alice started to her feet and ran across the field after the rabbit.
  the queen of hearts shouted off with her head.
  alice met the cheshire cat sitting on a branch.
  lewis carroll wrote alice's adventures in wonderland.
  the mad hatter and the march hare were having a tea party.
  alice found herself in the court of the king and queen of hearts.

Salida del modelo:
  PER  ali
  LOC  n
  PER  k
  PER  ali
  PER  a
  PER  ali
  PER  alice
  PER  a
  LOC  p
  PER  ali
```

Los resultados son claramente deficientes. El modelo detecta subtokens BPE aislados (`ali`, `a`, `k`, `n`, `p`) en lugar de los nombres completos, y no detecta ninguna entidad de tipo LOC correctamente (p.ej. `wonderland`). Las causas principales son:

<!-- Desarrollar estas razones en el informe final:
     - Dataset de anotaciones muy pequeño (pocas frases etiquetadas manualmente)
     - Desequilibrio extremo de clases: la mayoría de tokens son "o" → entity token accuracy 13.4%
     - La segmentación BPE rompe los nombres propios en subtokens: "alice" → "ali"+"ce", "queen" → "q"+"ueen", etc.
       El modelo aprende a etiquetar solo el primer subtoken (pi) pero no los siguientes (pc)
     - Fine-tuning con pocos datos sobre un backbone entrenado en dominio mixto (Alice+HP)
     - Las entidades de Alice son nombres propios muy específicos, poco frecuentes en el vocabulario BPE compartido con HP
-->

---

## 4. Generación de texto

Ejemplos de texto generado (100 tokens, temperatura=1.0):

**Prompt: `"alice"`**
```
aliceted astonishness .he come out an unpleasant run .it was a while you dont go !
hermione was learned .in my wand said to him .let me watch him .for questionly come
on he said .she could hear now said a frightened library looking at hermione for a rac
```

**Prompt: `"harry potter is"`**
```
harry potter isnt this evenes ?lill it you his hand ?something youre a put us when you
came to prival to the truth as i ?said dumbledore and george .he held up her hand hid
him .theyll find funny ive knowledge i had heard is in here wouldve y
```

**Prompt: `"The Queen said"`**
```
the queen saidseek above the imprisontains again he felt as though she were in these
three tall his ankle and things the spells was struggling all lumors on viere and worked
him through the platform beetle .spun diggory and sprieked by haywi
```

<!-- Observaciones a desarrollar:
     - El modelo genera texto sintácticamente parcialmente coherente pero semánticamente incoherente
     - Mezcla vocabulario de Alice y Harry Potter (hermione, dumbledore, wand, hogwarts) independientemente del prompt
       → consecuencia directa de entrenar con ambos corpus
     - Los nombres propios aparecen fragmentados o inventados ("aliceted", "imprisontains")
       → el tokenizador BPE de vocabulario pequeño (300 tokens) genera subwords que el modelo recombina mal
     - Puntuación irregular: espacios antes de signos, interrogaciones intercaladas
     - No hay coherencia temática sostenida más allá de 2-3 tokens
     - Limitación del modelo: tamaño (128d, 4 bloques), pocas épocas (10), corpus relativamente pequeño
-->

---

## 5. Conclusiones generales

<!-- Qué hemos aprendido de implementar un LLM desde cero: tokenización, atención, entrenamiento causal, fine-tuning NER. Limitaciones del sistema. Posibles mejoras. -->
