# Informe de la Práctica 5 — LM causal y NER (Alice in Wonderland)

**Asignatura**: Procesamiento del Lenguaje Natural — UCM 2025/26 
**Grupo**: 11  
**Integrantes**: María Romero Huertas, Javier Martín Fuentes

---

## 1. Descripción del sistema

El sistema implementa un LLM pequeño de principio a fin. El tokenizador es BPE (Byte Pair Encoding) con vocabulario de 300 tokens entrenado sobre el corpus. El modelo de lenguaje es un Transformer causal con 4 bloques de atención multi-head (4 cabezas, dimensión 128) y ventana de contexto de 128 tokens. Para NER se añade un cabezal de clasificación lineal sobre las representaciones del backbone, ajustado mediante fine-tuning con lr reducida en el backbone (factor 0.1). El esquema de etiquetado es BIO reducido: `o`, `pi`/`pc` (persona inicio/continuación) y `li`/`lc` (lugar inicio/continuación).

---

## 2. Exploración de hiperparámetros del LM causal

### 2.1 Grid search: learning rate × batch size

<!-- Referencia: informes/informe_grid_search.html -->

**Configuración explorada**: 9 combinaciones (lr ∈ {0.0001, 0.0003, 0.001} × batch ∈ {32, 64, 128}), 3 épocas por run.

**Mejor combinación**: lr=0.0003, batch=128 (test_loss=4.49)

**Observaciones**:

El `batch_size` es el factor dominante: aumentarlo de 32 a 128 reduce la test_loss en ~1.7 puntos de forma consistente en todas las lr. En cambio, con batch=128 las tres lr dan resultados casi idénticos (4.75, 4.49, 4.50), por lo que la lr exacta apenas importa cuando el gradiente es estable.

La brecha train/test es llamativa con batch=32 (ej. lr=0.001: train=2.83, test=6.51), lo que apunta a sobreajuste: con batches pequeños hay muchas más actualizaciones por época y el modelo tiende a memorizar. Con batch=128 la brecha se reduce a ~1.7 puntos.

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

Entrenar solo con Alice da val_loss=2.86 frente a 4.92 con Alice+4HP. El resultado es esperable: el corpus de validación es Alice, así que entrenar en el mismo dominio alinea perfectamente distribución de train y test. Añadir Harry Potter introduce vocabulario y estilo distintos que "diluyen" la representación de Alice; el modelo aprende más cosas pero predice peor el texto concreto de Alice. Esto no significa que el modelo entrenado solo en Alice generalice mejor, sino que está sobreajustado al dominio de evaluación.

#### Efecto del tamaño de ventana

Diferencia mínima: window_64 obtiene 4.886 frente a 4.921 con window_128. El contexto extra no aporta con este corpus y tamaño de modelo, lo que sugiere que las dependencias relevantes en Alice están dentro de las primeras 64 posiciones.

#### Efecto del vocabulario BPE

vocab_200 mejora ligeramente (4.706 vs 4.921). Con vocabulario más pequeño cada texto produce más tokens, lo que se traduce en más ventanas de entrenamiento (~180k más). Probablemente es ese volumen extra lo que explica la mejora, más que la granularidad del tokenizador en sí.

#### Efecto de la profundidad

Diferencia casi nula: depth_2 da 4.963 y depth_4 da 4.921. Con 5 épocas y este corpus, la capacidad extra de los 2 bloques adicionales no se aprovecha.

---

## 3. Entrenamiento NER

### 3.0 Elección del corpus de preentrenamiento

El backbone cumple dos funciones: es el modelo de lenguaje para generación de texto *y* el encoder sobre el que se apoya el fine-tuning NER. Para generación, más datos producen representaciones más ricas: entrenar con Alice + Harry Potter amplía el vocabulario efectivo del modelo y le da una base lingüística más sólida. Para NER, los datos de anotación son exclusivamente frases de Alice, por lo que un backbone entrenado únicamente en Alice habría estado mejor calibrado para ese dominio. Sin embargo, con solo 68 frases etiquetadas, el cuello de botella del NER es la cantidad de datos de anotación, no la calidad del backbone: la diferencia de rendimiento entre los dos corpus de preentrenamiento habría sido marginal. Por eso se optó por Alice + Harry Potter, priorizando el objetivo de generación sin esperar un coste significativo en NER.

<!-- Referencia: informes/ner_report.html -->

**Configuración**: 15 épocas de fine-tuning, batch=16, lr=0.0003, validación estratificada 80/20.  
**Mejor epoch**: 10 (criterio: macro F1 sin clase "o")  
**Entity token accuracy**: 13.4%

**Observaciones**:

La accuracy global (86.5%) es engañosa: el modelo aprende a predecir "o" casi siempre, lo que es correcto para la gran mayoría de tokens. La entity token accuracy del 13.4% refleja lo que realmente importa: el modelo apenas detecta entidades. El entrenamiento mejora hasta la época 10 medido por macro F1 sobre etiquetas de entidad, pero no llega a aprender a extender correctamente los spans con etiquetas de continuación (`pc`, `lc`).

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

Los resultados son claramente deficientes. El modelo detecta subtokens BPE aislados (`ali`, `a`, `k`, `n`, `p`) en lugar de los nombres completos, y no detecta ninguna entidad LOC correctamente (`wonderland` debería aparecer). Hay varias razones que se acumulan:

- **BPE rompe los nombres propios**: "alice" se tokeniza como `ali`+`ce`, "queen" como `q`+`ueen`, etc. El modelo puede aprender a marcar el primer subtoken como `pi` pero no aprende a encadenar los siguientes con `pc`, por lo que los spans quedan truncados a un solo subtoken.
- **Dataset de anotaciones pequeño**: pocas frases etiquetadas manualmente son insuficientes para aprender el etiquetado de continuación de forma fiable.
- **Desequilibrio extremo de clases**: la inmensa mayoría de tokens son `o`, lo que sesga el modelo hacia predecir `o` salvo señales muy claras.
- **Dominio mixto del backbone**: el tokenizador BPE fue entrenado con Alice+HP, por lo que los nombres propios de Alice no tienen representación propia clara en el vocabulario y se fragmentan más.

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

Los resultados son pobres pero no uniformemente malos. El modelo ignora completamente el tema del prompt: con `"alice"` genera texto de Harry Potter (hermione, wand) y con `"The Queen said"` produce referencias a Diggory y plataformas de Hogwarts. El dominio de HP domina simplemente porque su corpus es mucho mayor, y el modelo aprende a reproducir ese registro con más fluidez.

Lo que sí se nota gracias al volumen de HP es que algunas frases tienen estructura sintáctica parcialmente correcta: `"he felt as though she were in these three tall"` o `"it was a while you dont go"` tienen un ritmo reconocible aunque sin sentido. El modelo ha aprendido patrones de frase del inglés literario, pero no tiene mecanismo para mantener coherencia temática más allá de 2-3 tokens.

Los problemas más visibles son la puntuación errática (espacios antes de signos, interrogaciones intercaladas) y nombres propios fragmentados o inventados (`aliceted`, `imprisontains`, `evenes`), que son artefactos del vocabulario BPE de 300 tokens recomponiendo subwords en combinaciones que nunca vio en entrenamiento.

---

## 5. Conclusiones generales

Se eligió Harry Potter como corpus adicional por compartir características con Alice: ficción literaria en inglés, narración en tercera persona y público similar. El objetivo era ampliar el corpus manteniendo un registro coherente. Sin embargo, el experimento de corpus muestra que añadir HP empeora la val_loss en Alice (4.92 vs 2.86 entrenando solo en Alice), porque train y test dejan de coincidir en dominio. Para el modelo final de NER se optó igualmente por el backbone entrenado en Alice+HP, ya que un corpus mayor produce representaciones más ricas aunque la validación causal sea peor.

En NER el problema más importante es la falta de datos: con tan pocas frases anotadas manualmente, el modelo no tiene ejemplos suficientes para aprender el etiquetado de forma fiable. Hay que tener en cuenta además que el esquema interno usa etiquetas de inicio y continuación (`pi`, `pc`, `li`, `lc`), pero la salida al usuario agrupa esas etiquetas en tipos de entidad (`PER`, `LOC`): el modelo predice subtokens individuales con `pi` o `pc` y el postprocesado los une en un span con su tipo. Con pocos datos, el modelo aprende a emitir `pi` en algún subtoken pero raramente encadena `pc` correctamente, por lo que los spans quedan truncados a un solo subtoken.
