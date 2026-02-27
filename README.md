# Práctica 1 - Los agentes de Butler
<code>**Procesamiento del Lenguaje Natural**</code>

## Índice
- [Descripción breve](#descripción-breve)
- [Integrantes](#integrantes)
- [Requisitos](#requisitos)
- [Instrucciones de ejecución](#instrucciones-de-ejecución)
- [Funcionamiento y lógica del bot](#funcionamiento-y-lógica-del-bot)
- [Estructura del código](#estructura-del-código)
- [Pruebas con cartas de prueba](#pruebas-con-cartas-de-prueba)
- [Calidad y empaquetado](#calidad-y-empaquetado)

## Descripción breve
Bot de negociación de recursos entre agentes para Butler.  
El agente analiza cartas con LLM (Ollama), decide si aceptar/rechazar ofertas según su estado (necesidades, excedentes e inventario), envía paquetes y confirma intercambios por carta.

## Integrantes
- Javier Martín Fuentes
- María Romero Huertas

## Requisitos
- Python 3.12
- `uv`
- Butler (`fdi-pln-butler`)
- Ollama (necesario para el flujo con LLM)

## Instrucciones de ejecución

### 1) Lanzar Butler
Arranca el servidor Butler con el buzón de cartas habilitado:

```bash
fdi-pln-butler server --buzon
```

Para pruebas en local (modo monopuesto), añade la bandera `--monopuesto`:

```bash
fdi-pln-butler server --buzon --monopuesto
```

> **Nota:** En modo monopuesto el servidor asigna el alias automáticamente; no es necesario pasarlo por variables de entorno salvo que quieras forzar uno concreto.

### 2) Lanzar Ollama
El bot usa Ollama para interpretar cartas con el LLM. Asegúrate de que el servidor está en ejecución:

```bash
ollama serve
```

### 3) Verificar o descargar el modelo
Comprueba que tienes disponible un modelo compatible (por defecto: `qwen3-vl:8b`):

```bash
ollama list
ollama pull qwen3-vl:8b   # o el modelo que uses en config.json
```

Si usas otro modelo, configura la variable `FDI_PLN__MODEL` o la opción `--model`.

### 4) Ejecutar el agente
Desde el directorio del proyecto:

```bash
cd ~/fdi-pln-2611
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

El puerto `7719` es el que Butler usa por defecto. Si Butler corre en otra URL, ajusta `FDI_PLN__BUTLER_ADDRESS` en consecuencia.

### 5) Ejecutar dos agentes (modo monopuesto)
Para que sea modo monopuesto, Butler debe estar lanzado con el flag `--monopuesto` (véase paso 1). Ejecuta cada agente en una terminal distinta, asignando alias distintos:

**Terminal A:**
```bash
cd ~/fdi-pln-2611
FDI_PLN__ALIAS=ag001 FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

**Terminal B:**
```bash
cd ~/fdi-pln-2611
FDI_PLN__ALIAS=ag002 FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

Si Butler corre en otra URL, ajusta `FDI_PLN__BUTLER_ADDRESS` en consecuencia.

---

### Variables de entorno
La configuración sigue la prioridad: **variables de entorno > argumentos CLI > config.json**. Estas son las variables principales:

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `FDI_PLN__BUTLER_ADDRESS` | URL base de Butler | `http://127.0.0.1:7719` |
| `FDI_PLN__ALIAS` | Alias del agente (modo monopuesto) | `ag001` |
| `FDI_PLN__MODO_MONOPUESTO` | Activar modo monopuesto | `true` / `1` / `yes` |
| `FDI_PLN__MODEL` | Modelo de Ollama | `qwen3-vl:8b` |
| `FDI_PLN__OLLAMA_URL` | Endpoint de la API de Ollama | `http://localhost:11434/api/generate` |
| `FDI_PLN__GOLD_RESOURCE_NAME` | Nombre del recurso oro | `oro` |
| `FDI_PLN__MAILBOX_ENDPOINT` | Endpoint del buzón | `/buzon` |
| `FDI_PLN__LETTER_ENDPOINT` | Endpoint de cartas | `/carta` |
| `FDI_PLN__PACKAGE_ENDPOINT` | Endpoint de paquetes | `/paquete` |
| `FDI_PLN__LETTERS_BEFORE_REBROADCAST` | Cartas procesadas antes de reenviar ofertas | `5` |
| `FDI_PLN__OFFERS_PER_PERSON` | Ofertas aleatorias por destinatario | `2` |
| `FDI_PLN__REMITENTE_SISTEMA` | Remitente de cartas del sistema (se ignoran) | `Sistema` |
| `FDI_PLN__MAX_INTENTOS_OFERTA` | Intentos máximos al analizar una oferta con Ollama | `3` |

### Opciones CLI
Puedes sobreescribir cualquier valor con argumentos en la línea de comandos:

```bash
uv run fdi-pln-2611-p1 --help
```

**Opciones principales:**

| Opción | Variable de entorno | Descripción |
|--------|---------------------|-------------|
| `--api-base TEXT` | `FDI_PLN__BUTLER_ADDRESS` | URL base de Butler (ej. `http://127.0.0.1:7719`) |
| `--ollama-url TEXT` | `FDI_PLN__OLLAMA_URL` | URL de la API de Ollama |
| `--model TEXT` | `FDI_PLN__MODEL` | Modelo de Ollama (ej. `mistral:7b`) |
| `--alias TEXT` | `FDI_PLN__ALIAS` | Alias del agente |
| `--modo-monopuesto` / `--no-modo-monopuesto` | `FDI_PLN__MODO_MONOPUESTO` | Activar/desactivar modo monopuesto |
| `--letters-before-rebroadcast INT` | `FDI_PLN__LETTERS_BEFORE_REBROADCAST` | Cartas antes de reenviar ofertas |
| `--offers-per-person INT` | `FDI_PLN__OFFERS_PER_PERSON` | Ofertas por persona |
| `--gold-resource-name TEXT` | `FDI_PLN__GOLD_RESOURCE_NAME` | Nombre del recurso oro |
| `--remitente-sistema TEXT` | `FDI_PLN__REMITENTE_SISTEMA` | Remitente de cartas del sistema |
| `--max-intentos-oferta INT` | `FDI_PLN__MAX_INTENTOS_OFERTA` | Intentos máximos al analizar ofertas |

**Ejemplo con CLI:**
```bash
uv run fdi-pln-2611-p1 --api-base http://127.0.0.1:7719 --alias ag001 --model mistral:7b
```

También puedes modificar los valores por defecto en `config.json`, que se aplican cuando no se especifica una variable de entorno ni una opción CLI correspondiente.

## Funcionamiento y lógica del bot

### Diseño de prompts y uso del modelo

El bot utiliza **dos prompts principales** que incorporan el estado del agente y están pensados para obtener respuestas estructuradas y fiables con Ollama. Con el modelo por defecto (`qwen3-vl:8b`) por lo general devuelve JSONs válidos a la primera.

#### 1. Prompt de interpretación de cartas (`parse_letter`)

**Objetivo:** Extraer de cada carta entrante una estructura JSON con tipo (`oferta` | `confirmacion` | `otro`), recursos ofrecidos, solicitados y recibidos.

**Datos inyectados en el prompt:**
- Se incluyen dinámicamente `OFRECEMOS` (surplus) y `NECESITAMOS` (needs) para que el modelo entienda el marco de negociación.
- Se incluye la carta completa (`letter_data`) sobre la que debe trabajar.

**Robustez ante respuestas imperfectas:**
- **Salida JSON pura:** Se pide explícitamente "Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional" porque los LLM suelen añadir explicaciones o markdown.
- **Extracción robusta:** Si la respuesta incluye texto extra, el código busca y extrae el primer objeto JSON válido (`_parse_json_response`).
- **Normalización de cantidades:** Si la carta no contiene números explícitos, se normaliza cada recurso detectado a 1 unidad para evitar que el LLM invente cantidades (`_normalize_amounts_if_ambiguous`).
- **Esquema flexible:** Se soportan variantes de formato (`{"recurso":"queso"}` → `{"queso":1}`) porque distintos modelos estructuran el JSON de formas distintas.

#### 2. Prompt de decisión de ofertas (`analyze_offer`)

**Objetivo:** Decidir si aceptar o rechazar una oferta según el estado actual del agente.

**Reglas según el estado del agente:** El prompt incluye **tres variantes de reglas** según el estado:

| Estado del agente | Reglas inyectadas |
|-------------------|-------------------|
| Objetivo cumplido | Maximizar oro: aceptar si nos ofrecen oro a cambio de surplus; rechazar si no ofrecen oro o piden lo que no tenemos. |
| Solo tenemos oro (`gold_only`) | Aceptar si nos ofrecen 1 unidad de cualquier recurso a cambio de 1 oro. |
| Negociación normal | Aceptar si nos dan lo que necesitamos, no piden lo que necesitamos, podemos dar lo que piden, y las cantidades son razonables; incluye excepción para ofertas con oro y surplus. |

**Robustez ante respuestas imperfectas:**
- **Acotación de decisión:** Tras la respuesta del LLM, se aplica `_sanitize_decision_against_received_offer` para que la decisión solo incluya recursos y cantidades presentes en la oferta original; se evita que el modelo "invente" recursos o cantidades mayores.
- **Reintentos:** Se configura `MAX_INTENTOS_OFERTA` para reintentar ante errores de conexión o JSON inválido.
- **Formato estricto:** Se exige nuevamente JSON válido sin texto adicional.

---

### Flujo general

El agente arranca solicitando el estado inicial a Butler (`/info`): obtiene su alias, inventario, objetivo y buzón. Luego consulta la lista de jugadores (`/gente`), descarta su propio alias y calcula:

- **needs**: recursos que faltan para completar el objetivo.
- **surplus**: recursos sobrantes (excluyendo oro).

Con eso envía ofertas iniciales al resto de agentes y entra en bucle de buzón:

1. Lee las cartas del buzón y las ordena por fecha (antiguas primero).
2. Interpreta cada carta con el LLM, decide si acepta o rechaza y, si corresponde, envía paquete y confirmación.
3. Marca la carta como procesada y consulta de nuevo el buzón cada 5 segundos.

### Tipos de oferta que envía

Según el estado del agente, se usan distintos formatos de carta:

| Situación | Tipo de oferta | Ejemplo |
|-----------|----------------|---------|
| Necesita recursos y tiene surplus | `propuesta intercambio` | Intercambio 1:1: "Te ofrezco 1 arroz y tú me das 1 trigo." |
| Objetivo cumplido, maximizando oro | Surplus → oro | "Te ofrezco 1 arroz y tú me das 1 oro." |
| Solo puede ofrecer oro | Oro → recurso | "Te ofrezco 1 oro a cambio de 1 trigo." |

### Procesado de cartas entrantes

Para cada carta recibida:

1. **Interpretación:** Se llama al LLM para obtener un JSON estructurado. Si la respuesta incluye texto extra, se extrae el primer objeto JSON válido.
2. **Normalización:** Se convierten variantes de esquema al formato interno (p. ej. `{"recurso":"queso"}` → `{"queso":1}`). Si no hay cantidades explícitas en la carta, se asume 1 por recurso para evitar inflar cifras.
3. **Evaluación de ofertas:** Si la carta es una oferta, se pasa por `analyze_offer` y la decisión se acota a recursos y cantidades presentes en la oferta original.
4. **Confirmaciones:** Si la carta es confirmación, se comprueba si debemos devolver recursos según lo pactado.

### Criterios de aceptación y rechazo

El agente rechaza ofertas incompletas (sin `oferta` o sin `pide`) y no envía nunca recursos que necesita para su objetivo ni más de lo que tiene en inventario. El oro tiene reglas propias según la fase del juego. Cuando acepta una oferta, envía primero el paquete vía `/paquete/{dest}` y luego la carta de confirmación.

### Reenvío de ofertas (rebroadcast)

Se vuelven a enviar ofertas tras procesar `LETTERS_BEFORE_REBROADCAST` cartas. Si el buzón está vacío durante varias revisiones consecutivas (polling cada 5 s), también se hace rebroadcast para mantener la visibilidad de las ofertas.

## Estructura del código
```text
src/
├── app.py            # Flujo principal del bot y bucle de buzón
├── api.py            # Cliente HTTP Butler (/info, /gente, /carta, /mail, /paquete)
├── agent.py          # Parseo/decisión con LLM + normalización/validación de salida
├── trader.py         # Reglas de aceptación/rechazo y ejecución de intercambios
├── letters.py        # Plantillas de cartas y broadcast de ofertas
├── game_state.py     # Estado local: alias, inventario, objetivo, needs/surplus
├── ollama_client.py  # Cliente HTTP de Ollama
├── logs.py           # Salida por consola
├── config.py         # Configuración (env > CLI > config.json)
└── config.json       # Valores por defecto del proyecto
└── send_test_letter.sh  # Script para inyectar cartas de prueba
```

## Pruebas con cartas de prueba
Puedes inyectar cartas manuales sin levantar un tercer agente con:
`src/send_test_letter.sh`

Ayuda:
```bash
src/send_test_letter.sh --help
```

Ejemplo básico:
```bash
src/send_test_letter.sh \
  --dest ag002 \
  --from ag001 \
  --subject "propuesta intercambio" \
  --body "Te ofrezo 1 arroz y tu me das 1 trigo."
```

## Calidad y empaquetado
Formato/calidad:
```bash
uv format --check
```

Construcción de wheel:
```bash
uv build
```
