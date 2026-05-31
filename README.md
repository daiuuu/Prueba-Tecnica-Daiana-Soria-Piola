# Asistente Automatizado de Soporte Técnico

Sistema RAG (Retrieval-Augmented Generation) de soporte técnico que responde consultas usando documentación interna. El usuario hace una pregunta, el sistema busca los fragmentos más relevantes en la base de datos vectorial y genera una respuesta basada exclusivamente en esa documentación.

Construido con Python, FastAPI, ChromaDB, OpenAI y n8n.

---

## Cómo funciona

El flujo completo, desde la pregunta hasta la respuesta, pasa por estas etapas:

```
Usuario
  │
  ▼
n8n Webhook (POST /webhook/soporte)
  │  valida y reenvía el input
  ▼
FastAPI (POST /consulta)
  │
  ├─► 1. Valida la pregunta (longitud, contenido, caracteres)
  │
  ├─► 2. Genera un embedding de la pregunta (OpenAI)
  │
  ├─► 3. Busca en ChromaDB los fragmentos más similares
  │         └─► filtra por score mínimo de relevancia
  │
  ├─► 4. Construye el prompt con los fragmentos como contexto
  │
  └─► 5. Llama a OpenAI y retorna la respuesta estructurada
```

El sistema nunca inventa información: si no encuentra nada relevante en la documentación, informa al usuario y sugiere contactar soporte directo.

---

## Tecnologías utilizadas

- **Python 3.11+** — procesamiento de documentos y servidor API
- **FastAPI** — servidor REST con documentación interactiva automática
- **ChromaDB** — base de datos vectorial para búsqueda semántica local
- **OpenAI API** — embeddings (`text-embedding-3-small`) y generación de respuestas (`gpt-3.5-turbo`)
- **n8n** — orquestación del flujo webhook → API → respuesta

---

## Estructura del proyecto

```
asistente-automatizado-soporte/
│
├── docs/                          # Documentación técnica fuente
│   ├── errores_conexion_campos.txt
│   ├── troubleshooting_catalogo.pdf
│   ├── error_credenciales.md
│   └── errores_frecuentes_minecatalog.json
│
├── data/
│   └── chroma_db/                 # Base de datos vectorial (generada al ingestar)
│
├── src/
│   ├── loaders.py                 # Parsers por tipo de archivo (.txt .pdf .md .json)
│   ├── ingest.py                  # Pipeline de ingesta: carga → chunks → embeddings → índice
│   ├── search.py                  # Búsqueda semántica en ChromaDB
│   ├── app.py                     # Servidor FastAPI (endpoint REST)
│   ├── utils.py                   # Validaciones, prompts, logging, manejo de errores
│   └── prompts/
│       └── support_prompt.txt     # Instrucciones del sistema para el LLM
│
├── n8n/
│   └── workflow.json              # Workflow exportado de n8n
│
├── tests/
│   ├── test_search.py             # Tests de validación de preguntas y formateo de contexto
│   └── test_ingest.py             # Tests del pipeline de chunking
│
├── logs/                          # Logs de ejecución (generado automáticamente)
├── .env                           # Variables de entorno (no se sube al repo)
├── .env.example                   # Plantilla de variables de entorno
├── requirements.txt               # Dependencias Python
└── .gitignore
```

---

## Decisiones de diseño

### Por qué chunking por secciones (structure-based) y no por tamaño fijo

La mayoría de los sistemas RAG divide los documentos en fragmentos de N caracteres con solapamiento (overlap). Este sistema usa un enfoque distinto: divide por estructura semántica del documento.

**El problema con chunking por tamaño fijo:**

Imaginá este párrafo en la documentación:

```
Error ERR-DB-003: Timeout de conexión con la base de datos.
Causas: el servidor de base de datos no responde, configuración
```

Si el chunk termina justo ahí, el fragmento que llega al LLM está cortado a mitad de oración. El modelo recibe información incompleta y puede generar una respuesta incorrecta o confusa.

**La solución: respetar la estructura del documento**

Cada tipo de archivo se divide de forma distinta según cómo está organizado naturalmente:

- **`.txt`** — se divide por párrafos (bloques separados por línea en blanco doble). Cada párrafo tiende a describir un error o procedimiento completo.
- **`.md`** — se divide por secciones (encabezados `#`, `##`, `###`). Cada sección tiene un tema propio.
- **`.pdf`** — se divide por páginas. El PDF de troubleshooting está organizado por página temática.
- **`.json`** — cada objeto del array es un fragmento. En el JSON de errores frecuentes, cada entrada ya tiene toda la información de un error: título, causas, solución, palabras clave.

Como control de seguridad, si algún fragmento supera 1200 caracteres (configurable con `MAX_CHUNK_CHARS`), se subdivide por párrafos internos antes de indexar. Pero esto actúa como techo de emergencia, no como método principal de división.

**El resultado:** cada fragmento que llega al LLM es una unidad de información completa y coherente, lo que mejora significativamente la calidad de las respuestas.

### Por qué ChromaDB local

ChromaDB corre en disco, sin servidor externo ni cuenta de pago. Para un sistema de soporte técnico con documentación interna de tamaño acotado, es más que suficiente y elimina una dependencia externa.

### Por qué el prompt está separado en un archivo `.txt`

El prompt del sistema (`src/prompts/support_prompt.txt`) define el rol, las reglas y el tono del asistente. Tenerlo en un archivo separado en vez de hardcodeado en el código permite modificarlo sin tocar Python, lo que es útil cuando quien lo ajusta no es quien mantiene el código.

---

## Requisitos previos

- Python 3.11 o superior
- Node.js 18 o superior
- n8n instalado globalmente (`npm install -g n8n`)
- Cuenta en OpenAI con API key activa y créditos disponibles

---

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd asistente-automatizado-soporte
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Abrí el archivo `.env` y completá los valores. Como mínimo necesitás:

```env
OPENAI_API_KEY=sk-...tu-clave-aquí...
```

El resto de variables tiene valores por defecto razonables. Podés dejarlos como están para empezar.

### 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 4. Ingestar la documentación

Este paso lee los archivos de `/docs`, los divide en fragmentos, genera embeddings con OpenAI y los guarda en ChromaDB. Solo necesitás correrlo una vez, o cada vez que cambies los documentos.

```bash
cd src
python ingest.py
```

Deberías ver una salida similar a:

```
09:00:01 [INFO] ingest: INICIO DE INGESTA (structure-based chunking)
09:00:01 [INFO] ingest: Paso 1/4: Cargando documentos...
09:00:02 [INFO] ingest: Paso 2/4: Aplicando corte de seguridad...
09:00:02 [INFO] ingest: Paso 3/4: Generando embeddings con OpenAI...
09:00:05 [INFO] ingest: Paso 4/4: Indexando en ChromaDB...
09:00:05 [INFO] ingest: INGESTA COMPLETADA: 24 chunks indexados
```

### 5. Levantar el servidor API

```bash
cd src
uvicorn app:app --reload --port 8000
```

El servidor queda disponible en `http://localhost:8000`. Podés verificar que funciona abriendo `http://localhost:8000/docs` en el navegador — FastAPI muestra la documentación interactiva automáticamente.

### 6. Levantar n8n

En una terminal separada:

```bash
n8n start
```

n8n queda disponible en `http://localhost:5678`.

### 7. Importar el workflow en n8n

1. Abrí `http://localhost:5678`
2. Creá un usuario si es la primera vez
3. Menú izquierdo → **Workflows** → botón **+** → **Import from file**
4. Seleccioná el archivo `n8n/workflow.json`
5. Activá el workflow con el toggle **Active** (arriba a la derecha)

---

## Uso

### Probar desde terminal con curl

```bash
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"¿Cómo reinicio el servicio de autenticación?\"}"
```

También podés llamar directamente a la API sin pasar por n8n:

```bash
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"¿Cómo reinicio el servicio de autenticación?\"}"
```

### Ejemplos de preguntas

```bash
# Error de autenticación
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"No puedo iniciar sesión, me dice credenciales incorrectas\"}"

# Error de base de datos
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"El sistema devuelve error de conexión con la base de datos\"}"

# Pregunta sin información disponible
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"¿Cómo exporto reportes en PDF?\"}"
```

### Formato de respuesta

```json
{
  "exito": true,
  "respuesta": "El error de credenciales incorrectas (ERR-AUTH-001) puede tener estas causas...",
  "error": null,
  "metadatos": {
    "fuentes": ["error_credenciales.md"],
    "fragmentos_usados": 2,
    "encontrado_en_docs": true,
    "tiempo_segundos": 1.83,
    "modelo": "gpt-3.5-turbo"
  },
  "timestamp": "2026-01-01T09:00:00.000000"
}
```

---

## Tests

El proyecto tiene tests unitarios para los módulos de búsqueda e ingesta. No requieren conexión a OpenAI ni a ChromaDB — prueban la lógica pura de validación, chunking y formateo.

### Correr todos los tests

Desde la raíz del proyecto:

```bash
pytest tests/ -v
```

### Correr un archivo específico

```bash
pytest tests/test_search.py -v
pytest tests/test_ingest.py -v
```

### Qué cubre cada archivo

`tests/test_search.py` prueba dos funciones de `search.py`:

- `_validar_pregunta` — que rechace inputs vacíos, demasiado cortos, demasiado largos, no-strings, y que normalice espacios correctamente. También verifica los valores de borde exactos (3 y 2000 caracteres).
- `formatear_contexto` — que arme bien el texto de contexto para el LLM: orden de fragmentos, respeto del límite de caracteres, truncado con indicador, fallback cuando falta la clave `source`.

`tests/test_ingest.py` prueba `chunkear_fragmento` de `ingest.py`:

- Que un fragmento vacío devuelva lista vacía.
- Que un fragmento corto no se divida.
- Que un fragmento largo se divida por párrafos (`\n\n`), manteniendo cada párrafo completo.
- Que los metadatos (`source`, `type`) se preserven en todos los chunks resultantes.
- Que un párrafo único muy largo (sin `\n\n` internos) no se pierda — se incluye como chunk único.

### Salida esperada

```
tests/test_search.py::test_pregunta_valida PASSED
tests/test_search.py::test_pregunta_con_espacios PASSED
tests/test_search.py::test_pregunta_vacia PASSED
tests/test_search.py::test_pregunta_solo_espacios PASSED
tests/test_search.py::test_pregunta_demasiado_corta PASSED
tests/test_search.py::test_pregunta_demasiado_larga PASSED
tests/test_search.py::test_pregunta_no_es_string PASSED
tests/test_search.py::test_pregunta_none PASSED
tests/test_search.py::test_pregunta_exactamente_limite PASSED
tests/test_search.py::test_pregunta_exactamente_minimo PASSED
tests/test_search.py::test_formatear_contexto_vacio PASSED
tests/test_search.py::test_formatear_contexto_un_resultado PASSED
tests/test_search.py::test_formatear_contexto_varios_resultados PASSED
tests/test_search.py::test_formatear_contexto_respeta_limite PASSED
tests/test_search.py::test_formatear_contexto_orden_fragmentos PASSED
tests/test_search.py::test_formatear_contexto_sin_source PASSED
tests/test_ingest.py::test_texto_vacio PASSED
tests/test_ingest.py::test_texto_corto_no_se_divide PASSED
tests/test_ingest.py::test_texto_largo_se_divide_por_parrafos PASSED
tests/test_ingest.py::test_cada_chunk_es_parrafo_completo PASSED
tests/test_ingest.py::test_se_preservan_metadatos PASSED
tests/test_ingest.py::test_no_genera_chunks_vacios PASSED
tests/test_ingest.py::test_parrafo_unico_muy_largo_no_se_pierde PASSED
```

---

## Opciones avanzadas de ingesta

```bash
# Usar un directorio de docs diferente
python ingest.py --docs ../mis-docs

# Cambiar el límite de caracteres del corte de seguridad
python ingest.py --max-chars 800

# Limpiar la base de datos antes de reingestar
python ingest.py --limpiar

# Combinado
python ingest.py --docs ../mis-docs --max-chars 800 --limpiar
```

### Probar la búsqueda directamente sin levantar la API

```bash
cd src
python search.py "¿Cómo reinicio el servicio de autenticación?"
python search.py "error 502" --top-k 5
python search.py "no puedo acceder al dashboard" --verbose
```

---

## Agregar nueva documentación

1. Copiá el archivo a la carpeta `/docs`. Formatos soportados: `.txt`, `.pdf`, `.md`, `.json`.
2. Volvé a correr la ingesta con el flag `--limpiar` para reindexar todo desde cero:

```bash
cd src
python ingest.py --limpiar
```

No hace falta reiniciar el servidor API — ChromaDB se lee en cada consulta.

---

## Solución de problemas

**`python ingest.py` no encuentra documentos**
Verificá que los archivos estén en la carpeta `/docs` y que las extensiones sean `.txt`, `.pdf`, `.md` o `.json`.

**Error `OPENAI_API_KEY no configurada`**
Asegurate de que el archivo `.env` existe en la raíz del proyecto y tiene la clave correcta. No el `.env.example`, el `.env`.

**n8n no puede conectarse a la API**
Verificá que `uvicorn` esté corriendo en el puerto 8000 antes de activar el workflow. Podés confirmarlo abriendo `http://localhost:8000` en el navegador.

**ChromaDB vacío después de ingestar**
Revisá los logs de `ingest.py`. Si los embeddings fallaron, verificá que la API key de OpenAI sea válida y tenga créditos disponibles.

**Las respuestas no son relevantes o el sistema dice que no encontró información**
Probá bajar el valor de `MIN_SCORE` en el `.env` (por ejemplo a `0.20`). Si el problema persiste, revisá que la ingesta se haya completado correctamente y que los documentos en `/docs` contengan la información buscada.