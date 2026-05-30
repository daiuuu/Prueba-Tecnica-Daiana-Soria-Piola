# Asistente Automatizado de Soporte Técnico

Sistema de soporte técnico automatizado que responde consultas utilizando documentación interna. Construido con Python, FastAPI, ChromaDB, OpenAI y n8n.

---

## Tecnologías utilizadas

- **Python 3.11+** — procesamiento de documentos y servidor API
- **FastAPI** — servidor REST
- **ChromaDB** — base de datos vectorial para búsqueda semántica
- **OpenAI API** — embeddings y generación de respuestas
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
│       └── support_prompt.txt     # Instrucciones del asistente
│
├── n8n/
│   └── workflow.json              # Workflow exportado de n8n
│
├── logs/                          # Logs de ejecución (generado automáticamente)
│
├── .env                           # Variables de entorno (no se sube al repo)
├── .env.example                   # Plantilla de variables de entorno
├── requirements.txt               # Dependencias Python
└── .gitignore
```

---

## Requisitos previos

- Python 3.11 o superior
- Node.js 18 o superior
- n8n instalado globalmente (`npm install -g n8n`)
- Cuenta en OpenAI con API key activa

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

Abrí el archivo `.env` y completá los valores, como mínimo:

```
OPENAI_API_KEY=sk-...tu-clave-aquí...
```

### 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 4. Ingestar la documentación

Este paso lee los archivos de `/docs`, genera embeddings y los guarda en ChromaDB. Solo necesitás correrlo una vez, o cada vez que cambies los documentos.

```bash
cd src
python ingest.py
```

Deberías ver una salida similar a:

```
09:00:01 [INFO] ingest: INICIO DE INGESTA
09:00:01 [INFO] ingest: Paso 1/4: Cargando documentos...
09:00:02 [INFO] ingest: Paso 2/4: Generando chunks...
09:00:02 [INFO] ingest: Paso 3/4: Generando embeddings con OpenAI...
09:00:05 [INFO] ingest: Paso 4/4: Indexando en ChromaDB...
09:00:05 [INFO] ingest: INGESTA COMPLETADA: 24 chunks indexados
```

### 5. Levantar el servidor API

```bash
cd src
uvicorn app:app --reload --port 8000
```

El servidor queda disponible en `http://localhost:8000`.
Podés verificar que funciona abriendo `http://localhost:8000/docs` en el navegador — FastAPI muestra la documentación interactiva automáticamente.

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

### Ejemplos de preguntas

```bash
# Error de autenticación
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"No puedo iniciar sesión, me dice credenciales incorrectas\"}"

# Error de base de datos
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"El sistema devuelve error de conexión con la base de datos\"}"

# Pregunta sin información disponible
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"¿Cómo exporto reportes en PDF?\"}"
```

### Formato de respuesta

```json
{
  "success": true,
  "respuesta": "El error de credenciales incorrectas (ERR-AUTH-001) puede tener estas causas...",
  "error": null,
  "metadata": {
    "fuentes": ["error_credenciales.md"],
    "fragmentos_usados": 2,
    "encontrado_en_docs": true,
    "tiempo_segundos": 1.83
  },
  "timestamp": "2026-01-01T09:00:00.000000"
}
```

---

## Opciones avanzadas de ingesta

```bash
# Cambiar tamaño de chunks
python ingest.py --chunk-size 500 --overlap 100

# Usar un directorio de docs diferente
python ingest.py --docs ../mis-docs

# Limpiar la base de datos antes de reingestar
python ingest.py --limpiar
```

## Probar la búsqueda directamente

```bash
cd src
python search.py "¿Cómo reinicio el servicio de autenticación?"
python search.py "error 502" --top-k 5
python search.py "no puedo acceder al dashboard" --verbose
```

---

## Flujo interno del sistema

```
Usuario
  │
  ▼
n8n Webhook (POST /webhook/soporte)
  │  valida input
  ▼
FastAPI (POST /consulta)
  │
  ├─► ChromaDB — búsqueda semántica por embedding
  │       └─► fragmentos relevantes de la documentación
  │
  ├─► Construcción del prompt con contexto
  │
  └─► OpenAI API — genera respuesta
          │
          ▼
      Respuesta JSON al usuario
```

---

## Solución de problemas

**`python ingest.py` no encuentra documentos**
Verificá que los archivos estén en la carpeta `/docs` y que las extensiones sean `.txt`, `.pdf`, `.md` o `.json`.

**Error `OPENAI_API_KEY no configurada`**
Asegurate de que el archivo `.env` existe en la raíz del proyecto y tiene la clave correcta.

**n8n no puede conectarse a la API**
Verificá que `uvicorn` esté corriendo en el puerto 8000 antes de activar el workflow. Podés confirmarlo abriendo `http://localhost:8000` en el navegador.

**ChromaDB vacío después de ingestar**
Revisá los logs de `ingest.py`. Si los embeddings fallaron, verificá que la API key de OpenAI sea válida y tenga créditos disponibles.