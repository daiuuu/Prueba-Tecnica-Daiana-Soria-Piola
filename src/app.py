"""
app.py
Servidor FastAPI — punto de entrada HTTP del asistente de soporte.
Expone un endpoint POST /consulta que n8n llama vía webhook.

Uso:
    uvicorn app:app --reload --port 8000
    python app.py
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    configurar_logging,
    verificar_entorno,
    validar_pregunta,
    construir_prompt_con_historial,
    respuesta_ok,
    respuesta_error,
    InputInvalido,
    VARS_OPCIONALES,
)
from search import buscar, formatear_contexto

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
PORT         = int(os.getenv("PORT", "8000"))
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K        = int(os.getenv("TOP_K", "4"))
MIN_SCORE    = float(os.getenv("MIN_SCORE", "0.30"))

configurar_logging(nivel=LOG_LEVEL, log_file="../logs/app.log")
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Asistente de Soporte Técnico",
    description="API REST para consultas de soporte basadas en documentación interna.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class ConsultaRequest(BaseModel):
    pregunta: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pregunta del usuario sobre soporte técnico",
        examples=["¿Cómo reinicio el servicio de autenticación?"],
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=10,
        description="Cantidad máxima de fragmentos a recuperar",
    )

    @field_validator("pregunta")
    @classmethod
    def pregunta_no_vacia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La pregunta no puede estar vacía.")
        return v


class ConsultaResponse(BaseModel):
    exito: bool
    respuesta: str | None
    error: str | None
    metadatos: dict
    timestamp: str


# ---------------------------------------------------------------------------
# Llamada al LLM
# ---------------------------------------------------------------------------

def llamar_llm(mensajes: list[dict]) -> str:
    """
    Envía los mensajes a OpenAI en formato chat (system + user)
    y retorna el texto de la respuesta.
    Maneja errores de API, timeout y rate limit con mensajes claros.
    """
    try:
        from openai import OpenAI, APIError, APITimeoutError, RateLimitError
    except ImportError:
        raise RuntimeError("openai no instalado. Ejecutá: pip install openai")

    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")

    cliente = OpenAI(api_key=OPENAI_KEY, timeout=30.0)
    max_intentos = 3

    for intento in range(1, max_intentos + 1):
        try:
            respuesta = cliente.chat.completions.create(
                model=OPENAI_MODEL,
                messages=mensajes,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return respuesta.choices[0].message.content.strip()

        except RateLimitError:
            wait = 2 ** intento
            logger.warning(f"Rate limit OpenAI. Reintentando en {wait}s... ({intento}/{max_intentos})")
            time.sleep(wait)

        except APITimeoutError:
            wait = 2 ** intento
            logger.warning(f"Timeout OpenAI. Reintentando en {wait}s... ({intento}/{max_intentos})")
            time.sleep(wait)

        except APIError as e:
            logger.error(f"Error de API OpenAI: {e}")
            raise RuntimeError(f"Error de API OpenAI: {e}") from e

    raise RuntimeError("No se pudo obtener respuesta de OpenAI después de 3 intentos.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check():
    return {"estado": "ok", "servicio": "Asistente de Soporte Técnico", "version": "1.0.0"}


@app.post(
    "/consulta",
    response_model=ConsultaResponse,
    tags=["Soporte"],
    summary="Consulta al asistente de soporte",
)
async def consulta(request: ConsultaRequest):
    """
    Recibe una pregunta de soporte y retorna una respuesta basada
    en la documentación técnica interna.

    Flujo:
      1. Valida el input
      2. Busca fragmentos relevantes en ChromaDB
      3. Construye los mensajes con el contexto (formato system/user)
      4. Llama a OpenAI
      5. Retorna la respuesta estructurada
    """
    tiempo_inicio = time.time()
    logger.info(f"Consulta recibida: '{request.pregunta[:80]}...'")

    # --- 1. Validar pregunta ---
    try:
        pregunta_limpia = validar_pregunta(request.pregunta)
    except InputInvalido as e:
        logger.warning(f"Input inválido: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=respuesta_error(str(e), codigo=400),
        )

    # --- 2. Buscar en ChromaDB ---
    resultado_busqueda = buscar(
        pregunta=pregunta_limpia,
        top_k=request.top_k,
        min_score=MIN_SCORE,
    )

    if resultado_busqueda["error"]:
        logger.error(f"Error en búsqueda: {resultado_busqueda['error']}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=respuesta_error(
                "El servicio de búsqueda no está disponible en este momento. "
                "Por favor intentá más tarde.",
                codigo=503,
            ),
        )

    # --- 3. Construir mensajes para el LLM ---
    contexto = formatear_contexto(resultado_busqueda["resultados"])
    mensajes = construir_prompt_con_historial(
        pregunta=pregunta_limpia,
        contexto=contexto,
        historial=[],
    )

    # --- 4. Llamar al LLM ---
    try:
        respuesta_llm = llamar_llm(mensajes)
    except RuntimeError as e:
        logger.error(f"Error en LLM: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=respuesta_error(
                "El servicio de inteligencia artificial no está disponible. "
                "Por favor intentá más tarde.",
                codigo=503,
            ),
        )

    # --- 5. Armar respuesta ---
    tiempo_total = round(time.time() - tiempo_inicio, 2)
    fuentes = list({r["source"] for r in resultado_busqueda["resultados"]})

    metadatos = {
        "encontrado_en_docs": resultado_busqueda["encontrado"],
        "fragmentos_usados":  len(resultado_busqueda["resultados"]),
        "fuentes":            fuentes,
        "tiempo_segundos":    tiempo_total,
        "modelo":             OPENAI_MODEL,
    }

    logger.info(
        f"Respuesta generada en {tiempo_total}s | "
        f"Fragmentos: {metadatos['fragmentos_usados']} | "
        f"Fuentes: {fuentes}"
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=respuesta_ok(respuesta_llm, metadatos),
    )


@app.exception_handler(Exception)
async def handler_global(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=respuesta_error(
            "Ocurrió un error interno. Por favor intentá más tarde.",
            codigo=500,
        ),
    )


# ---------------------------------------------------------------------------
# Arranque directo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Verificando entorno...")
    advertencias = verificar_entorno()

    if advertencias:
        logger.error("Faltan variables de entorno requeridas. Revisá tu .env")
        sys.exit(1)

    logger.info(f"Iniciando servidor en http://0.0.0.0:{PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)