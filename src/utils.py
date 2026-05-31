"""
utils.py
Utilidades compartidas por el resto del sistema:
  - Configuración centralizada de logging
  - Validación y sanitización de inputs
  - Manejo y formateo de errores
  - Construcción del prompt para el LLM
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configurar_logging(nivel: str = "INFO", log_file: str | None = None) -> None:
    """
    Configura el sistema de logging para toda la aplicación.
    Si se especifica log_file, escribe también en disco.
    """
    nivel_numerico = getattr(logging, nivel.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=nivel_numerico,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Validación de inputs
# ---------------------------------------------------------------------------

class InputInvalido(Exception):
    """Se lanza cuando el input del usuario no puede procesarse."""
    pass


def validar_pregunta(pregunta: str) -> str:
    """
    Valida y normaliza la pregunta del usuario.
    Retorna la pregunta limpia o lanza InputInvalido.

    Controles:
      - No vacía
      - No solo espacios o símbolos
      - Longitud mínima y máxima
      - Sin caracteres de control peligrosos
    """
    if not isinstance(pregunta, str):
        raise InputInvalido("El campo pregunta debe ser texto.")

    # Normalizar espacios
    pregunta = pregunta.strip()
    pregunta = re.sub(r"[ \t]{2,}", " ", pregunta)
    pregunta = re.sub(r"\n{3,}", "\n", pregunta)

    if not pregunta:
        raise InputInvalido("La pregunta no puede estar vacía.")

    # Eliminar caracteres de control (excepto salto de línea y tab)
    pregunta = re.sub(r"[^\S\n\t ]", "", pregunta)

    if len(pregunta) < 3:
        raise InputInvalido(
            "La pregunta es demasiado corta. "
            "Ingresá al menos 3 caracteres."
        )

    if len(pregunta) > 2000:
        raise InputInvalido(
            "La pregunta supera el límite de 2000 caracteres. "
            "Por favor resumila."
        )

    # Detectar si es solo símbolos sin contenido semántico
    solo_simbolos = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9\s]", "", pregunta)
    if not solo_simbolos.strip():
        raise InputInvalido(
            "La pregunta debe contener palabras o números, "
            "no solo símbolos o caracteres especiales."
        )

    return pregunta


def validar_top_k(valor) -> int:
    """Valida el parámetro top_k y lo normaliza."""
    try:
        valor = int(valor)
    except (TypeError, ValueError):
        return 4
    return max(1, min(valor, 10))


# ---------------------------------------------------------------------------
# Construcción del prompt
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _cargar_prompt(nombre_archivo: str) -> str:
    """
    Lee un archivo de prompt desde src/prompts/.
    Lanza FileNotFoundError con mensaje claro si no existe.
    """
    ruta = _PROMPTS_DIR / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo de prompt no encontrado: {ruta}\n"
            "Verificá que la carpeta src/prompts/ exista y contenga el archivo."
        )
    return ruta.read_text(encoding="utf-8")


# Se cargan una sola vez al importar el módulo
_PROMPT_CON_CONTEXTO = _cargar_prompt("support_prompt.txt")

_PROMPT_SIN_CONTEXTO = """Sos un asistente de soporte técnico especializado.
Tu única fuente de información es la documentación técnica interna de la empresa.

El usuario realizó la siguiente consulta, pero no se encontró información
relevante en la documentación disponible.

PREGUNTA DEL USUARIO:
{pregunta}

Informale al usuario de forma amable que:
1. No encontraste información relacionada con su consulta en la documentación.
2. Le recomendés contactar al soporte técnico directamente.
3. El contacto disponible es: soporte.minecatalog@empresa.com

Respondé en español, de forma breve y cordial."""


def construir_prompt(pregunta: str, contexto: str) -> str:
    """
    Construye el prompt final para enviar al LLM.
    Usa support_prompt.txt (con placeholders {contexto} y {pregunta})
    si hay fragmentos relevantes, o la plantilla de "sin información" si no.
    """
    if contexto and contexto.strip():
        return _PROMPT_CON_CONTEXTO.format(
            contexto=contexto.strip(),
            pregunta=pregunta.strip(),
        )
    else:
        return _PROMPT_SIN_CONTEXTO.format(pregunta=pregunta.strip())


def construir_prompt_con_historial(
    pregunta: str,
    contexto: str,
    historial: list[dict],
) -> list[dict]:
    """
    Construye la lista de mensajes para la API de OpenAI con formato
    de chat (system + historial + user actual).
    Útil si en el futuro querés soportar conversaciones multi-turno.

    historial: lista de dicts [{"role": "user"|"assistant", "content": "..."}]
    """
    system_content = (
        "Sos un asistente de soporte técnico especializado. "
        "Respondé ÚNICAMENTE basándote en la documentación proporcionada. "
        "Si la información no está disponible, indicalo claramente. "
        "Respondé siempre en español."
    )

    if contexto and contexto.strip():
        system_content += f"\n\nDOCUMENTACIÓN RELEVANTE:\n{contexto.strip()}"

    mensajes = [{"role": "system", "content": system_content}]
    mensajes.extend(historial[-6:])
    mensajes.append({"role": "user", "content": pregunta.strip()})

    return mensajes


# ---------------------------------------------------------------------------
# Formateo de respuestas de error para la API
# ---------------------------------------------------------------------------

def respuesta_error(mensaje: str, codigo: int = 500) -> dict:
    return {
        "exito": False,
        "respuesta": None,
        "error": mensaje,
        "codigo": codigo,
        "timestamp": datetime.utcnow().isoformat(),
    }


def respuesta_ok(respuesta: str, metadatos: dict | None = None) -> dict:
    return {
        "exito": True,
        "respuesta": respuesta,
        "error": None,
        "metadatos": metadatos or {},
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Verificación de variables de entorno al arrancar
# ---------------------------------------------------------------------------

VARS_REQUERIDAS = ["OPENAI_API_KEY"]
VARS_OPCIONALES = {
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "OPENAI_MODEL":    "gpt-3.5-turbo",
    "CHROMA_DIR":      "../data/chroma_db",
    "CHROMA_COLLECTION": "soporte_docs",
    "TOP_K":           "4",
    "MIN_SCORE":       "0.30",
    "LOG_LEVEL":       "INFO",
    "PORT":            "8000",
}


def verificar_entorno() -> list[str]:
    """
    Verifica que las variables de entorno requeridas estén presentes.
    Retorna lista de advertencias/errores encontrados.
    Loguea el estado de cada variable.
    """
    logger = logging.getLogger("utils.entorno")
    advertencias = []

    for var in VARS_REQUERIDAS:
        valor = os.getenv(var, "")
        if not valor:
            msg = f"Variable requerida no configurada: {var}"
            logger.error(msg)
            advertencias.append(msg)
        else:
            # Mostrar solo los primeros 8 chars por seguridad
            preview = valor[:8] + "..." if len(valor) > 8 else "***"
            logger.info(f"  {var}: {preview} ✓")

    for var, default in VARS_OPCIONALES.items():
        valor = os.getenv(var, default)
        logger.info(f"  {var}: {valor}")

    return advertencias