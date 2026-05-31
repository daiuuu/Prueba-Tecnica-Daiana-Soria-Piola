"""
search.py
Búsqueda semántica sobre la colección ChromaDB.
Recibe una pregunta en texto, la convierte en embedding y recupera
los chunks más relevantes de la documentación.

Uso desde terminal (para probar):
    python search.py "¿Cómo reinicio el servicio de autenticación?"
    python search.py "error 502" --top-k 5
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes (sobreescribibles vía .env)
# ---------------------------------------------------------------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR       = Path(os.getenv("CHROMA_DIR", "../data/chroma_db"))
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "soporte_docs")
TOP_K_DEFAULT    = int(os.getenv("TOP_K", "4"))
MIN_SCORE        = float(os.getenv("MIN_SCORE", "0.30"))  # distancia coseno mínima aceptable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validar_pregunta(pregunta: str) -> str:
    """
    Valida y normaliza el input del usuario.
    Lanza ValueError con mensaje claro si el input no es usable.
    """
    if not isinstance(pregunta, str):
        raise ValueError("La pregunta debe ser un texto.")

    pregunta = pregunta.strip()

    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")

    if len(pregunta) < 3:
        raise ValueError("La pregunta es demasiado corta para buscar.")

    if len(pregunta) > 2000:
        raise ValueError("La pregunta supera el límite de 2000 caracteres.")

    return pregunta


def _embedding_con_retry(texto: str, modelo: str, api_key: str) -> list[float]:
    """
    Genera el embedding de un texto con hasta 3 reintentos (backoff exponencial).
    """
    try:
        from openai import OpenAI, APIError, APITimeoutError, RateLimitError
    except ImportError:
        logger.error("openai no instalado. Ejecutá: pip install openai")
        sys.exit(1)

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")

    cliente = OpenAI(api_key=api_key, timeout=15.0)
    max_intentos = 3

    for intento in range(1, max_intentos + 1):
        try:
            respuesta = cliente.embeddings.create(model=modelo, input=texto)
            return respuesta.data[0].embedding

        except RateLimitError:
            wait = 2 ** intento
            logger.warning(f"Rate limit de OpenAI. Reintentando en {wait}s... (intento {intento}/{max_intentos})")
            time.sleep(wait)

        except APITimeoutError:
            wait = 2 ** intento
            logger.warning(f"Timeout de OpenAI. Reintentando en {wait}s... (intento {intento}/{max_intentos})")
            time.sleep(wait)

        except APIError as e:
            logger.error(f"Error de API OpenAI: {e}")
            raise RuntimeError(f"Error de API OpenAI: {e}") from e

    raise RuntimeError("No se pudo generar el embedding después de 3 intentos.")


def _obtener_coleccion(chroma_dir: Path, collection_name: str):
    """
    Conecta a ChromaDB y retorna la colección.
    Lanza RuntimeError si la colección no existe o está vacía.
    """
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb no instalado. Ejecutá: pip install chromadb")
        sys.exit(1)

    if not chroma_dir.exists():
        raise RuntimeError(
            f"El directorio de ChromaDB no existe: {chroma_dir}\n"
            "Ejecutá primero: python ingest.py"
        )

    cliente = chromadb.PersistentClient(path=str(chroma_dir))

    colecciones_existentes = [c.name for c in cliente.list_collections()]
    if collection_name not in colecciones_existentes:
        raise RuntimeError(
            f"La colección '{collection_name}' no existe en ChromaDB.\n"
            "Ejecutá primero: python ingest.py"
        )

    coleccion = cliente.get_collection(name=collection_name)

    if coleccion.count() == 0:
        raise RuntimeError(
            f"La colección '{collection_name}' existe pero está vacía.\n"
            "Ejecutá: python ingest.py para cargar documentos."
        )

    return coleccion


# ---------------------------------------------------------------------------
# Función principal de búsqueda
# ---------------------------------------------------------------------------

def buscar(
    pregunta: str,
    top_k: int = TOP_K_DEFAULT,
    min_score: float = MIN_SCORE,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    api_key: str = OPENAI_API_KEY,
) -> dict:
    """
    Busca los chunks más relevantes para una pregunta.

    Retorna un dict con:
        - pregunta:    texto original validado
        - resultados:  lista de dicts con content, source, score, metadata
        - encontrado:  bool — True si hay al menos un resultado sobre el umbral
        - error:       str con mensaje de error (vacío si todo OK)

    Nunca lanza excepciones hacia afuera: errores se capturan y se devuelven
    en el campo 'error' para que app.py pueda manejarlos limpiamente.
    """
    respuesta_base = {
        "pregunta": pregunta,
        "resultados": [],
        "encontrado": False,
        "error": "",
    }

    # --- Validar input ---
    try:
        pregunta = _validar_pregunta(pregunta)
        respuesta_base["pregunta"] = pregunta
    except ValueError as e:
        respuesta_base["error"] = str(e)
        return respuesta_base

    # --- Generar embedding de la pregunta ---
    try:
        vector_pregunta = _embedding_con_retry(pregunta, embedding_model, api_key)
    except RuntimeError as e:
        respuesta_base["error"] = f"Error al generar embedding: {e}"
        logger.error(respuesta_base["error"])
        return respuesta_base

    # --- Conectar a ChromaDB ---
    try:
        coleccion = _obtener_coleccion(chroma_dir, collection_name)
    except RuntimeError as e:
        respuesta_base["error"] = str(e)
        logger.error(respuesta_base["error"])
        return respuesta_base

    # --- Ejecutar búsqueda ---
    try:
        resultados_raw = coleccion.query(
            query_embeddings=[vector_pregunta],
            n_results=min(top_k, coleccion.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        respuesta_base["error"] = f"Error en la búsqueda en ChromaDB: {e}"
        logger.error(respuesta_base["error"])
        return respuesta_base

    # --- Procesar y filtrar resultados ---
    documentos  = resultados_raw.get("documents", [[]])[0]
    metadatos   = resultados_raw.get("metadatas", [[]])[0]
    distancias  = resultados_raw.get("distances", [[]])[0]

    resultados_filtrados = []

    for doc, meta, distancia in zip(documentos, metadatos, distancias):
        # ChromaDB con distancia coseno devuelve valores entre 0 y 2.
        # Convertimos a similitud: 1 - (distancia / 2), rango [0, 1].
        similitud = round(1 - distancia, 4)

        if similitud < min_score:
            logger.debug(
                f"Chunk descartado (score {similitud:.3f} < umbral {min_score}): "
                f"{doc[:60]}..."
            )
            continue

        resultados_filtrados.append({
            "content":  doc,
            "source":   meta.get("source", "desconocido"),
            "score":    similitud,
            "metadata": meta,
        })

    # Ordenar por score descendente (el más relevante primero)
    resultados_filtrados.sort(key=lambda x: x["score"], reverse=True)

    respuesta_base["resultados"] = resultados_filtrados
    respuesta_base["encontrado"] = len(resultados_filtrados) > 0

    if resultados_filtrados:
        logger.info(
            f"Búsqueda exitosa: {len(resultados_filtrados)} resultado(s) "
            f"para '{pregunta[:50]}...'"
        )
    else:
        logger.info(f"Sin resultados relevantes para: '{pregunta[:50]}'")

    return respuesta_base


def formatear_contexto(resultados: list[dict], max_chars: int = 3000) -> str:
    """
    Convierte la lista de resultados en un bloque de texto listo para
    enviarlo como contexto al LLM en el prompt.

    Respeta un límite de caracteres para no exceder el contexto del modelo.
    """
    if not resultados:
        return ""

    partes = []
    total = 0

    for i, resultado in enumerate(resultados, start=1):
        fuente = resultado.get("source", "desconocido")
        contenido = resultado.get("content", "")
        score = resultado.get("score", 0)

        bloque = (
            f"[Fragmento {i} — Fuente: {fuente} | Relevancia: {score:.2f}]\n"
            f"{contenido}"
        )

        if total + len(bloque) > max_chars:
            # Truncar el último bloque si no entra completo
            espacio_restante = max_chars - total
            if espacio_restante > 100:
                bloque = bloque[:espacio_restante] + "...[truncado]"
                partes.append(bloque)
            break

        partes.append(bloque)
        total += len(bloque)

    return "\n\n---\n\n".join(partes)


# ---------------------------------------------------------------------------
# CLI para pruebas rápidas
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Búsqueda semántica en la documentación de soporte"
    )
    parser.add_argument(
        "pregunta",
        type=str,
        help='Pregunta a buscar. Ejemplo: "¿Cómo reinicio el servicio?"',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Cantidad máxima de resultados (default: {TOP_K_DEFAULT})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help=f"Score mínimo de relevancia entre 0 y 1 (default: {MIN_SCORE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar logs de debug",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    resultado = buscar(
        pregunta=args.pregunta,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if resultado["error"]:
        print(f"\n❌ Error: {resultado['error']}")
        sys.exit(1)

    if not resultado["encontrado"]:
        print("\n⚠️  No se encontró información relevante para esa pregunta.")
        sys.exit(0)

    print(f"\n✅ {len(resultado['resultados'])} resultado(s) encontrado(s):\n")
    print(formatear_contexto(resultado["resultados"]))