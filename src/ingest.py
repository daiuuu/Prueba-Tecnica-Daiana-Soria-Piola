"""
ingest.py
Orquesta el pipeline completo de ingesta:
  1. Carga documentos con loaders.py (división structure-based)
  2. Aplica corte de seguridad solo a fragmentos excesivamente largos
  3. Genera embeddings con OpenAI
  4. Indexa en ChromaDB

Uso desde terminal:
    python ingest.py
    python ingest.py --docs ../docs --db ../data/chroma_db --max-chars 1200
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Constantes por defecto (sobreescribibles vía args o .env)
# ---------------------------------------------------------------------------
_BASE_DIR       = Path(__file__).parent.parent
DOCS_DIR        = Path(os.getenv("DOCS_DIR", str(_BASE_DIR / "docs")))
CHROMA_DIR      = Path(os.getenv("CHROMA_DIR", str(_BASE_DIR / "data" / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "soporte_docs")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Límite de seguridad: fragmentos más largos que esto se cortan en párrafos.
# No es el tamaño del chunk objetivo — los loaders ya definen eso
# por estructura. Este valor solo actúa como techo de emergencia.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 1200))


# ---------------------------------------------------------------------------
# Chunking structure-based con corte de seguridad
# ---------------------------------------------------------------------------

def chunkear_fragmento(fragmento: dict, max_chars: int) -> list[dict]:
    """
    Recibe un fragmento producido por un loader y lo devuelve como uno o
    más chunks, preservando los metadatos originales.

    Estrategia:
      - Si el contenido entra en max_chars → devuelve el fragmento tal cual.
      - Si no → lo divide por párrafos (doble salto de línea), que es el
        límite estructural más natural que queda después del trabajo del loader.
        Nunca corta por caracteres arbitrarios: si un párrafo solo ya supera
        max_chars, se incluye igual como un chunk único antes de seguir.

    Preserva la coherencia semántica: un chunk siempre contiene
    secciones o párrafos completos, nunca fragmentos a mitad de oración.
    """
    contenido = fragmento.get("content", "").strip()
    if not contenido:
        return []

    # Caso feliz: el fragmento cabe entero
    if len(contenido) <= max_chars:
        return [fragmento]

    # Corte de seguridad por párrafos
    parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
    chunks_resultantes = []
    acumulado = ""

    for parrafo in parrafos:
        if not acumulado:
            acumulado = parrafo
        elif len(acumulado) + len(parrafo) + 2 <= max_chars:
            acumulado += "\n\n" + parrafo
        else:
            # Guardar lo acumulado y empezar nuevo chunk
            chunk = {**fragmento, "content": acumulado}
            chunks_resultantes.append(chunk)
            acumulado = parrafo

    # Último acumulado
    if acumulado:
        chunk = {**fragmento, "content": acumulado}
        chunks_resultantes.append(chunk)

    logger.debug(
        f"Fragmento largo ({len(contenido)} chars) dividido en "
        f"{len(chunks_resultantes)} chunks por párrafos"
    )
    return chunks_resultantes


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def generar_embeddings(textos: list[str], modelo: str, api_key: str) -> list[list[float]]:
    """
    Llama a la API de OpenAI para generar embeddings en batch.
    Respeta el límite de 2048 inputs por llamada dividiendo en batches.
    Retorna lista de vectores en el mismo orden que los textos de entrada.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai no está instalado. Ejecutá: pip install openai")
        sys.exit(1)

    if not api_key:
        logger.error("OPENAI_API_KEY no configurada en .env")
        sys.exit(1)

    cliente = OpenAI(api_key=api_key)
    BATCH_SIZE = 512
    todos_los_vectores = []

    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i : i + BATCH_SIZE]
        intentos = 0
        max_intentos = 3

        while intentos < max_intentos:
            try:
                respuesta = cliente.embeddings.create(
                    model=modelo,
                    input=batch,
                )
                vectores = [item.embedding for item in respuesta.data]
                todos_los_vectores.extend(vectores)
                logger.info(
                    f"Embeddings generados: batch {i // BATCH_SIZE + 1} "
                    f"({len(batch)} textos)"
                )
                break

            except Exception as e:
                intentos += 1
                if intentos == max_intentos:
                    logger.error(
                        f"Error generando embeddings después de {max_intentos} intentos: {e}"
                    )
                    sys.exit(1)
                wait = 2 ** intentos
                logger.warning(f"Error OpenAI: {e}. Reintentando en {wait}s...")
                time.sleep(wait)

    return todos_los_vectores


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def obtener_coleccion(chroma_dir: Path, collection_name: str):
    """
    Crea o recupera una colección ChromaDB persistente.
    Usa embeddings propios (no los de ChromaDB) para mayor control.
    """
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb no está instalado. Ejecutá: pip install chromadb")
        sys.exit(1)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(chroma_dir))

    coleccion = cliente.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"Colección '{collection_name}' lista. "
        f"Documentos existentes: {coleccion.count()}"
    )
    return coleccion


def indexar_chunks(
    coleccion,
    chunks: list[str],
    metadatos: list[dict],
    vectores: list[list[float]],
) -> int:
    """
    Inserta chunks en ChromaDB.
    Genera IDs únicos basados en UUID para evitar colisiones entre ejecuciones.
    Retorna la cantidad de documentos insertados.
    """
    if not chunks:
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatos_limpios = []
    for meta in metadatos:
        meta_limpio = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                meta_limpio[k] = v
            else:
                meta_limpio[k] = str(v)
        metadatos_limpios.append(meta_limpio)

    coleccion.add(
        documents=chunks,
        embeddings=vectores,
        metadatas=metadatos_limpios,
        ids=ids,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_ingesta(
    docs_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    max_chars: int,
    embedding_model: str,
    api_key: str,
    limpiar_coleccion: bool = False,
) -> None:
    """
    Pipeline completo:
      cargar → chunkear por estructura → embedir → indexar
    """
    try:
        from loaders import cargar_directorio
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from loaders import cargar_directorio

    logger.info("=" * 60)
    logger.info("INICIO DE INGESTA (structure-based chunking)")
    logger.info(f"  Docs:       {docs_dir.resolve()}")
    logger.info(f"  ChromaDB:   {chroma_dir.resolve()}")
    logger.info(f"  Colección:  {collection_name}")
    logger.info(f"  Max chars:  {max_chars} (corte de seguridad)")
    logger.info(f"  Modelo:     {embedding_model}")
    logger.info("=" * 60)

    # 1. Cargar documentos
    logger.info("Paso 1/4: Cargando documentos...")
    fragmentos = cargar_directorio(docs_dir)

    if not fragmentos:
        logger.error("No se encontraron documentos en el directorio especificado.")
        sys.exit(1)

    logger.info(f"  Fragmentos estructurales cargados: {len(fragmentos)}")

    # 2. Aplicar corte de seguridad a fragmentos excesivamente largos
    logger.info("Paso 2/4: Aplicando corte de seguridad...")
    todos_los_chunks = []
    todos_los_metadatos = []
    fragmentos_cortados = 0

    for fragmento in fragmentos:
        chunks = chunkear_fragmento(fragmento, max_chars)

        if len(chunks) > 1:
            fragmentos_cortados += 1

        for i, chunk in enumerate(chunks):
            todos_los_chunks.append(chunk["content"])
            todos_los_metadatos.append({
                "source":      chunk.get("source", "desconocido"),
                "type":        chunk.get("type", ""),
                "chunk_index": i,
                "page":        chunk.get("page", 0),
                "doc_id":      chunk.get("id", ""),
            })

    if not todos_los_chunks:
        logger.error("No se generaron chunks. Verificá el contenido de los documentos.")
        sys.exit(1)

    logger.info(f"  Chunks finales: {len(todos_los_chunks)}")
    if fragmentos_cortados:
        logger.info(
            f"  Fragmentos que superaron {max_chars} chars y fueron cortados: "
            f"{fragmentos_cortados}"
        )

    # 3. Generar embeddings
    logger.info("Paso 3/4: Generando embeddings con OpenAI...")
    vectores = generar_embeddings(todos_los_chunks, embedding_model, api_key)
    logger.info(f"  Vectores generados: {len(vectores)}")

    # 4. Indexar en ChromaDB
    logger.info("Paso 4/4: Indexando en ChromaDB...")
    coleccion = obtener_coleccion(chroma_dir, collection_name)

    if limpiar_coleccion and coleccion.count() > 0:
        logger.warning(
            f"  Limpiando colección '{collection_name}' "
            f"({coleccion.count()} docs existentes)..."
        )
        coleccion.delete(where={"source": {"$ne": "__never__"}})

    insertados = indexar_chunks(coleccion, todos_los_chunks, todos_los_metadatos, vectores)

    logger.info("=" * 60)
    logger.info(f"INGESTA COMPLETADA: {insertados} chunks indexados")
    logger.info(f"Total en colección: {coleccion.count()} documentos")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingesta de documentación técnica en ChromaDB (structure-based chunking)"
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DOCS_DIR,
        help=f"Ruta al directorio de documentos (default: {DOCS_DIR})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=CHROMA_DIR,
        help=f"Ruta al directorio de ChromaDB (default: {CHROMA_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Nombre de la colección (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHUNK_CHARS,
        help=(
            f"Límite de caracteres por chunk (default: {MAX_CHUNK_CHARS}). "
            "Solo se aplica como corte de seguridad a fragmentos muy largos; "
            "la división principal la hacen los loaders por estructura."
        ),
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        help="Eliminar documentos existentes antes de indexar",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingesta(
        docs_dir=args.docs,
        chroma_dir=args.db,
        collection_name=args.collection,
        max_chars=args.max_chars,
        embedding_model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        limpiar_coleccion=args.limpiar,
    )