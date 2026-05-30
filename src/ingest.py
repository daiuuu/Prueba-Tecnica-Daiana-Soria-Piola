"""
ingest.py
Orquesta el pipeline completo de ingesta:
  1. Carga documentos con loaders.py
  2. Divide en chunks con overlap
  3. Genera embeddings con OpenAI
  4. Indexa en ChromaDB

Uso desde terminal:
    python ingest.py
    python ingest.py --docs ../docs --db ../data/chroma_db --chunk-size 400 --overlap 80
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
DOCS_DIR = Path(os.getenv("DOCS_DIR", "../docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "../data/chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "soporte_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))      # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80)) # solapamiento entre chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def dividir_en_chunks(texto: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Divide un texto en chunks de tamaño máximo `chunk_size` caracteres,
    con `overlap` caracteres de solapamiento entre chunks consecutivos.
    Intenta respetar párrafos y oraciones como límites naturales de corte.
    """
    if not texto or not texto.strip():
        return []

    # Si el texto ya cabe en un chunk, devolverlo directo
    if len(texto) <= chunk_size:
        return [texto.strip()]

    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + chunk_size

        if fin >= len(texto):
            # Último fragmento
            chunk = texto[inicio:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Buscar un corte natural: doble salto → salto → punto → espacio
        corte = None
        for separador in ["\n\n", "\n", ". ", " "]:
            pos = texto.rfind(separador, inicio, fin)
            if pos > inicio:
                corte = pos + len(separador)
                break

        if corte is None:
            corte = fin  # corte duro si no hay separador

        chunk = texto[inicio:corte].strip()
        if chunk:
            chunks.append(chunk)

        # Retroceder `overlap` chars para el próximo chunk
        inicio = max(corte - overlap, inicio + 1)

    return chunks


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
    BATCH_SIZE = 512  # conservador para evitar límites de tokens
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

    # Obtener o crear colección. embedding_function=None porque
    # vamos a pasar los vectores directamente.
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

    # ChromaDB acepta metadatos con valores str, int, float, bool únicamente
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
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    api_key: str,
    limpiar_coleccion: bool = False,
) -> None:
    """
    Pipeline completo:
      cargar → chunkear → embedir → indexar
    """

    # --- Importar loaders aquí para evitar dependencia circular ---
    try:
        from loaders import cargar_directorio
    except ImportError:
        # Si se ejecuta desde src/ directamente
        sys.path.insert(0, str(Path(__file__).parent))
        from loaders import cargar_directorio

    logger.info("=" * 60)
    logger.info("INICIO DE INGESTA")
    logger.info(f"  Docs:       {docs_dir.resolve()}")
    logger.info(f"  ChromaDB:   {chroma_dir.resolve()}")
    logger.info(f"  Colección:  {collection_name}")
    logger.info(f"  Chunk size: {chunk_size} chars | Overlap: {overlap} chars")
    logger.info(f"  Modelo:     {embedding_model}")
    logger.info("=" * 60)

    # 1. Cargar documentos
    logger.info("Paso 1/4: Cargando documentos...")
    fragmentos = cargar_directorio(docs_dir)

    if not fragmentos:
        logger.error("No se encontraron documentos en el directorio especificado.")
        sys.exit(1)

    logger.info(f"  Fragmentos cargados: {len(fragmentos)}")

    # 2. Dividir en chunks
    logger.info("Paso 2/4: Generando chunks...")
    todos_los_chunks = []
    todos_los_metadatos = []

    for fragmento in fragmentos:
        contenido = fragmento.get("content", "")
        if not contenido:
            continue

        sub_chunks = dividir_en_chunks(contenido, chunk_size, overlap)

        for i, chunk in enumerate(sub_chunks):
            todos_los_chunks.append(chunk)
            todos_los_metadatos.append({
                "source": fragmento.get("source", "desconocido"),
                "type": fragmento.get("type", ""),
                "chunk_index": i,
                "page": fragmento.get("page", 0),
                "doc_id": fragmento.get("id", ""),
            })

    if not todos_los_chunks:
        logger.error("No se generaron chunks. Verificá el contenido de los documentos.")
        sys.exit(1)

    logger.info(f"  Total de chunks generados: {len(todos_los_chunks)}")

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
        description="Ingesta de documentación técnica en ChromaDB"
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
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Tamaño máximo de chunk en caracteres (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Solapamiento entre chunks (default: {CHUNK_OVERLAP})",
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
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        limpiar_coleccion=args.limpiar,
    )