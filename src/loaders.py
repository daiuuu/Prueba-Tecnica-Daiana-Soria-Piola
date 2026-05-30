"""
loaders.py
Parsers para cada tipo de archivo soportado: .txt, .pdf, .md, .json
Cada función recibe una ruta de archivo y devuelve una lista de dicts con:
    - content: texto limpio del fragmento
    - source:  nombre del archivo de origen
    - type:    tipo de documento
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers de limpieza
# ---------------------------------------------------------------------------

def _limpiar_texto(texto: str) -> str:
    """Limpieza general aplicable a cualquier texto extraído."""
    # Normalizar saltos de línea
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    # Eliminar caracteres de control excepto salto de línea y tab
    texto = re.sub(r"[^\S\n\t ]+", " ", texto)
    # Colapsar espacios múltiples en una misma línea
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    # Colapsar más de 2 líneas en blanco consecutivas
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _limpiar_markdown(texto: str) -> str:
    """Elimina sintaxis Markdown dejando solo el texto plano."""
    # Quitar bloques de código (``` o ~~~)
    texto = re.sub(r"```[\s\S]*?```", "", texto)
    texto = re.sub(r"~~~[\s\S]*?~~~", "", texto)
    # Quitar código inline
    texto = re.sub(r"`[^`]+`", lambda m: m.group(0).strip("`"), texto)
    # Convertir encabezados en texto plano (conservar el texto)
    texto = re.sub(r"^#{1,6}\s+", "", texto, flags=re.MULTILINE)
    # Quitar negrita / cursiva
    texto = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", texto)
    texto = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", texto)
    # Quitar links [texto](url)
    texto = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", texto)
    # Quitar imágenes ![alt](url)
    texto = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", texto)
    # Quitar líneas horizontales
    texto = re.sub(r"^[-*_]{3,}\s*$", "", texto, flags=re.MULTILINE)
    # Quitar marcadores de listas (-, *, +, números)
    texto = re.sub(r"^[\s]*[-*+]\s+", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"^[\s]*\d+\.\s+", "", texto, flags=re.MULTILINE)
    return _limpiar_texto(texto)


# ---------------------------------------------------------------------------
# Loaders individuales
# ---------------------------------------------------------------------------

def cargar_txt(ruta: Path) -> list[dict]:
    """
    Lee un archivo .txt y devuelve su contenido como un único bloque limpio.
    Si el archivo tiene secciones separadas por líneas en blanco,
    las devuelve como fragmentos independientes.
    """
    try:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"[TXT] No se pudo leer {ruta.name}: {e}")
        return []

    texto = _limpiar_texto(texto)
    if not texto:
        logger.warning(f"[TXT] Archivo vacío: {ruta.name}")
        return []

    # Dividir en bloques separados por línea en blanco doble
    bloques = [b.strip() for b in re.split(r"\n{2,}", texto) if b.strip()]

    fragmentos = []
    for bloque in bloques:
        fragmentos.append({
            "content": bloque,
            "source": ruta.name,
            "type": "txt",
        })

    logger.info(f"[TXT] {ruta.name}: {len(fragmentos)} fragmento(s) extraído(s)")
    return fragmentos


def cargar_pdf(ruta: Path) -> list[dict]:
    """
    Extrae texto de un archivo .pdf página por página usando pdfplumber.
    Cada página se convierte en un fragmento independiente.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("[PDF] pdfplumber no está instalado. Ejecutá: pip install pdfplumber")
        return []

    fragmentos = []
    try:
        with pdfplumber.open(ruta) as pdf:
            for num_pagina, pagina in enumerate(pdf.pages, start=1):
                texto = pagina.extract_text() or ""
                texto = _limpiar_texto(texto)
                if texto:
                    fragmentos.append({
                        "content": texto,
                        "source": ruta.name,
                        "type": "pdf",
                        "page": num_pagina,
                    })
    except Exception as e:
        logger.error(f"[PDF] Error al procesar {ruta.name}: {e}")
        return []

    logger.info(f"[PDF] {ruta.name}: {len(fragmentos)} página(s) extraída(s)")
    return fragmentos


def cargar_md(ruta: Path) -> list[dict]:
    """
    Lee un archivo .md, limpia la sintaxis Markdown y divide por secciones
    (encabezados ## o ###) para generar fragmentos temáticos.
    Secciones pequeñas (menos de 80 chars) se fusionan con la anterior
    para evitar chunks sin contenido útil (ej: solo "Palabras clave: login").
    """
    MIN_SECCION_CHARS = 80  # secciones más cortas se fusionan con la anterior

    try:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"[MD] No se pudo leer {ruta.name}: {e}")
        return []

    if not texto.strip():
        logger.warning(f"[MD] Archivo vacío: {ruta.name}")
        return []

    # Dividir por encabezados para preservar contexto temático
    secciones = re.split(r"(?=^#{1,3} )", texto, flags=re.MULTILINE)
    secciones = [s.strip() for s in secciones if s.strip()]

    # Fusionar secciones pequeñas con la anterior
    secciones_fusionadas = []
    for seccion in secciones:
        texto_limpio = _limpiar_markdown(seccion)
        if not texto_limpio:
            continue
        if secciones_fusionadas and len(texto_limpio) < MIN_SECCION_CHARS:
            # Agregar al chunk anterior en lugar de crear uno nuevo
            secciones_fusionadas[-1] = secciones_fusionadas[-1] + "\n" + texto_limpio
        else:
            secciones_fusionadas.append(texto_limpio)

    fragmentos = []
    for texto_final in secciones_fusionadas:
        if texto_final.strip():
            fragmentos.append({
                "content": texto_final.strip(),
                "source": ruta.name,
                "type": "md",
            })

    logger.info(f"[MD] {ruta.name}: {len(fragmentos)} sección(es) extraída(s)")
    return fragmentos


def cargar_json(ruta: Path) -> list[dict]:
    """
    Parsea un archivo .json con estructura de documentación técnica.
    Soporta dos formatos:
      - Lista de objetos en la raíz: [{ ... }, { ... }]
      - Objeto con clave 'contenido' que contiene la lista (formato MineCatalog)
    Cada objeto del array se convierte en un fragmento de texto estructurado.
    """
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        logger.error(f"[JSON] JSON inválido en {ruta.name}: {e}")
        return []
    except Exception as e:
        logger.error(f"[JSON] No se pudo leer {ruta.name}: {e}")
        return []

    # Detectar lista raíz o clave 'contenido'
    if isinstance(datos, list):
        items = datos
    elif isinstance(datos, dict):
        # Buscar la primera clave que contenga una lista
        items = None
        for clave in ("contenido", "content", "items", "data", "errores"):
            if clave in datos and isinstance(datos[clave], list):
                items = datos[clave]
                break
        if items is None:
            # Si no hay lista anidada, tratar el dict completo como un solo item
            items = [datos]
    else:
        logger.warning(f"[JSON] Formato no reconocido en {ruta.name}")
        return []

    fragmentos = []
    for item in items:
        if not isinstance(item, dict):
            continue
        texto = _aplanar_objeto_json(item)
        if texto:
            fragmentos.append({
                "content": texto,
                "source": ruta.name,
                "type": "json",
                "id": item.get("id", ""),
            })

    logger.info(f"[JSON] {ruta.name}: {len(fragmentos)} entrada(s) extraída(s)")
    return fragmentos


def _aplanar_objeto_json(obj: dict) -> str:
    """
    Convierte un objeto JSON de documentación en texto plano legible.
    Maneja listas de strings convirtiéndolas en viñetas.
    """
    partes = []

    # Campos que se muestran como título
    for campo_titulo in ("titulo", "title", "nombre", "name"):
        if campo_titulo in obj:
            partes.append(f"Título: {obj[campo_titulo]}")
            break

    # Campos de identificación
    for campo_id in ("id", "codigo", "code"):
        if campo_id in obj:
            partes.append(f"ID: {obj[campo_id]}")
            break

    # Campos de categoría
    for campo_cat in ("categoria", "category", "modulo", "module", "tipo", "type"):
        if campo_cat in obj and campo_cat not in ("type",):
            partes.append(f"Categoría: {obj[campo_cat]}")
            break

    # Mensaje al usuario
    for campo_msg in ("mensaje_usuario", "mensaje", "message", "descripcion", "description"):
        if campo_msg in obj:
            partes.append(f"Mensaje: {obj[campo_msg]}")
            break

    # Campos que son listas (causas, soluciones, palabras clave, etc.)
    campos_lista = {
        "causas_posibles": "Causas posibles",
        "causes": "Causas posibles",
        "solucion": "Solución",
        "solution": "Solución",
        "acciones": "Acciones recomendadas",
        "palabras_clave": "Palabras clave",
        "keywords": "Palabras clave",
    }
    for campo, etiqueta in campos_lista.items():
        if campo in obj:
            valor = obj[campo]
            if isinstance(valor, list):
                items_str = "\n  - ".join(str(v) for v in valor)
                partes.append(f"{etiqueta}:\n  - {items_str}")
            elif isinstance(valor, str):
                partes.append(f"{etiqueta}: {valor}")

    # Nivel de soporte
    for campo_nivel in ("nivel_soporte", "nivel", "support_level"):
        if campo_nivel in obj:
            partes.append(f"Nivel de soporte: {obj[campo_nivel]}")
            break

    return _limpiar_texto("\n".join(partes))


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

LOADERS = {
    ".txt": cargar_txt,
    ".pdf": cargar_pdf,
    ".md": cargar_md,
    ".json": cargar_json,
}


def cargar_documento(ruta: Path) -> list[dict]:
    """
    Punto de entrada único. Recibe una ruta y delega al loader correspondiente.
    Retorna lista de fragmentos o lista vacía si el tipo no está soportado.
    """
    extension = ruta.suffix.lower()
    loader = LOADERS.get(extension)

    if loader is None:
        logger.warning(f"Tipo de archivo no soportado: {ruta.name} ({extension})")
        return []

    return loader(ruta)


def cargar_directorio(directorio: Path) -> list[dict]:
    """
    Carga todos los archivos soportados de un directorio.
    Retorna todos los fragmentos concatenados.
    """
    if not directorio.exists():
        logger.error(f"El directorio no existe: {directorio}")
        return []

    todos_los_fragmentos = []
    archivos = sorted(directorio.iterdir())

    for archivo in archivos:
        if archivo.is_file() and archivo.suffix.lower() in LOADERS:
            fragmentos = cargar_documento(archivo)
            todos_los_fragmentos.extend(fragmentos)

    logger.info(
        f"Directorio {directorio.name}: "
        f"{len(archivos)} archivo(s) procesado(s), "
        f"{len(todos_los_fragmentos)} fragmento(s) total"
    )
    return todos_los_fragmentos