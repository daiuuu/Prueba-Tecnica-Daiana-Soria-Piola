from src.ingest import dividir_en_chunks


def test_texto_vacio():
    resultado = dividir_en_chunks("", 100, 20)
    assert resultado == []


def test_texto_corto():
    texto = "Hola mundo"
    resultado = dividir_en_chunks(texto, 100, 20)

    assert len(resultado) == 1
    assert resultado[0] == texto


def test_genera_varios_chunks():
    texto = "A" * 1000

    resultado = dividir_en_chunks(
        texto,
        chunk_size=200,
        overlap=50
    )

    assert len(resultado) > 1


def test_overlap_funciona():
    texto = "A" * 500

    chunks = dividir_en_chunks(
        texto,
        chunk_size=100,
        overlap=20
    )

    assert chunks[0][-20:] == chunks[1][:20]

def test_no_genera_chunks_vacios():
    texto = "Texto de prueba " * 100

    chunks = dividir_en_chunks(
        texto,
        chunk_size=100,
        overlap=10
    )

    assert all(chunk.strip() for chunk in chunks)