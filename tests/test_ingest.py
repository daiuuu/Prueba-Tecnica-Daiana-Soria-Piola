from src.ingest import chunkear_fragmento


def _fragmento(texto):
    """Helper para construir un fragmento con la estructura que espera chunkear_fragmento."""
    return {"content": texto, "source": "test.txt", "type": "txt"}


def test_texto_vacio():
    resultado = chunkear_fragmento(_fragmento(""), 100)
    assert resultado == []


def test_texto_corto_no_se_divide():
    texto = "Hola mundo"
    resultado = chunkear_fragmento(_fragmento(texto), 100)

    assert len(resultado) == 1
    assert resultado[0]["content"] == texto


def test_texto_largo_se_divide_por_parrafos():
    parrafo = "A" * 200
    texto = f"{parrafo}\n\n{parrafo}\n\n{parrafo}"

    resultado = chunkear_fragmento(_fragmento(texto), 250)

    assert len(resultado) > 1


def test_cada_chunk_es_parrafo_completo():
    p1 = "Primer párrafo con contenido real."
    p2 = "Segundo párrafo con contenido real."
    p3 = "Tercer párrafo con contenido real."
    texto = f"{p1}\n\n{p2}\n\n{p3}"

    resultado = chunkear_fragmento(_fragmento(texto), 50)

    contenidos = [r["content"] for r in resultado]
    # Cada párrafo original debe aparecer completo en algún chunk
    assert any(p1 in c for c in contenidos)
    assert any(p2 in c for c in contenidos)
    assert any(p3 in c for c in contenidos)


def test_se_preservan_metadatos():
    fragmento = {"content": "A" * 300, "source": "manual.txt", "type": "txt"}

    resultado = chunkear_fragmento(fragmento, 100)

    for chunk in resultado:
        assert chunk["source"] == "manual.txt"
        assert chunk["type"] == "txt"


def test_no_genera_chunks_vacios():
    texto = "\n\n".join(["Párrafo " + str(i) for i in range(20)])

    resultado = chunkear_fragmento(_fragmento(texto), 50)

    assert all(r["content"].strip() for r in resultado)


def test_parrafo_unico_muy_largo_no_se_pierde():
    # Un párrafo solo que supera max_chars igual debe incluirse como chunk único
    texto = "B" * 500

    resultado = chunkear_fragmento(_fragmento(texto), 100)

    assert len(resultado) == 1
    assert resultado[0]["content"] == texto