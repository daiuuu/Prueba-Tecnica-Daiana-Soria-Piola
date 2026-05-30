"""
test_search.py

Tests unitarios para search.py
Ejecutar con:

    pytest tests/test_search.py -v
"""

import pytest

from src.search import (
    _validar_pregunta,
    formatear_contexto
)


# ==========================================================
# Tests de _validar_pregunta()
# ==========================================================

def test_pregunta_valida():
    pregunta = "¿Qué significa error 502?"

    resultado = _validar_pregunta(pregunta)

    assert resultado == pregunta


def test_pregunta_con_espacios():
    pregunta = "   ¿Cómo reinicio el servicio?   "

    resultado = _validar_pregunta(pregunta)

    assert resultado == "¿Cómo reinicio el servicio?"


def test_pregunta_vacia():
    with pytest.raises(ValueError) as exc:
        _validar_pregunta("")

    assert "vacía" in str(exc.value)


def test_pregunta_solo_espacios():
    with pytest.raises(ValueError) as exc:
        _validar_pregunta("      ")

    assert "vacía" in str(exc.value)


def test_pregunta_demasiado_corta():
    with pytest.raises(ValueError) as exc:
        _validar_pregunta("ok")

    assert "demasiado corta" in str(exc.value)


def test_pregunta_demasiado_larga():
    pregunta = "a" * 2001

    with pytest.raises(ValueError) as exc:
        _validar_pregunta(pregunta)

    assert "2000 caracteres" in str(exc.value)


def test_pregunta_no_es_string():
    with pytest.raises(ValueError) as exc:
        _validar_pregunta(123)

    assert "texto" in str(exc.value)


def test_pregunta_none():
    with pytest.raises(ValueError):
        _validar_pregunta(None)


# ==========================================================
# Tests de formatear_contexto()
# ==========================================================

def test_formatear_contexto_vacio():
    resultado = formatear_contexto([])

    assert resultado == ""


def test_formatear_contexto_un_resultado():
    resultados = [
        {
            "content": "El error 502 indica que el gateway recibió una respuesta inválida.",
            "source": "errores.md",
            "score": 0.95
        }
    ]

    contexto = formatear_contexto(resultados)

    assert "errores.md" in contexto
    assert "error 502" in contexto
    assert "0.95" in contexto


def test_formatear_contexto_varios_resultados():
    resultados = [
        {
            "content": "Primer fragmento",
            "source": "doc1.txt",
            "score": 0.90
        },
        {
            "content": "Segundo fragmento",
            "source": "doc2.txt",
            "score": 0.85
        }
    ]

    contexto = formatear_contexto(resultados)

    assert "Primer fragmento" in contexto
    assert "Segundo fragmento" in contexto
    assert "doc1.txt" in contexto
    assert "doc2.txt" in contexto


def test_formatear_contexto_respeta_limite():
    resultados = [
        {
            "content": "A" * 5000,
            "source": "gigante.txt",
            "score": 0.99
        }
    ]

    contexto = formatear_contexto(
        resultados,
        max_chars=500
    )

    assert len(contexto) <= 520
    assert "truncado" in contexto