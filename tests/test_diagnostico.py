"""Pruebas del diagnostico de aptitud de los datos y del mapeo de columnas."""

from __future__ import annotations

import pandas as pd
import pytest

from pronostico.datos import esquema as esq
from pronostico.datos.diagnostico import diagnosticar
from pronostico.datos.esquema import ErrorEsquema, aplicar_mapeo


def test_el_mapeo_renombra_solo_las_columnas_presentes():
    datos = pd.DataFrame({"FECHA_COMP": ["2024-01-01"], "COD_ART": ["A"], "CANT": [1]})
    mapeo = {"FECHA_COMP": "fecha", "COD_ART": "sku", "CANT": "cantidad", "AUSENTE": "canal"}
    resultado = aplicar_mapeo(datos, mapeo)
    assert list(resultado.columns) == ["fecha", "sku", "cantidad"]


def test_el_mapeo_vacio_no_altera_la_tabla():
    datos = pd.DataFrame({"fecha": ["2024-01-01"], "sku": ["A"]})
    pd.testing.assert_frame_equal(aplicar_mapeo(datos, None), datos)
    pd.testing.assert_frame_equal(aplicar_mapeo(datos, {}), datos)


def test_el_mapeo_rechaza_colisiones_de_nombres():
    datos = pd.DataFrame({"CANT": [1], "cantidad": [2]})
    with pytest.raises(ErrorEsquema, match="duplicados"):
        aplicar_mapeo(datos, {"CANT": "cantidad"})


def test_datos_suficientes_son_aptos(movimientos, catalogo):
    informe = diagnosticar(movimientos, catalogo, min_periodos=40, min_demanda_total=5)
    assert informe["apto_para_entrenar"] is True
    assert informe["bloqueantes"] == []
    assert informe["skus_aptos"] > 0
    assert 0.0 <= informe["proporcion_periodos_en_cero"] <= 1.0
    assert "regimenes" in informe


def test_historia_insuficiente_es_bloqueante():
    datos = pd.DataFrame(
        {
            esq.COL_FECHA: pd.date_range("2024-01-01", periods=10, freq="W-MON"),
            esq.COL_SKU: "A",
            esq.COL_CANTIDAD: 5.0,
        }
    )
    informe = diagnosticar(datos, None, min_periodos=52)
    assert informe["apto_para_entrenar"] is False
    assert any("periodos de historia" in m for m in informe["bloqueantes"])


def test_advierte_por_columnas_opcionales_ausentes(movimientos):
    minimos = movimientos[[esq.COL_FECHA, esq.COL_SKU, esq.COL_CANTIDAD]]
    informe = diagnosticar(minimos, None, min_periodos=40, min_demanda_total=5)
    advertencias = " ".join(informe["advertencias"])
    assert esq.COL_QUIEBRE in advertencias
    assert esq.COL_PRECIO in advertencias
    assert "catalogo" in advertencias


def test_advierte_si_el_lead_time_supera_el_horizonte(movimientos, catalogo):
    largo = catalogo.copy()
    largo[esq.COL_LEAD_TIME] = 200
    informe = diagnosticar(
        movimientos, largo, min_periodos=40, min_demanda_total=5, horizonte=4
    )
    assert any("lead time" in m for m in informe["advertencias"])
    assert informe["lead_time_maximo_dias"] == 200.0


def test_advierte_por_skus_sin_ficha_en_el_catalogo(movimientos, catalogo):
    recortado = catalogo.iloc[2:]
    informe = diagnosticar(movimientos, recortado, min_periodos=40, min_demanda_total=5)
    assert informe["skus_sin_ficha_en_catalogo"] == 2
    assert any("no estan en el catalogo" in m for m in informe["advertencias"])
