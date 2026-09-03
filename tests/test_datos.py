"""Pruebas de la generacion y la preparacion de datos."""

from __future__ import annotations

import pandas as pd
import pytest

from pronostico.datos import esquema as esq
from pronostico.datos.preparacion import (
    agregar_panel,
    clasificar_abc_xyz,
    completar_grilla,
    filtrar_skus,
    preparar_panel,
    winsorizar_demanda,
)
from pronostico.datos.esquema import ErrorEsquema, validar_movimientos


def test_validacion_rechaza_columnas_faltantes():
    with pytest.raises(ErrorEsquema):
        validar_movimientos(pd.DataFrame({"fecha": ["2024-01-01"]}))


def test_validacion_lleva_devoluciones_a_cero():
    datos = pd.DataFrame(
        {"fecha": ["2024-01-01", "2024-01-02"], "sku": ["A", "A"], "cantidad": [5, -3]}
    )
    resultado = validar_movimientos(datos)
    assert resultado[esq.COL_CANTIDAD].tolist() == [5.0, 0.0]


def test_agregacion_semanal_suma_por_periodo():
    datos = pd.DataFrame(
        {
            # Lunes, martes y el lunes siguiente.
            "fecha": ["2024-01-01", "2024-01-02", "2024-01-08"],
            "sku": ["A", "A", "A"],
            "cantidad": [2, 3, 4],
        }
    )
    panel = agregar_panel(datos, frecuencia="S")
    assert len(panel) == 2
    assert panel[esq.COL_DEMANDA].tolist() == [5.0, 4.0]
    # Los periodos semanales arrancan en lunes.
    assert set(panel[esq.COL_FECHA].dt.dayofweek) == {0}


def test_la_grilla_rellena_los_periodos_sin_venta_con_cero():
    datos = pd.DataFrame(
        {"fecha": ["2024-01-01", "2024-02-05"], "sku": ["A", "A"], "cantidad": [1, 1]}
    )
    panel = completar_grilla(agregar_panel(datos, frecuencia="S"), frecuencia="S")
    assert len(panel) == 6
    assert panel[esq.COL_DEMANDA].sum() == 2.0
    assert (panel[esq.COL_DEMANDA] == 0).sum() == 4


def test_panel_preparado_no_tiene_huecos(panel):
    for _, grupo in panel.groupby(esq.COL_SKU, observed=True):
        diferencias = grupo[esq.COL_FECHA].diff().dropna().dt.days.unique()
        assert set(diferencias) <= {7}


def test_filtrado_descarta_series_cortas():
    panel = pd.DataFrame(
        {
            esq.COL_SKU: ["A"] * 10 + ["B"] * 60,
            esq.COL_FECHA: list(pd.date_range("2024-01-01", periods=10, freq="W-MON"))
            + list(pd.date_range("2024-01-01", periods=60, freq="W-MON")),
            esq.COL_DEMANDA: [5.0] * 70,
        }
    )
    resultado = filtrar_skus(panel, min_periodos=52, min_demanda_total=12)
    assert resultado[esq.COL_SKU].unique().tolist() == ["B"]


def test_winsorizado_recorta_los_picos():
    panel = pd.DataFrame(
        {
            esq.COL_SKU: ["A"] * 100,
            esq.COL_FECHA: pd.date_range("2022-01-03", periods=100, freq="W-MON"),
            esq.COL_DEMANDA: [1.0] * 99 + [500.0],
        }
    )
    resultado = winsorizar_demanda(panel, cuantil=0.95)
    assert resultado[esq.COL_DEMANDA].max() < 500.0
    assert resultado["demanda_observada"].max() == 500.0


def test_clasificacion_abc_xyz_y_regimen(panel, catalogo):
    clases = clasificar_abc_xyz(panel, catalogo)
    assert set(clases["clase_abc"]) <= {"A", "B", "C"}
    assert set(clases["clase_xyz"]) <= {"X", "Y", "Z"}
    assert set(clases["regimen"]) <= {"suave", "intermitente", "erratica", "grumosa"}
    assert len(clases) == panel[esq.COL_SKU].nunique()


def test_pipeline_de_preparacion_completo(movimientos):
    panel = preparar_panel(movimientos, min_periodos=40, min_demanda_total=5)
    assert not panel.empty
    assert panel[esq.COL_DEMANDA].min() >= 0
    assert not panel[[esq.COL_SKU, esq.COL_FECHA]].duplicated().any()
