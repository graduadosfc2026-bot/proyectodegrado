"""Pruebas de la validacion por origen movil."""

from __future__ import annotations

import pandas as pd
import pytest

from pronostico.datos import esquema as esq
from pronostico.evaluacion.backtesting import (
    ejecutar_backtest,
    elegir_mejor_modelo,
    generar_origenes,
    resumir_backtest,
)
from pronostico.modelos.registro import crear_modelo


def test_los_origenes_dejan_espacio_para_el_horizonte(panel):
    origenes = generar_origenes(panel, n_origenes=3, paso=4, horizonte=6)
    ultima = panel[esq.COL_FECHA].max()
    assert len(origenes) == 3
    assert origenes == sorted(origenes)
    for origen in origenes:
        posteriores = panel.loc[panel[esq.COL_FECHA] > origen, esq.COL_FECHA].nunique()
        assert posteriores >= 6
    assert max(origenes) < ultima


def test_los_periodos_reservados_alejan_los_origenes(panel):
    sin_reserva = generar_origenes(panel, n_origenes=1, paso=4, horizonte=4)
    con_reserva = generar_origenes(
        panel, n_origenes=1, paso=4, horizonte=4, periodos_reservados=10
    )
    assert con_reserva[0] < sin_reserva[0]


def test_falla_si_el_horizonte_no_entra_en_la_historia(panel):
    with pytest.raises(ValueError):
        generar_origenes(panel, n_origenes=1, paso=1, horizonte=10_000)


def test_el_backtesting_no_usa_datos_futuros(panel, catalogo):
    """Cada pronostico debe corresponder a fechas posteriores a su origen."""
    resultados = ejecutar_backtest(
        panel,
        fabricas={"naive": lambda: crear_modelo("naive")},
        horizonte=4,
        catalogo=catalogo,
        n_origenes=2,
        paso=4,
    )
    assert (resultados["fecha_objetivo"] > resultados[esq.COL_ORIGEN_BACKTEST]).all()
    assert resultados[esq.COL_OBJETIVO].notna().all()
    assert set(resultados[esq.COL_HORIZONTE]) == {1, 2, 3, 4}


def test_el_resumen_produce_las_tablas_esperadas(panel, catalogo):
    resultados = ejecutar_backtest(
        panel,
        fabricas={
            "naive": lambda: crear_modelo("naive"),
            "sba": lambda: crear_modelo("sba"),
        },
        horizonte=4,
        catalogo=catalogo,
        n_origenes=2,
        paso=4,
    )
    tablas = resumir_backtest(resultados, panel)
    assert {"global", "por_horizonte", "por_origen", "por_sku"} <= set(tablas)
    assert len(tablas["global"]) == 2
    assert "mase" in tablas["global"].columns
    assert tablas["global"]["wape"].notna().all()


def test_seleccion_del_mejor_modelo():
    tabla = pd.DataFrame({esq.COL_MODELO: ["a", "b"], "wape": [0.8, 0.4]})
    assert elegir_mejor_modelo(tabla, "wape") == ("b", 0.4)
    with pytest.raises(ValueError):
        elegir_mejor_modelo(tabla, "metrica_inexistente")
