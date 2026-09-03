"""Pruebas de la traduccion del pronostico a politica de inventario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pronostico.datos import esquema as esq
from pronostico.inventario.politica import (
    COL_CANTIDAD_PEDIDO,
    COL_DEMANDA_LT,
    COL_PUNTO_REORDEN,
    COL_STOCK_SEGURIDAD,
    calcular_politica,
    error_por_sku_desde_backtest,
    factor_servicio,
    periodos_cobertura,
    resumen_politica,
)


@pytest.fixture()
def pronostico_simple() -> pd.DataFrame:
    """Dos SKU con demanda constante de 10 unidades por semana."""
    filas = []
    for sku in ("A", "B"):
        for h in range(1, 9):
            filas.append(
                {
                    esq.COL_SKU: sku,
                    "fecha_origen": pd.Timestamp("2024-01-01"),
                    esq.COL_HORIZONTE: h,
                    "fecha_objetivo": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=h),
                    esq.COL_PREDICCION: 10.0,
                    "q95": 16.0,
                }
            )
    return pd.DataFrame(filas)


@pytest.fixture()
def catalogo_simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            esq.COL_SKU: ["A", "B"],
            esq.COL_LEAD_TIME: [7, 21],
            esq.COL_COSTO: [100.0, 200.0],
            esq.COL_MOQ: [1, 5],
        }
    )


def test_periodos_de_cobertura_redondean_hacia_arriba():
    assert periodos_cobertura(7, 7, "S") == 2
    assert periodos_cobertura(21, 7, "S") == 4
    assert periodos_cobertura(1, 0, "S") == 1


def test_factor_de_servicio_conocido():
    assert factor_servicio(0.95) == pytest.approx(1.645, abs=1e-3)
    assert factor_servicio(0.5) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        factor_servicio(1.0)


def test_demanda_de_reposicion_usa_el_ciclo_correcto(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple,
        catalogo=catalogo_simple,
        error_por_sku=pd.Series({"A": 0.0, "B": 0.0}),
        nivel_servicio=0.95,
    ).set_index(esq.COL_SKU)
    # A: lead time 7 dias + revision 7 -> 2 semanas -> 20 unidades.
    assert politica.loc["A", COL_DEMANDA_LT] == pytest.approx(20.0)
    # B: lead time 21 + 7 -> 4 semanas -> 40 unidades.
    assert politica.loc["B", COL_DEMANDA_LT] == pytest.approx(40.0)


def test_sin_error_de_pronostico_no_hay_stock_de_seguridad(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple,
        catalogo=catalogo_simple,
        error_por_sku=pd.Series({"A": 0.0, "B": 0.0}),
    )
    assert (politica[COL_STOCK_SEGURIDAD] == 0.0).all()
    assert np.allclose(politica[COL_PUNTO_REORDEN], politica[COL_DEMANDA_LT])


def test_mayor_nivel_de_servicio_exige_mas_stock(pronostico_simple, catalogo_simple):
    errores = pd.Series({"A": 3.0, "B": 3.0})
    bajo = calcular_politica(
        pronostico_simple, catalogo_simple, errores, nivel_servicio=0.80
    )[COL_STOCK_SEGURIDAD].sum()
    alto = calcular_politica(
        pronostico_simple, catalogo_simple, errores, nivel_servicio=0.99
    )[COL_STOCK_SEGURIDAD].sum()
    assert alto > bajo


def test_metodo_por_cuantil_usa_la_columna_del_pronostico(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple, catalogo_simple, nivel_servicio=0.95, metodo="cuantil"
    ).set_index(esq.COL_SKU)
    # Brecha de 6 unidades por semana (q95 = 16 contra 10 esperadas) sobre un
    # ciclo de 2 semanas: 6 * raiz(2). No 12, porque el cuantil de la suma no es
    # la suma de los cuantiles.
    assert politica.loc["A", COL_STOCK_SEGURIDAD] == pytest.approx(6.0 * np.sqrt(2))


def test_el_metodo_por_cuantil_no_suma_cuantiles(pronostico_simple, catalogo_simple):
    """Sumar el cuantil de cada periodo sobredimensiona el stock de seguridad."""
    politica = calcular_politica(
        pronostico_simple, catalogo_simple, nivel_servicio=0.95, metodo="cuantil"
    ).set_index(esq.COL_SKU)
    # B cubre 4 semanas: la suma ingenua daria 24 unidades, el ciclo correcto 12.
    assert politica.loc["B", COL_STOCK_SEGURIDAD] == pytest.approx(12.0)


def test_metodo_por_cuantil_falla_sin_la_columna(pronostico_simple, catalogo_simple):
    sin_cuantil = pronostico_simple.drop(columns=["q95"])
    with pytest.raises(ValueError, match="q95"):
        calcular_politica(sin_cuantil, catalogo_simple, nivel_servicio=0.95, metodo="cuantil")


def test_la_cantidad_sugerida_respeta_el_lote_minimo(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple,
        catalogo_simple,
        error_por_sku=pd.Series({"A": 0.0, "B": 0.0}),
        stock_actual=pd.Series({"A": 5.0, "B": 0.0}),
    ).set_index(esq.COL_SKU)
    assert politica.loc["A", COL_CANTIDAD_PEDIDO] == pytest.approx(15.0)
    # B tiene lote minimo de 5: 40 unidades es multiplo exacto.
    assert politica.loc["B", COL_CANTIDAD_PEDIDO] % 5 == 0


def test_stock_suficiente_no_genera_pedido(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple,
        catalogo_simple,
        error_por_sku=pd.Series({"A": 0.0, "B": 0.0}),
        stock_actual=pd.Series({"A": 1000.0, "B": 1000.0}),
    )
    assert (politica[COL_CANTIDAD_PEDIDO] == 0.0).all()


def test_error_por_sku_desde_backtest():
    resultados = pd.DataFrame(
        {
            esq.COL_MODELO: ["m"] * 4,
            esq.COL_SKU: ["A", "A", "B", "B"],
            esq.COL_OBJETIVO: [10.0, 10.0, 5.0, 5.0],
            esq.COL_PREDICCION: [8.0, 12.0, 5.0, 5.0],
        }
    )
    errores = error_por_sku_desde_backtest(resultados, "m")
    assert errores.loc["A"] == pytest.approx(2.0)
    assert errores.loc["B"] == pytest.approx(0.0)


def test_resumen_de_politica(pronostico_simple, catalogo_simple):
    politica = calcular_politica(
        pronostico_simple, catalogo_simple, pd.Series({"A": 2.0, "B": 2.0})
    )
    resumen = resumen_politica(politica)
    assert resumen["skus"] == 2.0
    assert resumen["valor_stock_seguridad"] > 0
