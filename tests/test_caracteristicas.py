"""Pruebas de la ingenieria de caracteristicas.

La prueba mas importante de este modulo es la de fuga de informacion: si una
variable del origen `t` usara datos posteriores, todas las metricas del
proyecto serian optimistas y el modelo fallaria en produccion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pronostico.caracteristicas.calendario import caracteristicas_calendario, intensidad_campanas
from pronostico.caracteristicas.constructor import (
    ConstructorCaracteristicas,
    construir_objetivos,
)
from pronostico.datos import esquema as esq


def test_intensidad_de_campana_es_acotada_y_estacional():
    fechas = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
    intensidades = intensidad_campanas(fechas)
    columnas = [c for c in intensidades.columns if c.startswith("campana_")]
    assert intensidades[columnas].to_numpy().min() >= 0.0
    # La cosecha gruesa tiene su pico en otono austral (marzo a julio).
    pico = intensidades["campana_cosecha_gruesa"].idxmax()
    assert fechas.iloc[pico].month in {4, 5, 6}


def test_terminos_de_fourier_son_periodicos():
    fechas = pd.Series(pd.to_datetime(["2022-03-15", "2023-03-15"]))
    calendario = caracteristicas_calendario(fechas, ordenes_fourier=2)
    assert calendario.loc[0, "anual_sin_1"] == pytest.approx(
        calendario.loc[1, "anual_sin_1"], abs=1e-3
    )


def test_no_hay_fuga_de_informacion_desde_el_futuro():
    """Las variables del origen `t` no pueden cambiar si se altera el futuro."""
    fechas = pd.date_range("2022-01-03", periods=80, freq="W-MON")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            esq.COL_SKU: "A",
            esq.COL_FECHA: fechas,
            esq.COL_DEMANDA: rng.poisson(3.0, size=80).astype(float),
        }
    )
    constructor = ConstructorCaracteristicas(rezagos=[1, 2, 4], ventanas_moviles=[4, 13])
    original = constructor.construir(panel, catalogo=None)

    alterado = panel.copy()
    alterado.loc[alterado.index[-10:], esq.COL_DEMANDA] += 1000.0
    posterior = constructor.construir(alterado, catalogo=None)

    # Las primeras 70 filas (anteriores al cambio) deben ser identicas.
    columnas = [c for c in original.columns if c != esq.COL_SKU]
    pd.testing.assert_frame_equal(
        original.loc[:69, columnas], posterior.loc[:69, columnas], check_dtype=False
    )


def test_los_objetivos_miran_hacia_adelante():
    fechas = pd.date_range("2022-01-03", periods=10, freq="W-MON")
    matriz = pd.DataFrame(
        {esq.COL_SKU: "A", esq.COL_FECHA: fechas, esq.COL_DEMANDA: np.arange(10.0)}
    )
    objetivos = construir_objetivos(matriz, horizonte=2)
    assert objetivos.loc[0, "y_1"] == 1.0
    assert objetivos.loc[0, "y_2"] == 2.0
    # Las ultimas filas no tienen futuro observado.
    assert pd.isna(objetivos.loc[9, "y_1"])


def test_los_objetivos_no_cruzan_entre_skus():
    fechas = pd.date_range("2022-01-03", periods=5, freq="W-MON")
    matriz = pd.DataFrame(
        {
            esq.COL_SKU: ["A"] * 5 + ["B"] * 5,
            esq.COL_FECHA: list(fechas) * 2,
            esq.COL_DEMANDA: [1.0] * 5 + [100.0] * 5,
        }
    )
    objetivos = construir_objetivos(matriz, horizonte=1)
    assert pd.isna(objetivos.loc[4, "y_1"])  # ultima fila del SKU A


def test_el_constructor_fija_el_esquema(panel, catalogo):
    constructor = ConstructorCaracteristicas()
    matriz = constructor.construir(panel, catalogo)
    X = constructor.ajustar_transformar(matriz)
    assert list(X.columns) == constructor.columnas_
    assert set(constructor.columnas_categoricas_) == {
        esq.COL_FAMILIA,
        esq.COL_MAQUINA,
        esq.COL_CRITICIDAD,
        esq.COL_ORIGEN,
    }
    # Transformar de nuevo debe dar exactamente el mismo esquema.
    pd.testing.assert_index_equal(constructor.transformar(matriz).columns, X.columns)


def test_transformar_falla_si_faltan_predictores(panel, catalogo):
    constructor = ConstructorCaracteristicas()
    matriz = constructor.construir(panel, catalogo)
    constructor.ajustar(matriz)
    with pytest.raises(ValueError, match="Faltan predictores"):
        constructor.transformar(matriz.drop(columns=["rezago_1"]))
