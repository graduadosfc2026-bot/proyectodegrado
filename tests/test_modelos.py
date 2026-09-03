"""Pruebas de los modelos de pronostico."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pronostico.datos import esquema as esq
from pronostico.modelos.base import COL_FECHA_OBJETIVO, fechas_futuras
from pronostico.modelos.referencia import SBA, Croston, MediaMovil, Naive, NaiveEstacional
from pronostico.modelos.registro import crear_modelo, modelos_disponibles


def _serie(valores: list[float], sku: str = "A") -> pd.DataFrame:
    fechas = pd.date_range("2022-01-03", periods=len(valores), freq="W-MON")
    return pd.DataFrame(
        {esq.COL_SKU: sku, esq.COL_FECHA: fechas, esq.COL_DEMANDA: list(map(float, valores))}
    )


def test_fechas_futuras_continuan_la_serie_semanal():
    futuras = fechas_futuras(pd.Timestamp("2024-01-01"), 3, "S")
    assert list(futuras.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-15", "2024-01-22"]


def test_naive_repite_el_ultimo_valor():
    modelo = Naive().entrenar(_serie([1, 2, 3, 9]))
    pronostico = modelo.predecir(_serie([1, 2, 3, 9]), horizonte=3)
    assert pronostico[esq.COL_PREDICCION].tolist() == [9.0, 9.0, 9.0]


def test_naive_estacional_repite_la_temporada_anterior():
    historia = _serie(list(np.tile([1.0, 5.0, 2.0, 8.0], 5)))
    modelo = NaiveEstacional(periodo_estacional=4).entrenar(historia)
    pronostico = modelo.predecir(historia, horizonte=4)
    assert pronostico[esq.COL_PREDICCION].tolist() == [1.0, 5.0, 2.0, 8.0]


def test_media_movil_promedia_la_ventana():
    historia = _serie([10, 10, 10, 4, 4])
    modelo = MediaMovil(ventana=2).entrenar(historia)
    assert modelo.predecir(historia, 2)[esq.COL_PREDICCION].tolist() == [4.0, 4.0]


def test_croston_estima_la_tasa_de_demanda_intermitente():
    # Demanda de 6 unidades cada 3 periodos -> tasa esperada cercana a 2/periodo.
    historia = _serie([6, 0, 0] * 20)
    tasa = Croston(alfa=0.2).entrenar(historia).predecir(historia, 1)[esq.COL_PREDICCION].iloc[0]
    assert tasa == pytest.approx(2.0, rel=0.2)


def test_sba_corrige_a_la_baja_el_sesgo_de_croston():
    historia = _serie([6, 0, 0] * 20)
    croston = Croston(alfa=0.2).entrenar(historia).predecir(historia, 1)[esq.COL_PREDICCION].iloc[0]
    sba = SBA(alfa=0.2).entrenar(historia).predecir(historia, 1)[esq.COL_PREDICCION].iloc[0]
    assert sba < croston


def test_serie_sin_demanda_pronostica_cero():
    historia = _serie([0.0] * 30)
    for nombre in ("croston", "sba", "tsb", "naive"):
        modelo = crear_modelo(nombre).entrenar(historia)
        assert modelo.predecir(historia, 3)[esq.COL_PREDICCION].sum() == 0.0


def test_los_pronosticos_nunca_son_negativos(panel, catalogo):
    for nombre in modelos_disponibles():
        if nombre == "gbrt_global":
            continue
        modelo = crear_modelo(nombre, periodo_estacional=52).entrenar(panel, catalogo)
        pronostico = modelo.predecir(panel, horizonte=6)
        assert (pronostico[esq.COL_PREDICCION] >= 0).all()
        assert len(pronostico) == panel[esq.COL_SKU].nunique() * 6


def test_el_pronostico_arranca_despues_del_ultimo_dato(panel):
    modelo = Naive().entrenar(panel)
    pronostico = modelo.predecir(panel, horizonte=4)
    assert pronostico[COL_FECHA_OBJETIVO].min() > panel[esq.COL_FECHA].max()


def test_cuantiles_por_defecto_son_monotonos(panel):
    modelo = MediaMovil(ventana=8, periodo_estacional=52).entrenar(panel)
    pronostico = modelo.predecir_cuantiles(panel, 3, [0.5, 0.9])
    assert (pronostico["q90"] >= pronostico["q50"] - 1e-9).all()


def test_registro_rechaza_modelos_desconocidos():
    with pytest.raises(KeyError):
        crear_modelo("modelo_inexistente")


@pytest.fixture(scope="module")
def modelo_entrenado(panel, catalogo):
    """Modelo global pequeno, compartido por las pruebas de esta seccion."""
    modelo = crear_modelo(
        "gbrt_global",
        horizonte=3,
        cuantiles=[0.5, 0.9],
        hiperparametros={"max_iter": 30, "max_bins": 32, "early_stopping": False},
        periodo_estacional=52,
    )
    return modelo.entrenar(panel, catalogo)


class TestModeloGlobal:
    """Pruebas del modelo principal de gradient boosting."""

    def test_entrena_un_estimador_por_horizonte(self, modelo_entrenado):
        assert sorted(modelo_entrenado.modelos_) == [1, 2, 3]
        assert len(modelo_entrenado.modelos_cuantil_) == 6

    def test_predice_para_todos_los_skus(self, modelo_entrenado, panel):
        pronostico = modelo_entrenado.predecir(panel, horizonte=3)
        assert len(pronostico) == panel[esq.COL_SKU].nunique() * 3
        assert (pronostico[esq.COL_PREDICCION] >= 0).all()
        assert pronostico[esq.COL_PREDICCION].notna().all()

    def test_los_cuantiles_son_monotonos(self, modelo_entrenado, panel):
        pronostico = modelo_entrenado.predecir_cuantiles(panel, 3, [0.5, 0.9])
        assert (pronostico["q90"] >= pronostico["q50"] - 1e-9).all()

    def test_rechaza_horizontes_mayores_al_entrenado(self, modelo_entrenado, panel):
        with pytest.raises(ValueError, match="entrenado hasta"):
            modelo_entrenado.predecir(panel, horizonte=10)

    def test_rechaza_cuantiles_no_entrenados(self, modelo_entrenado, panel):
        with pytest.raises(ValueError, match="cuantiles"):
            modelo_entrenado.predecir_cuantiles(panel, 2, [0.99])

    def test_calcula_importancia_de_variables(self, modelo_entrenado, panel):
        importancia = modelo_entrenado.importancia_permutacion(
            panel, horizonte=1, n_repeticiones=2, max_filas=400
        )
        assert set(importancia.columns) == {"variable", "importancia", "desvio"}
        assert len(importancia) == len(modelo_entrenado.constructor.columnas_)


def test_el_decaimiento_temporal_pondera_la_historia_reciente(panel, catalogo):
    """Con decaimiento, la historia antigua pesa menos y el ajuste cambia."""
    def construir(decaimiento: float):
        return crear_modelo(
            "gbrt_global",
            horizonte=1,
            hiperparametros={"max_iter": 20, "max_bins": 32, "early_stopping": False},
            decaimiento_temporal=decaimiento,
            periodo_estacional=52,
        ).entrenar(panel, catalogo)

    sin_decaimiento = construir(1.0).predecir(panel, 1)[esq.COL_PREDICCION].to_numpy()
    con_decaimiento = construir(0.95).predecir(panel, 1)[esq.COL_PREDICCION].to_numpy()
    assert sin_decaimiento.shape == con_decaimiento.shape
    assert not np.allclose(sin_decaimiento, con_decaimiento)
