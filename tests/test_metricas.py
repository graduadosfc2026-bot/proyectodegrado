"""Pruebas de las metricas de error."""

from __future__ import annotations

import numpy as np
import pytest

from pronostico.evaluacion import metricas as m


def test_pronostico_perfecto_da_error_cero():
    y = np.array([0.0, 3.0, 0.0, 7.0, 2.0])
    assert m.mae(y, y) == pytest.approx(0.0)
    assert m.rmse(y, y) == pytest.approx(0.0)
    assert m.wape(y, y) == pytest.approx(0.0)
    assert m.smape(y, y) == pytest.approx(0.0)


def test_wape_esta_definido_con_ceros():
    """El WAPE es la metrica principal justamente porque tolera los ceros."""
    y = np.array([0.0, 0.0, 10.0])
    p = np.array([1.0, 1.0, 8.0])
    assert m.wape(y, p) == pytest.approx(4.0 / 10.0)
    assert np.isnan(m.mape(np.zeros(3), np.ones(3)))


def test_sesgo_detecta_sobreestimacion():
    y = np.array([2.0, 2.0, 2.0])
    assert m.sesgo(y, y + 1.0) == pytest.approx(1.0)
    assert m.sesgo_relativo(y, y + 1.0) == pytest.approx(0.5)


def test_mase_compara_contra_el_naive_estacional():
    rng = np.random.default_rng(0)
    serie = np.tile([1.0, 5.0, 2.0, 8.0], 20) + rng.normal(0.0, 0.5, 80)
    escala = m.escala_naive_estacional(serie, periodo_estacional=4)
    assert escala > 0

    # Repetir la temporada anterior es, por definicion, un MASE de 1.
    assert m.mase(serie[4:], serie[:-4], escala) == pytest.approx(1.0, rel=1e-6)
    # Un pronostico perfecto da 0 y uno peor que el ingenuo da mas de 1.
    assert m.mase(serie, serie, escala) == pytest.approx(0.0)
    assert m.mase(serie, np.zeros_like(serie), escala) > 1.0


def test_escala_del_mase_es_indefinida_en_una_serie_perfectamente_estacional():
    """Sin error del modelo ingenuo el MASE no esta definido: se devuelve NaN."""
    serie = np.tile([1.0, 5.0, 2.0, 8.0], 20)
    assert np.isnan(m.escala_naive_estacional(serie, periodo_estacional=4))
    assert np.isnan(m.mase(serie, serie, float("nan")))


def test_pinball_penaliza_asimetricamente():
    y = np.array([10.0])
    # Con q=0.9 subestimar cuesta mucho mas que sobrestimar.
    subestima = m.perdida_pinball(y, np.array([5.0]), 0.9)
    sobrestima = m.perdida_pinball(y, np.array([15.0]), 0.9)
    assert subestima > sobrestima


def test_cobertura_y_tasa_de_llenado():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert m.cobertura(y, np.array([5.0] * 4), 0.95) == pytest.approx(1.0)
    assert m.tasa_llenado(y, np.array([1.0, 1.0, 3.0, 4.0])) == pytest.approx(9.0 / 10.0)


def test_dimensiones_incompatibles_fallan():
    with pytest.raises(ValueError):
        m.mae([1.0, 2.0], [1.0])


def test_metricas_por_grupo_solo_escala_el_mase_por_serie():
    """La escala del MASE es por SKU: sin esa columna en el grupo no hay MASE."""
    import pandas as pd

    datos = pd.DataFrame(
        {
            "modelo": ["m"] * 4,
            "sku": ["A", "A", "B", "B"],
            "y": [10.0, 10.0, 4.0, 4.0],
            "prediccion": [8.0, 12.0, 4.0, 4.0],
        }
    )
    escalas = pd.Series({"A": 2.0, "B": 1.0})

    por_sku = m.metricas_por_grupo(datos, ["modelo", "sku"], escalas=escalas)
    assert por_sku.set_index("sku").loc["A", "mase"] == pytest.approx(1.0)
    assert por_sku.set_index("sku").loc["B", "mase"] == pytest.approx(0.0)

    # Agrupando solo por modelo no corresponde ninguna escala: no hay MASE.
    por_modelo = m.metricas_por_grupo(datos, ["modelo"], escalas=escalas)
    assert "mase" not in por_modelo.columns
