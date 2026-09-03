"""Metricas de error para pronosticos de demanda intermitente.

En repuestos, las metricas clasicas enganan: el MAPE es indefinido cuando la
demanda real es cero (la mayoria de los periodos) y el RMSE premia a los
modelos que pronostican siempre cero. Por eso el proyecto usa como metricas
principales el **WAPE** (error absoluto sobre demanda total) y el **MASE**
(error relativo al de un modelo ingenuo estacional), y complementa con
metricas de sesgo y de calibracion de los cuantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _alinear(y_real, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convierte a arreglos 1D y descarta las posiciones con valores faltantes."""
    real = np.asarray(y_real, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if real.shape != pred.shape:
        raise ValueError(f"Dimensiones incompatibles: {real.shape} vs {pred.shape}")
    valido = np.isfinite(real) & np.isfinite(pred)
    return real[valido], pred[valido]


def mae(y_real, y_pred) -> float:
    """Error absoluto medio."""
    real, pred = _alinear(y_real, y_pred)
    return float(np.mean(np.abs(real - pred))) if real.size else float("nan")


def rmse(y_real, y_pred) -> float:
    """Raiz del error cuadratico medio."""
    real, pred = _alinear(y_real, y_pred)
    return float(np.sqrt(np.mean((real - pred) ** 2))) if real.size else float("nan")


def wape(y_real, y_pred) -> float:
    """Error absoluto ponderado: suma de errores sobre suma de demanda real.

    Es la metrica principal del proyecto: esta definida aunque haya ceros y se
    interpreta como "porcentaje de unidades mal pronosticadas".
    """
    real, pred = _alinear(y_real, y_pred)
    denominador = np.sum(np.abs(real))
    if denominador <= EPS:
        return float("nan")
    return float(np.sum(np.abs(real - pred)) / denominador)


def mape(y_real, y_pred) -> float:
    """Error porcentual absoluto medio, calculado solo sobre periodos con demanda."""
    real, pred = _alinear(y_real, y_pred)
    mascara = np.abs(real) > EPS
    if not mascara.any():
        return float("nan")
    return float(np.mean(np.abs((real[mascara] - pred[mascara]) / real[mascara])))


def smape(y_real, y_pred) -> float:
    """MAPE simetrico: acotado y definido cuando la demanda real es cero."""
    real, pred = _alinear(y_real, y_pred)
    denominador = (np.abs(real) + np.abs(pred)) / 2.0
    mascara = denominador > EPS
    if not mascara.any():
        return 0.0
    return float(np.mean(np.abs(real[mascara] - pred[mascara]) / denominador[mascara]))


def sesgo(y_real, y_pred) -> float:
    """Sesgo medio (positivo = el modelo sobrestima la demanda)."""
    real, pred = _alinear(y_real, y_pred)
    return float(np.mean(pred - real)) if real.size else float("nan")


def sesgo_relativo(y_real, y_pred) -> float:
    """Sesgo acumulado como fraccion de la demanda total real."""
    real, pred = _alinear(y_real, y_pred)
    denominador = np.sum(real)
    if abs(denominador) <= EPS:
        return float("nan")
    return float(np.sum(pred - real) / denominador)


def escala_naive_estacional(serie, periodo_estacional: int = 52) -> float:
    """Denominador del MASE: error del modelo ingenuo estacional en la muestra.

    Si la serie es mas corta que el periodo estacional se degrada al error del
    modelo ingenuo de un paso.
    """
    valores = np.asarray(serie, dtype=float).ravel()
    valores = valores[np.isfinite(valores)]
    if valores.size <= 1:
        return float("nan")
    paso = periodo_estacional if valores.size > periodo_estacional else 1
    diferencias = np.abs(valores[paso:] - valores[:-paso])
    if diferencias.size == 0:
        return float("nan")
    escala = float(np.mean(diferencias))
    return escala if escala > EPS else float("nan")


def mase(y_real, y_pred, escala: float) -> float:
    """Error absoluto escalado: < 1 significa mejor que el ingenuo estacional."""
    if not np.isfinite(escala) or escala <= EPS:
        return float("nan")
    return mae(y_real, y_pred) / escala


def perdida_pinball(y_real, y_pred, cuantil: float) -> float:
    """Perdida pinball: metrica propia de los pronosticos por cuantiles."""
    real, pred = _alinear(y_real, y_pred)
    if real.size == 0:
        return float("nan")
    diferencia = real - pred
    return float(np.mean(np.maximum(cuantil * diferencia, (cuantil - 1.0) * diferencia)))


def cobertura(y_real, y_pred, cuantil: float) -> float:
    """Fraccion de observaciones por debajo del cuantil pronosticado.

    Un cuantil bien calibrado al 0.95 deberia cubrir aproximadamente el 95% de
    los casos: desvios grandes indican stock de seguridad mal dimensionado.
    """
    real, pred = _alinear(y_real, y_pred)
    if real.size == 0:
        return float("nan")
    return float(np.mean(real <= pred))


def tasa_llenado(y_real, y_pred) -> float:
    """Fraccion de la demanda real cubierta por el pronostico (fill rate).

    Traduce el error a lenguaje de negocio: cuanta demanda habria podido
    atenderse si se hubiese comprado exactamente lo pronosticado.
    """
    real, pred = _alinear(y_real, y_pred)
    total = np.sum(real)
    if total <= EPS:
        return float("nan")
    return float(np.sum(np.minimum(real, np.maximum(pred, 0.0))) / total)


def resumen_metricas(
    y_real,
    y_pred,
    escala_mase: float | None = None,
    cuantiles: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Calcula el tablero completo de metricas para un conjunto de pronosticos."""
    resultado: dict[str, float] = {
        "n": float(np.size(_alinear(y_real, y_pred)[0])),
        "mae": mae(y_real, y_pred),
        "rmse": rmse(y_real, y_pred),
        "wape": wape(y_real, y_pred),
        "mape": mape(y_real, y_pred),
        "smape": smape(y_real, y_pred),
        "sesgo": sesgo(y_real, y_pred),
        "sesgo_relativo": sesgo_relativo(y_real, y_pred),
        "tasa_llenado": tasa_llenado(y_real, y_pred),
    }
    if escala_mase is not None:
        resultado["mase"] = mase(y_real, y_pred, escala_mase)
    if cuantiles:
        for nivel, prediccion in sorted(cuantiles.items()):
            etiqueta = f"{int(round(nivel * 100)):02d}"
            resultado[f"pinball_q{etiqueta}"] = perdida_pinball(y_real, prediccion, nivel)
            resultado[f"cobertura_q{etiqueta}"] = cobertura(y_real, prediccion, nivel)
    return resultado


def metricas_por_grupo(
    df: pd.DataFrame,
    columnas_grupo: list[str],
    columna_real: str = "y",
    columna_pred: str = "prediccion",
    escalas: pd.Series | None = None,
    columna_escala: str = "sku",
) -> pd.DataFrame:
    """Aplica `resumen_metricas` dentro de cada grupo (por SKU, horizonte, clase).

    Args:
        escalas: escala MASE indexada por el valor de `columna_escala`.
        columna_escala: columna de agrupacion que indexa `escalas`. El MASE solo
            se calcula si esa columna forma parte de `columnas_grupo`, porque la
            escala es especifica de cada serie.
    """
    if df.empty:
        return pd.DataFrame()
    usar_escalas = escalas is not None and columna_escala in columnas_grupo
    posicion_escala = columnas_grupo.index(columna_escala) if usar_escalas else -1

    filas = []
    for claves, grupo in df.groupby(columnas_grupo, observed=True, sort=True):
        claves = claves if isinstance(claves, tuple) else (claves,)
        escala = None
        if usar_escalas:
            clave = claves[posicion_escala]
            if clave in escalas.index:
                escala = float(escalas.loc[clave])
        fila = dict(zip(columnas_grupo, claves))
        fila.update(
            resumen_metricas(grupo[columna_real], grupo[columna_pred], escala_mase=escala)
        )
        filas.append(fila)
    return pd.DataFrame(filas)
