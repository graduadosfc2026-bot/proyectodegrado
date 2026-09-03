"""Del pronostico a la decision de compra.

Un pronostico solo genera valor cuando se traduce en una politica de
reposicion. Este modulo convierte las predicciones en:

* demanda esperada durante el tiempo de reposicion (lead time + revision),
* stock de seguridad para el nivel de servicio objetivo,
* punto de reorden y cantidad sugerida a pedir.

Se ofrecen dos formas de dimensionar el stock de seguridad:

1. **Parametrica** (por defecto): usa el desvio del error de pronostico medido
   en el backtesting. Es el metodo de manual, y es honesto porque el desvio
   proviene del error real del modelo, no de la variabilidad de la demanda.
2. **Por cuantiles**: usa directamente los cuantiles altos que estima el
   modelo. No supone normalidad, lo que importa en repuestos con demanda
   grumosa, pero exige que el modelo produzca cuantiles.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..datos import esquema as esq
from ..datos.preparacion import dias_por_periodo
from ..modelos.arbol import etiqueta_cuantil
from ..utilidades.registro_log import obtener_logger

logger = obtener_logger(__name__)

COL_PERIODOS_COBERTURA = "periodos_cobertura"
COL_DEMANDA_LT = "demanda_esperada_reposicion"
COL_SIGMA_LT = "desvio_error_reposicion"
COL_STOCK_SEGURIDAD = "stock_seguridad"
COL_PUNTO_REORDEN = "punto_reorden"
COL_CANTIDAD_PEDIDO = "cantidad_sugerida"
COL_VALOR_INMOVILIZADO = "valor_stock_seguridad"


def periodos_cobertura(
    lead_time_dias: float, periodo_revision_dias: float, frecuencia: str = "S"
) -> int:
    """Periodos de la frecuencia de trabajo que cubre el ciclo de reposicion.

    El ciclo relevante es lead time + periodo de revision: entre dos revisiones
    el stock debe alcanzar hasta que llegue el pedido siguiente.
    """
    dias = float(lead_time_dias) + float(periodo_revision_dias)
    return max(1, math.ceil(dias / dias_por_periodo(frecuencia)))


def factor_servicio(nivel_servicio: float) -> float:
    """Factor `z` de la normal estandar asociado al nivel de servicio."""
    if not 0.5 <= nivel_servicio < 1.0:
        raise ValueError("El nivel de servicio debe estar en [0.5, 1)")
    return float(norm.ppf(nivel_servicio))


def _dentro_del_ciclo(pronosticos: pd.DataFrame, periodos: pd.Series) -> pd.Series:
    """Mascara de las filas del pronostico que caen dentro del ciclo de cada SKU."""
    limites = pronosticos[esq.COL_SKU].map(periodos)
    return pronosticos[esq.COL_HORIZONTE] <= limites


def _acumular_horizonte(
    pronosticos: pd.DataFrame, periodos: pd.Series, columna: str
) -> pd.Series:
    """Suma una columna del pronostico sobre los primeros periodos de cada SKU."""
    dentro = _dentro_del_ciclo(pronosticos, periodos)
    return pronosticos.loc[dentro].groupby(esq.COL_SKU, observed=True)[columna].sum()


def _promediar_horizonte(
    pronosticos: pd.DataFrame, periodos: pd.Series, columna: str
) -> pd.Series:
    """Promedia una columna del pronostico sobre los primeros periodos de cada SKU."""
    dentro = _dentro_del_ciclo(pronosticos, periodos)
    return pronosticos.loc[dentro].groupby(esq.COL_SKU, observed=True)[columna].mean()


def calcular_politica(
    pronosticos: pd.DataFrame,
    catalogo: pd.DataFrame | None = None,
    error_por_sku: pd.Series | None = None,
    nivel_servicio: float = 0.95,
    lead_time_dias_por_defecto: float = 21.0,
    periodo_revision_dias: float = 7.0,
    frecuencia: str = "S",
    stock_actual: pd.Series | None = None,
    metodo: str = "parametrico",
) -> pd.DataFrame:
    """Calcula la politica de reposicion por SKU a partir del pronostico.

    Args:
        pronosticos: salida de `ModeloPronostico.predecir` o `predecir_cuantiles`
            para un unico origen.
        catalogo: aporta lead time, costo y lote minimo por SKU.
        error_por_sku: desvio del error de pronostico por periodo y por SKU
            (por ejemplo el RMSE del backtesting). Requerido por el metodo
            parametrico.
        stock_actual: existencias actuales por SKU; si se entrega se calcula la
            cantidad a pedir, si no solo el punto de reorden.
        metodo: "parametrico" o "cuantil".

    Returns:
        Una fila por SKU con la demanda de reposicion, el stock de seguridad,
        el punto de reorden y la cantidad sugerida.
    """
    if pronosticos.empty:
        return pd.DataFrame()

    atributos = (
        esq.validar_catalogo(catalogo).set_index(esq.COL_SKU)
        if catalogo is not None
        else pd.DataFrame()
    )
    skus = pd.Index(sorted(pronosticos[esq.COL_SKU].unique()), name=esq.COL_SKU)

    lead_times = pd.Series(lead_time_dias_por_defecto, index=skus, dtype=float)
    if esq.COL_LEAD_TIME in atributos.columns:
        lead_times = (
            atributos[esq.COL_LEAD_TIME]
            .reindex(skus)
            .astype(float)
            .fillna(lead_time_dias_por_defecto)
        )

    periodos = lead_times.apply(
        lambda dias: periodos_cobertura(dias, periodo_revision_dias, frecuencia)
    ).astype(int)
    horizonte_disponible = int(pronosticos[esq.COL_HORIZONTE].max())
    recortados = int((periodos > horizonte_disponible).sum())
    if recortados:
        logger.warning(
            "%d SKU tienen un ciclo de reposicion mayor al horizonte (%d periodos): "
            "se extrapola con la demanda media del horizonte",
            recortados,
            horizonte_disponible,
        )

    demanda_horizonte = _acumular_horizonte(
        pronosticos, periodos.clip(upper=horizonte_disponible), esq.COL_PREDICCION
    ).reindex(skus).fillna(0.0)
    media_periodo = (
        pronosticos.groupby(esq.COL_SKU, observed=True)[esq.COL_PREDICCION]
        .mean()
        .reindex(skus)
        .fillna(0.0)
    )
    # Los ciclos mas largos que el horizonte se completan con la demanda media.
    faltantes = (periodos - periodos.clip(upper=horizonte_disponible)).clip(lower=0)
    demanda_lt = demanda_horizonte + faltantes * media_periodo

    resultado = pd.DataFrame(
        {
            COL_PERIODOS_COBERTURA: periodos,
            esq.COL_LEAD_TIME: lead_times,
            COL_DEMANDA_LT: demanda_lt,
        },
        index=skus,
    )

    if metodo == "cuantil":
        columna_q = etiqueta_cuantil(nivel_servicio)
        if columna_q not in pronosticos.columns:
            raise ValueError(
                f"El pronostico no incluye el cuantil '{columna_q}' requerido por el "
                f"nivel de servicio {nivel_servicio}"
            )
        # La brecha entre el cuantil y la media mide, en unidades por periodo,
        # cuanta incertidumbre reconoce el propio modelo. No se suma a lo largo
        # del ciclo: el cuantil de la suma no es la suma de los cuantiles. Con
        # errores independientes entre periodos la brecha crece con la raiz del
        # numero de periodos, igual que en el metodo parametrico, pero sin
        # suponer que la distribucion es normal.
        con_brecha = pronosticos.assign(
            _brecha=(pronosticos[columna_q] - pronosticos[esq.COL_PREDICCION]).clip(lower=0.0)
        )
        brecha_periodo = (
            _promediar_horizonte(
                con_brecha, periodos.clip(upper=horizonte_disponible), "_brecha"
            )
            .reindex(skus)
            .fillna(0.0)
        )
        raiz_periodos = np.sqrt(periodos.astype(float))
        resultado[COL_SIGMA_LT] = brecha_periodo * raiz_periodos / factor_servicio(
            nivel_servicio
        )
        resultado[COL_STOCK_SEGURIDAD] = (brecha_periodo * raiz_periodos).clip(lower=0.0)
    elif metodo == "parametrico":
        if error_por_sku is None:
            raise ValueError(
                "El metodo parametrico necesita `error_por_sku` (desvio del error "
                "de pronostico por periodo, normalmente el RMSE del backtesting)"
            )
        sigma_periodo = error_por_sku.reindex(skus).astype(float)
        sigma_periodo = sigma_periodo.fillna(float(sigma_periodo.median(skipna=True) or 0.0))
        # Errores de periodos consecutivos se suponen independientes: la varianza
        # del ciclo es la suma de varianzas.
        sigma_lt = sigma_periodo * np.sqrt(periodos.astype(float))
        resultado[COL_SIGMA_LT] = sigma_lt
        resultado[COL_STOCK_SEGURIDAD] = (factor_servicio(nivel_servicio) * sigma_lt).clip(
            lower=0.0
        )
    else:
        raise ValueError(f"Metodo de stock de seguridad desconocido: '{metodo}'")

    resultado[COL_PUNTO_REORDEN] = resultado[COL_DEMANDA_LT] + resultado[COL_STOCK_SEGURIDAD]

    lote_minimo = pd.Series(1.0, index=skus)
    if esq.COL_MOQ in atributos.columns:
        lote_minimo = atributos[esq.COL_MOQ].reindex(skus).astype(float).fillna(1.0).clip(lower=1.0)

    if stock_actual is not None:
        existencias = stock_actual.reindex(skus).astype(float).fillna(0.0)
        resultado["stock_actual"] = existencias
        faltante = (resultado[COL_PUNTO_REORDEN] - existencias).clip(lower=0.0)
        # Se redondea hacia arriba al multiplo del lote minimo del proveedor.
        resultado[COL_CANTIDAD_PEDIDO] = np.ceil(faltante / lote_minimo) * lote_minimo
    else:
        resultado[COL_CANTIDAD_PEDIDO] = np.ceil(
            resultado[COL_PUNTO_REORDEN] / lote_minimo
        ) * lote_minimo

    if esq.COL_COSTO in atributos.columns:
        costos = atributos[esq.COL_COSTO].reindex(skus).astype(float).fillna(0.0)
        resultado[esq.COL_COSTO] = costos
        resultado[COL_VALOR_INMOVILIZADO] = resultado[COL_STOCK_SEGURIDAD] * costos

    resultado["nivel_servicio"] = nivel_servicio
    resultado["metodo_stock_seguridad"] = metodo
    return resultado.reset_index()


def error_por_sku_desde_backtest(
    resultados: pd.DataFrame, modelo: str | None = None
) -> pd.Series:
    """Desvio del error de pronostico por SKU (RMSE) a partir del backtesting."""
    tabla = resultados
    if modelo is not None:
        tabla = tabla.loc[tabla[esq.COL_MODELO] == modelo]
    if tabla.empty:
        raise ValueError(f"No hay resultados de backtesting para el modelo '{modelo}'")
    error = tabla[esq.COL_OBJETIVO] - tabla[esq.COL_PREDICCION]
    return (
        error.pow(2)
        .groupby(tabla[esq.COL_SKU], observed=True)
        .mean()
        .pow(0.5)
        .rename("rmse_periodo")
    )


def resumen_politica(politica: pd.DataFrame) -> dict[str, float]:
    """Indicadores agregados de la politica propuesta, para el reporte."""
    if politica.empty:
        return {}
    resumen = {
        "skus": float(len(politica)),
        "demanda_reposicion_total": float(politica[COL_DEMANDA_LT].sum()),
        "stock_seguridad_total": float(politica[COL_STOCK_SEGURIDAD].sum()),
        "punto_reorden_medio": float(politica[COL_PUNTO_REORDEN].mean()),
    }
    if COL_VALOR_INMOVILIZADO in politica.columns:
        resumen["valor_stock_seguridad"] = float(politica[COL_VALOR_INMOVILIZADO].sum())
    return resumen
