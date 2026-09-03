"""Validacion temporal por origen movil (rolling origin).

Una particion aleatoria entrena con el futuro y da resultados imposibles de
reproducir en produccion. Aqui la validacion imita la operacion real: se elige
un origen, se entrena **solo** con lo anterior a ese origen y se pronostica el
horizonte completo. Repitiendo sobre varios origenes se obtiene una medida
estable del error y de su comportamiento segun el horizonte.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ..datos import esquema as esq
from ..modelos.base import COL_FECHA_OBJETIVO, COL_FECHA_ORIGEN, ModeloPronostico
from ..utilidades.registro_log import obtener_logger
from .metricas import (
    cobertura,
    escala_naive_estacional,
    metricas_por_grupo,
    perdida_pinball,
)

logger = obtener_logger(__name__)

FabricaModelo = Callable[[], ModeloPronostico]


def generar_origenes(
    panel: pd.DataFrame,
    n_origenes: int = 6,
    paso: int = 4,
    horizonte: int = 13,
    periodos_reservados: int = 0,
) -> list[pd.Timestamp]:
    """Elige los origenes de pronostico dentro del historico.

    Los origenes se ubican hacia atras desde el final del panel, dejando
    espacio para que cada uno tenga el horizonte completo observado y
    reservando opcionalmente los ultimos periodos como prueba final.
    """
    fechas = np.sort(panel[esq.COL_FECHA].unique())
    if len(fechas) < horizonte + 2:
        raise ValueError("El panel es demasiado corto para el horizonte solicitado")

    ultimo_utilizable = len(fechas) - horizonte - 1 - periodos_reservados
    if ultimo_utilizable < 1:
        raise ValueError(
            "No hay periodos suficientes: reduzca el horizonte o los periodos reservados"
        )

    posiciones = [ultimo_utilizable - i * paso for i in range(n_origenes)]
    posiciones = [p for p in posiciones if p >= 1]
    if not posiciones:
        raise ValueError("No se pudo ubicar ningun origen de validacion")
    return [pd.Timestamp(fechas[p]) for p in sorted(posiciones)]


def escalas_mase(
    panel: pd.DataFrame, hasta: pd.Timestamp, periodo_estacional: int = 52
) -> pd.Series:
    """Escala del MASE por SKU, calculada solo con datos previos a `hasta`."""
    historia = panel.loc[panel[esq.COL_FECHA] <= hasta]
    return historia.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA].apply(
        lambda s: escala_naive_estacional(s, periodo_estacional)
    )


def ejecutar_backtest(
    panel: pd.DataFrame,
    fabricas: Mapping[str, FabricaModelo],
    horizonte: int = 13,
    catalogo: pd.DataFrame | None = None,
    n_origenes: int = 6,
    paso: int = 4,
    periodos_reservados: int = 0,
    cuantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Corre el backtesting de todos los modelos sobre los mismos origenes.

    Returns:
        Tabla larga con una fila por (modelo, origen, SKU, horizonte) que
        incluye la prediccion, el valor real y, si se piden, los cuantiles.
    """
    origenes = generar_origenes(
        panel,
        n_origenes=n_origenes,
        paso=paso,
        horizonte=horizonte,
        periodos_reservados=periodos_reservados,
    )
    logger.info(
        "Backtesting sobre %d origenes (%s a %s) y %d modelos",
        len(origenes),
        origenes[0].date(),
        origenes[-1].date(),
        len(fabricas),
    )

    reales = panel[[esq.COL_SKU, esq.COL_FECHA, esq.COL_DEMANDA]].rename(
        columns={esq.COL_FECHA: COL_FECHA_OBJETIVO, esq.COL_DEMANDA: esq.COL_OBJETIVO}
    )

    resultados: list[pd.DataFrame] = []
    for origen in origenes:
        historia = panel.loc[panel[esq.COL_FECHA] <= origen]
        if historia.empty:
            continue
        for nombre, fabrica in fabricas.items():
            modelo = fabrica()
            modelo.entrenar(historia, catalogo)
            if cuantiles:
                pronostico = modelo.predecir_cuantiles(historia, horizonte, cuantiles)
            else:
                pronostico = modelo.predecir(historia, horizonte)
            if pronostico.empty:
                continue
            pronostico = pronostico.assign(**{esq.COL_MODELO: nombre, esq.COL_ORIGEN_BACKTEST: origen})
            resultados.append(pronostico)
        logger.info("Origen %s completado", origen.date())

    if not resultados:
        raise RuntimeError("El backtesting no produjo resultados")

    tabla = pd.concat(resultados, ignore_index=True)
    tabla = tabla.merge(reales, on=[esq.COL_SKU, COL_FECHA_OBJETIVO], how="inner")
    return tabla.sort_values(
        [esq.COL_MODELO, esq.COL_ORIGEN_BACKTEST, esq.COL_SKU, esq.COL_HORIZONTE]
    ).reset_index(drop=True)


def resumir_backtest(
    resultados: pd.DataFrame,
    panel: pd.DataFrame,
    periodo_estacional: int = 52,
    clases: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Agrega el resultado del backtesting en tablas de metricas.

    Returns:
        Diccionario con las tablas "global", "por_horizonte", "por_origen",
        "por_sku" y, si se entrega la clasificacion, "por_clase".
    """
    if resultados.empty:
        return {}

    primer_origen = resultados[esq.COL_ORIGEN_BACKTEST].min()
    escalas = escalas_mase(panel, primer_origen, periodo_estacional)

    tablas: dict[str, pd.DataFrame] = {
        "global": metricas_por_grupo(resultados, [esq.COL_MODELO]),
        "por_horizonte": metricas_por_grupo(
            resultados, [esq.COL_MODELO, esq.COL_HORIZONTE]
        ),
        "por_origen": metricas_por_grupo(
            resultados, [esq.COL_MODELO, esq.COL_ORIGEN_BACKTEST]
        ),
        "por_sku": metricas_por_grupo(
            resultados, [esq.COL_MODELO, esq.COL_SKU], escalas=escalas
        ),
    }

    # El MASE global se promedia por SKU: cada serie tiene su propia escala.
    mase_medio = (
        tablas["por_sku"].groupby(esq.COL_MODELO, observed=True)["mase"].mean().rename("mase")
    )
    tablas["global"] = tablas["global"].merge(
        mase_medio, on=esq.COL_MODELO, how="left"
    )

    # Si el backtesting incluyo cuantiles se agrega su calibracion y su perdida
    # pinball, que es lo que valida el dimensionamiento del stock de seguridad.
    columnas_cuantil = [c for c in resultados.columns if c.startswith("q") and c[1:].isdigit()]
    if columnas_cuantil:
        filas = []
        for modelo, grupo in resultados.groupby(esq.COL_MODELO, observed=True, sort=True):
            fila: dict[str, float | str] = {esq.COL_MODELO: modelo}
            for columna in columnas_cuantil:
                nivel = int(columna[1:]) / 100.0
                fila[f"cobertura_{columna}"] = cobertura(
                    grupo[esq.COL_OBJETIVO], grupo[columna], nivel
                )
                fila[f"pinball_{columna}"] = perdida_pinball(
                    grupo[esq.COL_OBJETIVO], grupo[columna], nivel
                )
            filas.append(fila)
        tablas["calibracion_cuantiles"] = pd.DataFrame(filas)

    if clases is not None and not clases.empty:
        columnas_clase = [
            c for c in ("clase_abc", "clase_xyz", "clase_abc_xyz", "regimen") if c in clases.columns
        ]
        enriquecido = resultados.merge(
            clases[[esq.COL_SKU] + columnas_clase], on=esq.COL_SKU, how="left"
        )
        if "regimen" in columnas_clase:
            tablas["por_regimen"] = metricas_por_grupo(
                enriquecido, [esq.COL_MODELO, "regimen"]
            )
        if "clase_abc" in columnas_clase:
            tablas["por_clase"] = metricas_por_grupo(
                enriquecido, [esq.COL_MODELO, "clase_abc"]
            )
    return tablas


def elegir_mejor_modelo(
    tabla_global: pd.DataFrame, metrica: str = "wape"
) -> tuple[str, float]:
    """Devuelve el modelo con menor valor de la metrica de seleccion."""
    if tabla_global.empty or metrica not in tabla_global.columns:
        raise ValueError(f"No se puede seleccionar por la metrica '{metrica}'")
    ordenado = tabla_global.dropna(subset=[metrica]).sort_values(metrica)
    if ordenado.empty:
        raise ValueError(f"Todos los valores de '{metrica}' son invalidos")
    fila = ordenado.iloc[0]
    return str(fila[esq.COL_MODELO]), float(fila[metrica])
