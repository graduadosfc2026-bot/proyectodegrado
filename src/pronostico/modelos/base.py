"""Interfaz comun de los modelos de pronostico.

Todos los modelos, tanto los estadisticos por serie como el modelo global de
aprendizaje automatico, exponen el mismo contrato:

    modelo.entrenar(panel, catalogo)
    modelo.predecir(historia, horizonte) -> tabla larga de pronosticos

`historia` es el panel truncado en el origen del pronostico. Gracias a eso el
backtesting y el pipeline de produccion tratan a todos los modelos por igual.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..datos import esquema as esq
from ..datos.preparacion import frecuencia_offset

COL_FECHA_ORIGEN = "fecha_origen"
COL_FECHA_OBJETIVO = "fecha_objetivo"

COLUMNAS_PRONOSTICO = [
    esq.COL_SKU,
    COL_FECHA_ORIGEN,
    esq.COL_HORIZONTE,
    COL_FECHA_OBJETIVO,
    esq.COL_PREDICCION,
]


def fechas_futuras(
    fecha_origen: pd.Timestamp, horizonte: int, frecuencia: str = "S"
) -> pd.DatetimeIndex:
    """Fechas de los `horizonte` periodos siguientes al origen."""
    offset = frecuencia_offset(frecuencia)
    futuras = pd.date_range(start=fecha_origen, periods=horizonte + 1, freq=offset)
    return futuras[1:]


def esqueleto_pronostico(
    origenes: pd.Series, horizonte: int, frecuencia: str = "S"
) -> pd.DataFrame:
    """Arma la grilla (SKU x horizonte) sobre la que se escriben las predicciones.

    Args:
        origenes: ultima fecha observada de cada SKU, indexada por SKU.
    """
    if len(origenes) == 0:
        return pd.DataFrame(columns=COLUMNAS_PRONOSTICO)
    filas = []
    for sku, fecha_origen in origenes.items():
        objetivo = fechas_futuras(pd.Timestamp(fecha_origen), horizonte, frecuencia)
        filas.append(
            pd.DataFrame(
                {
                    esq.COL_SKU: sku,
                    COL_FECHA_ORIGEN: pd.Timestamp(fecha_origen),
                    esq.COL_HORIZONTE: np.arange(1, horizonte + 1),
                    COL_FECHA_OBJETIVO: objetivo,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


class ModeloPronostico(ABC):
    """Clase base de todos los modelos de pronostico del proyecto."""

    nombre: str = "base"

    def __init__(self, frecuencia: str = "S", periodo_estacional: int = 52) -> None:
        self.frecuencia = frecuencia
        self.periodo_estacional = periodo_estacional
        self.entrenado_ = False

    # --------------------------------------------------------------- contrato
    @abstractmethod
    def entrenar(
        self, panel: pd.DataFrame, catalogo: pd.DataFrame | None = None
    ) -> "ModeloPronostico":
        """Ajusta el modelo con el panel historico de demanda."""

    @abstractmethod
    def predecir(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """Pronostica `horizonte` periodos hacia adelante desde el fin de `historia`."""

    # -------------------------------------------------------------- opcional
    def predecir_cuantiles(
        self, historia: pd.DataFrame, horizonte: int, cuantiles: list[float]
    ) -> pd.DataFrame:
        """Pronostico por cuantiles.

        La implementacion por defecto usa la distribucion empirica de la demanda
        reciente de cada SKU, escalada por el pronostico puntual. Los modelos
        que estiman cuantiles de forma nativa sobrescriben este metodo.
        """
        puntual = self.predecir(historia, horizonte)
        ventana = max(self.periodo_estacional, 26)
        recientes = (
            historia.sort_values(esq.COL_FECHA)
            .groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA]
            .apply(lambda s: s.tail(ventana).to_numpy())
        )
        resultado = puntual.copy()
        for nivel in cuantiles:
            columna = f"q{int(round(nivel * 100)):02d}"
            valores = []
            for sku, prediccion in zip(resultado[esq.COL_SKU], resultado[esq.COL_PREDICCION]):
                muestra = recientes.get(sku)
                if muestra is None or len(muestra) == 0:
                    valores.append(prediccion)
                    continue
                empirico = float(np.quantile(muestra, nivel))
                media = float(np.mean(muestra))
                # Se reescala el cuantil empirico al nivel del pronostico puntual.
                factor = prediccion / media if media > 1e-9 else 1.0
                valores.append(max(empirico * factor, 0.0))
            resultado[columna] = valores
        return resultado

    # ------------------------------------------------------------- auxiliares
    def _ultimos_origenes(self, historia: pd.DataFrame) -> pd.Series:
        """Ultima fecha observada por SKU."""
        return historia.groupby(esq.COL_SKU, observed=True)[esq.COL_FECHA].max()

    def _verificar_entrenado(self) -> None:
        if not self.entrenado_:
            raise RuntimeError(f"El modelo '{self.nombre}' no fue entrenado")

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"{self.__class__.__name__}(nombre={self.nombre!r})"
