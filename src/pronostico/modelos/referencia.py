"""Modelos de referencia (baselines).

Ningun modelo de aprendizaje automatico se justifica si no mejora a estos.
Se incluyen tanto los ingenuos clasicos como los metodos especificos de
demanda intermitente (Croston, SBA y TSB), que son el estandar de la industria
para repuestos y el punto de comparacion honesto en este dominio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datos import esquema as esq
from .base import COLUMNAS_PRONOSTICO, ModeloPronostico, esqueleto_pronostico


def _series_por_sku(historia: pd.DataFrame) -> dict[str, np.ndarray]:
    """Devuelve la serie de demanda ordenada de cada SKU."""
    ordenado = historia.sort_values([esq.COL_SKU, esq.COL_FECHA])
    return {
        str(sku): grupo[esq.COL_DEMANDA].to_numpy(dtype=float)
        for sku, grupo in ordenado.groupby(esq.COL_SKU, observed=True)
    }


class ModeloPorSerie(ModeloPronostico):
    """Base de los modelos que se calculan independientemente para cada SKU."""

    def entrenar(
        self, panel: pd.DataFrame, catalogo: pd.DataFrame | None = None
    ) -> "ModeloPorSerie":
        # Estos modelos no tienen parametros globales que aprender: la estimacion
        # se hace sobre la historia recibida en cada llamada a `predecir`.
        self.entrenado_ = True
        return self

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        """Devuelve el vector de `horizonte` pronosticos para una serie."""
        raise NotImplementedError

    def predecir(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        self._verificar_entrenado()
        if historia.empty:
            return pd.DataFrame(columns=COLUMNAS_PRONOSTICO)
        series = _series_por_sku(historia)
        esqueleto = esqueleto_pronostico(
            self._ultimos_origenes(historia), horizonte, self.frecuencia
        )
        predicciones = np.concatenate(
            [
                self._pronostico_serie(series[str(sku)], horizonte)
                for sku in esqueleto[esq.COL_SKU].unique()
            ]
        )
        esqueleto[esq.COL_PREDICCION] = np.clip(predicciones, 0.0, None)
        return esqueleto[COLUMNAS_PRONOSTICO]


class Naive(ModeloPorSerie):
    """Repite el ultimo valor observado en todo el horizonte."""

    nombre = "naive"

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        valor = serie[-1] if serie.size else 0.0
        return np.full(horizonte, valor, dtype=float)


class NaiveEstacional(ModeloPorSerie):
    """Repite el valor del mismo periodo de la campana anterior.

    Es la referencia natural en un negocio estacional: "el ano pasado, en esta
    misma semana, vendimos esto".
    """

    nombre = "naive_estacional"

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        m = self.periodo_estacional
        if serie.size < m:
            valor = float(np.mean(serie)) if serie.size else 0.0
            return np.full(horizonte, valor, dtype=float)
        ultima_temporada = serie[-m:]
        indices = (np.arange(horizonte)) % m
        return ultima_temporada[indices].astype(float)


class MediaMovil(ModeloPorSerie):
    """Media de los ultimos `ventana` periodos, constante en el horizonte."""

    nombre = "media_movil"

    def __init__(self, ventana: int = 13, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ventana = ventana

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        if serie.size == 0:
            return np.zeros(horizonte)
        valor = float(np.mean(serie[-self.ventana :]))
        return np.full(horizonte, valor, dtype=float)


class Croston(ModeloPorSerie):
    """Metodo de Croston para demanda intermitente.

    Descompone la serie en dos procesos suavizados exponencialmente: el tamano
    de los pedidos cuando ocurren y el intervalo entre pedidos. El pronostico
    por periodo es el cociente entre ambos.
    """

    nombre = "croston"
    correccion_sesgo = 1.0

    def __init__(self, alfa: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 < alfa <= 1.0:
            raise ValueError("alfa debe estar en (0, 1]")
        self.alfa = alfa

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        positivos = np.flatnonzero(serie > 0)
        if positivos.size == 0:
            return np.zeros(horizonte)
        if positivos.size == 1:
            tasa = serie[positivos[0]] / max(serie.size, 1)
            return np.full(horizonte, tasa * self.correccion_sesgo, dtype=float)

        tamano = float(serie[positivos[0]])
        intervalo = float(np.mean(np.diff(positivos))) or 1.0
        contador = 1
        for indice in range(positivos[0] + 1, serie.size):
            if serie[indice] > 0:
                tamano = self.alfa * serie[indice] + (1 - self.alfa) * tamano
                intervalo = self.alfa * contador + (1 - self.alfa) * intervalo
                contador = 1
            else:
                contador += 1
        tasa = tamano / max(intervalo, 1e-9) * self.correccion_sesgo
        return np.full(horizonte, tasa, dtype=float)


class SBA(Croston):
    """Croston con la correccion de sesgo de Syntetos-Boylan.

    Croston sobrestima sistematicamente la demanda; el factor (1 - alfa/2)
    corrige ese sesgo y suele mejorar el resultado en repuestos.
    """

    nombre = "sba"

    def __init__(self, alfa: float = 0.1, **kwargs) -> None:
        super().__init__(alfa=alfa, **kwargs)
        self.correccion_sesgo = 1.0 - self.alfa / 2.0


class TSB(ModeloPorSerie):
    """Metodo de Teunter-Syntetos-Babai.

    A diferencia de Croston, actualiza la probabilidad de demanda en todos los
    periodos (tambien en los ceros), por lo que reacciona a los repuestos que
    salen de circulacion en vez de quedar anclado en su ultimo pedido.
    """

    nombre = "tsb"

    def __init__(self, alfa: float = 0.1, beta: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alfa = alfa
        self.beta = beta

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        positivos = np.flatnonzero(serie > 0)
        if positivos.size == 0:
            return np.zeros(horizonte)
        tamano = float(serie[positivos[0]])
        probabilidad = float(np.mean(serie > 0))
        for valor in serie[positivos[0] + 1 :]:
            if valor > 0:
                tamano = self.alfa * valor + (1 - self.alfa) * tamano
                probabilidad = self.beta + (1 - self.beta) * probabilidad
            else:
                probabilidad = (1 - self.beta) * probabilidad
        return np.full(horizonte, probabilidad * tamano, dtype=float)


class MediaEstacional(ModeloPorSerie):
    """Media historica del mismo periodo estacional, con ajuste de nivel.

    Combina la forma estacional de toda la historia con el nivel reciente:
    es una referencia sorprendentemente fuerte en negocios de campana.
    """

    nombre = "media_estacional"

    def __init__(self, ventana_nivel: int = 13, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ventana_nivel = ventana_nivel

    def _pronostico_serie(self, serie: np.ndarray, horizonte: int) -> np.ndarray:
        m = self.periodo_estacional
        if serie.size < 2 * m:
            valor = float(np.mean(serie[-self.ventana_nivel :])) if serie.size else 0.0
            return np.full(horizonte, valor, dtype=float)

        posiciones = np.arange(serie.size) % m
        perfil = np.array(
            [serie[posiciones == p].mean() if (posiciones == p).any() else 0.0 for p in range(m)]
        )
        media_perfil = float(perfil.mean())
        if media_perfil <= 1e-9:
            return np.zeros(horizonte)
        nivel_reciente = float(np.mean(serie[-self.ventana_nivel :]))
        indices = (posiciones[-1] + 1 + np.arange(horizonte)) % m
        return perfil[indices] / media_perfil * nivel_reciente
