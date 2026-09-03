"""Modelo global de arboles con impulso de gradiente (gradient boosting).

Es el modelo principal del proyecto. Sus decisiones de diseno:

* **Global.** Un unico modelo entrenado con todos los SKU a la vez. Los
  repuestos comparten estacionalidad de campana y comportamiento por familia,
  asi que un modelo global aprende de los SKU con mucha historia y transfiere
  ese conocimiento a los de baja rotacion, donde una serie individual no
  alcanza para estimar nada.
* **Estrategia directa multi-horizonte.** Un estimador por horizonte `h`, en
  lugar de realimentar el pronostico. Evita la acumulacion de error de los
  metodos recursivos y permite que cada horizonte use su propia combinacion de
  variables.
* **Perdida de Poisson.** La demanda de repuestos son conteos no negativos con
  muchos ceros. La perdida de Poisson modela justamente eso y garantiza
  predicciones positivas, a diferencia del error cuadratico.
* **Cuantiles nativos.** Ademas de la media condicional se estiman los
  cuantiles altos, que son los que realmente dimensionan el stock de
  seguridad.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..caracteristicas.constructor import ConstructorCaracteristicas, construir_objetivos
from ..datos import esquema as esq
from ..utilidades.registro_log import obtener_logger
from .base import COLUMNAS_PRONOSTICO, ModeloPronostico, esqueleto_pronostico

logger = obtener_logger(__name__)

HIPERPARAMETROS_POR_DEFECTO: dict[str, Any] = {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
}


def etiqueta_cuantil(nivel: float) -> str:
    """Nombre de columna asociado a un nivel de cuantil (0.95 -> 'q95')."""
    return f"q{int(round(nivel * 100)):02d}"


class GBRTGlobal(ModeloPronostico):
    """Modelo global multi-horizonte basado en `HistGradientBoostingRegressor`."""

    nombre = "gbrt_global"

    def __init__(
        self,
        horizonte: int = 13,
        cuantiles: list[float] | None = None,
        hiperparametros: dict[str, Any] | None = None,
        constructor: ConstructorCaracteristicas | None = None,
        decaimiento_temporal: float = 1.0,
        semilla: int = 42,
        frecuencia: str = "S",
        periodo_estacional: int = 52,
    ) -> None:
        super().__init__(frecuencia=frecuencia, periodo_estacional=periodo_estacional)
        self.horizonte = horizonte
        self.cuantiles = list(cuantiles or [])
        self.hiperparametros = {**HIPERPARAMETROS_POR_DEFECTO, **(hiperparametros or {})}
        self.constructor = constructor or ConstructorCaracteristicas(
            periodo_estacional=periodo_estacional
        )
        self.decaimiento_temporal = decaimiento_temporal
        self.semilla = semilla

        self.modelos_: dict[int, HistGradientBoostingRegressor] = {}
        self.modelos_cuantil_: dict[tuple[int, float], HistGradientBoostingRegressor] = {}
        self.catalogo_: pd.DataFrame | None = None
        self.filas_entrenamiento_: dict[int, int] = {}

    # ------------------------------------------------------------- interno
    def _crear_estimador(
        self, perdida: str, cuantil: float | None = None
    ) -> HistGradientBoostingRegressor:
        """Instancia un estimador con la perdida pedida y las columnas categoricas."""
        parametros = dict(self.hiperparametros)
        parametros.update(
            loss=perdida,
            random_state=self.semilla,
            categorical_features=self.constructor.columnas_categoricas_ or None,
        )
        if cuantil is not None:
            parametros["quantile"] = cuantil
        return HistGradientBoostingRegressor(**parametros)

    def _pesos_recencia(self, fechas: pd.Series) -> np.ndarray | None:
        """Pesos que priorizan las observaciones recientes (None si no hay decaimiento).

        Util cuando el surtido o la cartera de clientes cambiaron: la historia
        antigua sigue aportando forma estacional pero pesa menos en el nivel.
        """
        if self.decaimiento_temporal >= 1.0:
            return None
        maxima = fechas.max()
        antiguedad_dias = (maxima - fechas).dt.days.to_numpy(dtype=float)
        periodos = antiguedad_dias / 7.0 if self.frecuencia.upper().startswith("S") else (
            antiguedad_dias / 30.44
        )
        return np.power(self.decaimiento_temporal, periodos)

    # ------------------------------------------------------------ contrato
    def entrenar(
        self, panel: pd.DataFrame, catalogo: pd.DataFrame | None = None
    ) -> "GBRTGlobal":
        """Ajusta un estimador por horizonte (y por cuantil solicitado)."""
        self.catalogo_ = catalogo
        matriz = self.constructor.construir(panel, catalogo)
        X = self.constructor.ajustar_transformar(matriz)
        Y = construir_objetivos(matriz, self.horizonte)
        pesos = self._pesos_recencia(matriz[esq.COL_FECHA])

        self.modelos_.clear()
        self.modelos_cuantil_.clear()
        for h in range(1, self.horizonte + 1):
            objetivo = Y[f"y_{h}"]
            valido = objetivo.notna().to_numpy()
            if valido.sum() < 50:
                logger.warning(
                    "Horizonte %d con solo %d observaciones: se omite", h, int(valido.sum())
                )
                continue
            X_h, y_h = X.loc[valido], objetivo.loc[valido].to_numpy(dtype=float)
            pesos_h = pesos[valido] if pesos is not None else None
            self.filas_entrenamiento_[h] = int(valido.sum())

            modelo = self._crear_estimador("poisson")
            modelo.fit(X_h, y_h, sample_weight=pesos_h)
            self.modelos_[h] = modelo

            for nivel in self.cuantiles:
                modelo_q = self._crear_estimador("quantile", cuantil=nivel)
                modelo_q.fit(X_h, y_h, sample_weight=pesos_h)
                self.modelos_cuantil_[(h, nivel)] = modelo_q

        if not self.modelos_:
            raise RuntimeError(
                "No se pudo entrenar ningun horizonte: la historia disponible es insuficiente"
            )
        self.entrenado_ = True
        logger.info(
            "Modelo global entrenado: %d horizontes, %d cuantiles, %d filas en h=1",
            len(self.modelos_),
            len(self.cuantiles),
            self.filas_entrenamiento_.get(1, 0),
        )
        return self

    def _filas_origen(self, historia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ultima fila de caracteristicas de cada SKU (el origen del pronostico)."""
        matriz = self.constructor.construir(historia, self.catalogo_)
        ultimas = matriz.groupby(esq.COL_SKU, observed=True).tail(1).reset_index(drop=True)
        return ultimas, self.constructor.transformar(ultimas)

    def predecir(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """Pronostico puntual (media condicional) para h = 1..horizonte."""
        self._verificar_entrenado()
        if horizonte > self.horizonte:
            raise ValueError(
                f"El modelo fue entrenado hasta h={self.horizonte}, se pidio h={horizonte}"
            )
        if historia.empty:
            return pd.DataFrame(columns=COLUMNAS_PRONOSTICO)

        ultimas, X = self._filas_origen(historia)
        origenes = ultimas.set_index(esq.COL_SKU)[esq.COL_FECHA]
        esqueleto = esqueleto_pronostico(origenes, horizonte, self.frecuencia)

        predicciones = np.zeros((len(ultimas), horizonte), dtype=float)
        for h in range(1, horizonte + 1):
            modelo = self.modelos_.get(h)
            predicciones[:, h - 1] = (
                modelo.predict(X) if modelo is not None else np.nan
            )
        mapa = pd.DataFrame(
            predicciones,
            index=pd.Index(ultimas[esq.COL_SKU], name=esq.COL_SKU),
            columns=pd.Index(range(1, horizonte + 1), name=esq.COL_HORIZONTE),
        )
        valores = mapa.stack().rename(esq.COL_PREDICCION).reset_index()
        esqueleto = esqueleto.merge(valores, on=[esq.COL_SKU, esq.COL_HORIZONTE], how="left")
        esqueleto[esq.COL_PREDICCION] = esqueleto[esq.COL_PREDICCION].clip(lower=0.0)
        return esqueleto[COLUMNAS_PRONOSTICO]

    def predecir_cuantiles(
        self, historia: pd.DataFrame, horizonte: int, cuantiles: list[float]
    ) -> pd.DataFrame:
        """Pronostico puntual mas los cuantiles estimados de forma nativa."""
        self._verificar_entrenado()
        resultado = self.predecir(historia, horizonte)
        if resultado.empty:
            return resultado
        faltantes = [q for q in cuantiles if not any(k[1] == q for k in self.modelos_cuantil_)]
        if faltantes:
            raise ValueError(
                f"El modelo no fue entrenado para los cuantiles {faltantes}. "
                f"Disponibles: {sorted({k[1] for k in self.modelos_cuantil_})}"
            )

        ultimas, X = self._filas_origen(historia)
        for nivel in cuantiles:
            matriz = np.zeros((len(ultimas), horizonte), dtype=float)
            for h in range(1, horizonte + 1):
                modelo = self.modelos_cuantil_.get((h, nivel))
                matriz[:, h - 1] = modelo.predict(X) if modelo is not None else np.nan
            columna = etiqueta_cuantil(nivel)
            valores = (
                pd.DataFrame(
                    matriz,
                    index=pd.Index(ultimas[esq.COL_SKU], name=esq.COL_SKU),
                    columns=pd.Index(range(1, horizonte + 1), name=esq.COL_HORIZONTE),
                )
                .stack()
                .rename(columna)
                .reset_index()
            )
            resultado = resultado.merge(valores, on=[esq.COL_SKU, esq.COL_HORIZONTE], how="left")
            resultado[columna] = resultado[columna].clip(lower=0.0)

        # Los cuantiles se ordenan de forma monotona: un q90 no puede quedar por
        # debajo de un q80 por ruido de estimacion.
        columnas_q = [etiqueta_cuantil(q) for q in sorted(cuantiles)]
        if len(columnas_q) > 1:
            resultado[columnas_q] = np.maximum.accumulate(
                resultado[columnas_q].to_numpy(dtype=float), axis=1
            )
        return resultado

    # ------------------------------------------------------- interpretacion
    def importancia_permutacion(
        self,
        panel: pd.DataFrame,
        horizonte: int = 1,
        n_repeticiones: int = 3,
        max_filas: int = 5000,
    ) -> pd.DataFrame:
        """Importancia por permutacion de cada variable en un horizonte dado.

        Mide cuanto empeora el error al mezclar aleatoriamente una variable: es
        el sustento de la lectura de negocio del modelo (que impulsa la demanda).
        """
        self._verificar_entrenado()
        modelo = self.modelos_.get(horizonte)
        if modelo is None:
            raise ValueError(f"No hay modelo entrenado para el horizonte {horizonte}")

        matriz = self.constructor.construir(panel, self.catalogo_)
        X = self.constructor.transformar(matriz)
        y = construir_objetivos(matriz, horizonte)[f"y_{horizonte}"]
        valido = y.notna().to_numpy()
        X, y = X.loc[valido], y.loc[valido]
        if len(X) > max_filas:
            muestra = X.sample(max_filas, random_state=self.semilla).index
            X, y = X.loc[muestra], y.loc[muestra]

        resultado = permutation_importance(
            modelo,
            X,
            y,
            n_repeats=n_repeticiones,
            random_state=self.semilla,
            scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame(
                {
                    "variable": X.columns,
                    "importancia": resultado.importances_mean,
                    "desvio": resultado.importances_std,
                }
            )
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )
