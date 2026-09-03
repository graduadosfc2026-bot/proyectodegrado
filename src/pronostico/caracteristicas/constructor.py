"""Construccion de la matriz de caracteristicas del modelo global.

Reglas de diseno:

* **Sin fuga de informacion.** Toda variable de la fila con origen `t` se
  calcula unicamente con datos disponibles hasta `t` inclusive. Los objetivos
  son la demanda en `t + h`, con `h >= 1`.
* **Modelo global.** Una sola matriz para todos los SKU. El modelo aprende
  patrones compartidos (estacionalidad de campana, comportamiento por familia)
  y los aplica incluso a repuestos con poca historia propia.
* **Reproducible.** El orden y el tipo de las columnas quedan fijados al
  ajustar el constructor, de modo que entrenamiento y prediccion coincidan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datos import esquema as esq
from ..datos.preparacion import (
    COL_CENSURADO,
    COL_DIAS_PROMOCION,
    COL_DIAS_QUIEBRE,
    COL_N_CLIENTES,
    COL_N_TRANSACCIONES,
    COL_PRECIO_MEDIO,
)
from ..utilidades.registro_log import obtener_logger
from .calendario import caracteristicas_calendario

logger = obtener_logger(__name__)

EPS = 1e-9


def _periodos_desde_evento(evento: np.ndarray) -> np.ndarray:
    """Periodos transcurridos desde el ultimo `True` (0 si ocurre en el periodo).

    Devuelve NaN mientras no haya ocurrido ningun evento.
    """
    indices = np.arange(len(evento), dtype=float)
    ultimos = np.where(evento, indices, np.nan)
    ultimos = pd.Series(ultimos).ffill().to_numpy()
    return indices - ultimos


@dataclass
class ConstructorCaracteristicas:
    """Genera la matriz de predictores a partir del panel de demanda."""

    rezagos: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8, 13, 26, 52])
    ventanas_moviles: list[int] = field(default_factory=lambda: [4, 8, 13, 26, 52])
    periodo_estacional: int = 52
    ordenes_fourier: int = 3
    usar_calendario_agricola: bool = True
    usar_atributos_sku: bool = True

    columnas_: list[str] = field(default_factory=list, repr=False)
    columnas_categoricas_: list[str] = field(default_factory=list, repr=False)
    categorias_: dict[str, list[str]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------- bloques
    def _bloque_historia(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Rezagos, estadisticos moviles y variables de intermitencia."""
        agrupado = panel.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA]
        bloques: dict[str, pd.Series] = {"demanda_actual": panel[esq.COL_DEMANDA].astype(float)}

        for k in self.rezagos:
            bloques[f"rezago_{k}"] = agrupado.shift(k)

        for ventana in self.ventanas_moviles:
            movil = agrupado.rolling(ventana, min_periods=max(2, ventana // 4))
            bloques[f"media_movil_{ventana}"] = movil.mean().reset_index(level=0, drop=True)
            bloques[f"desvio_movil_{ventana}"] = movil.std().reset_index(level=0, drop=True)
            bloques[f"max_movil_{ventana}"] = movil.max().reset_index(level=0, drop=True)
            bloques[f"prop_ceros_{ventana}"] = (
                panel.assign(_cero=(panel[esq.COL_DEMANDA] <= 0).astype(float))
                .groupby(esq.COL_SKU, observed=True)["_cero"]
                .rolling(ventana, min_periods=max(2, ventana // 4))
                .mean()
                .reset_index(level=0, drop=True)
            )

        resultado = pd.DataFrame(bloques, index=panel.index)

        # Razon entre el nivel reciente y el nivel de largo plazo: mide impulso.
        corta, larga = min(self.ventanas_moviles), max(self.ventanas_moviles)
        if corta != larga:
            resultado["impulso_corto_largo"] = resultado[f"media_movil_{corta}"] / (
                resultado[f"media_movil_{larga}"] + EPS
            )
        # Dispersion relativa reciente: separa series estables de erraticas.
        media_media = resultado[f"media_movil_{larga}"]
        resultado["cv_movil"] = resultado[f"desvio_movil_{larga}"] / (media_media + EPS)

        # Variables de intermitencia (Croston): cada cuanto hay demanda y de que
        # tamano es cuando ocurre.
        hay_demanda = panel[esq.COL_DEMANDA].to_numpy() > 0
        antiguedad, desde_demanda = [], []
        for _, indices in panel.groupby(esq.COL_SKU, observed=True, sort=False).indices.items():
            orden = np.sort(indices)
            antiguedad.append(pd.Series(np.arange(len(orden), dtype=float), index=orden))
            desde_demanda.append(
                pd.Series(_periodos_desde_evento(hay_demanda[orden]), index=orden)
            )
        resultado["antiguedad_sku"] = pd.concat(antiguedad).sort_index().to_numpy()
        resultado["periodos_sin_demanda"] = pd.concat(desde_demanda).sort_index().to_numpy()

        positivos = panel[esq.COL_DEMANDA].where(panel[esq.COL_DEMANDA] > 0)
        grupo_pos = positivos.groupby(panel[esq.COL_SKU], observed=True)
        resultado["tamano_medio_pedido"] = (
            grupo_pos.expanding().mean().reset_index(level=0, drop=True)
        )
        resultado["frecuencia_demanda_acum"] = (
            panel.assign(_p=(panel[esq.COL_DEMANDA] > 0).astype(float))
            .groupby(esq.COL_SKU, observed=True)["_p"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        # Intervalo medio entre demandas (ADI) acumulado hasta el periodo actual.
        resultado["adi_acumulado"] = 1.0 / (resultado["frecuencia_demanda_acum"] + EPS)
        return resultado

    def _bloque_contexto(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Precio, promociones, quiebres y actividad comercial del periodo."""
        resultado = pd.DataFrame(index=panel.index)
        for columna in (
            COL_N_TRANSACCIONES,
            COL_N_CLIENTES,
            COL_DIAS_PROMOCION,
            COL_DIAS_QUIEBRE,
            COL_CENSURADO,
        ):
            if columna in panel.columns:
                resultado[columna] = panel[columna].astype(float)
                resultado[f"{columna}_media_13"] = (
                    panel.groupby(esq.COL_SKU, observed=True)[columna]
                    .rolling(13, min_periods=3)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        if COL_PRECIO_MEDIO in panel.columns:
            precio = panel[COL_PRECIO_MEDIO].astype(float)
            referencia = (
                panel.groupby(esq.COL_SKU, observed=True)[COL_PRECIO_MEDIO]
                .rolling(self.periodo_estacional, min_periods=4)
                .mean()
                .reset_index(level=0, drop=True)
            )
            resultado["precio_medio"] = precio
            resultado["precio_relativo"] = precio / (referencia + EPS)
            resultado["variacion_precio"] = (
                panel.groupby(esq.COL_SKU, observed=True)[COL_PRECIO_MEDIO].pct_change(4)
            )
        return resultado

    def _bloque_transversal(self, panel: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
        """Demanda agregada de la familia y de la maquina en el mismo periodo.

        Es informacion conocida en el origen `t` (se observa en todos los SKU a
        la vez) y aporta la senal de mercado que una serie individual no tiene.
        """
        resultado = pd.DataFrame(index=panel.index)
        if catalogo is None:
            return resultado
        atributos = catalogo.set_index(esq.COL_SKU)
        for columna, alias in ((esq.COL_FAMILIA, "familia"), (esq.COL_MAQUINA, "maquina")):
            if columna not in atributos.columns:
                continue
            clave = panel[esq.COL_SKU].map(atributos[columna])
            medias = panel.groupby([clave, panel[esq.COL_FECHA]], observed=True)[
                esq.COL_DEMANDA
            ].transform("mean")
            resultado[f"demanda_media_{alias}"] = medias.astype(float)
            resultado[f"ratio_vs_{alias}"] = panel[esq.COL_DEMANDA].astype(float) / (
                medias.astype(float) + EPS
            )
        return resultado

    def _bloque_atributos(self, panel: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
        """Atributos estaticos del catalogo de repuestos."""
        resultado = pd.DataFrame(index=panel.index)
        if catalogo is None or not self.usar_atributos_sku:
            return resultado
        atributos = catalogo.set_index(esq.COL_SKU)
        for columna in esq.ATRIBUTOS_NUMERICOS_SKU:
            if columna in atributos.columns:
                resultado[columna] = (
                    panel[esq.COL_SKU].map(atributos[columna]).astype(float).to_numpy()
                )
        for columna in esq.ATRIBUTOS_CATEGORICOS_SKU:
            if columna in atributos.columns:
                resultado[columna] = (
                    panel[esq.COL_SKU].map(atributos[columna]).astype(str).to_numpy()
                )
        return resultado

    # -------------------------------------------------------------- interfaz
    def construir(
        self, panel: pd.DataFrame, catalogo: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Devuelve el panel con las columnas de identificacion y los predictores."""
        if panel.empty:
            raise ValueError("El panel de demanda esta vacio")
        base = panel.sort_values([esq.COL_SKU, esq.COL_FECHA]).reset_index(drop=True)

        calendario = caracteristicas_calendario(
            base[esq.COL_FECHA],
            ordenes_fourier=self.ordenes_fourier,
            usar_calendario_agricola=self.usar_calendario_agricola,
        )
        calendario.index = base.index

        bloques = [
            base[[esq.COL_SKU, esq.COL_FECHA, esq.COL_DEMANDA]],
            self._bloque_historia(base),
            self._bloque_contexto(base),
            self._bloque_transversal(base, catalogo),
            calendario,
            self._bloque_atributos(base, catalogo),
        ]
        matriz = pd.concat(bloques, axis=1)
        matriz = matriz.loc[:, ~matriz.columns.duplicated()]
        return matriz.replace([np.inf, -np.inf], np.nan)

    def ajustar(self, matriz: pd.DataFrame) -> "ConstructorCaracteristicas":
        """Fija el orden de columnas y las categorias vistas en entrenamiento."""
        excluidas = {esq.COL_SKU, esq.COL_FECHA, esq.COL_DEMANDA}
        self.columnas_ = [c for c in matriz.columns if c not in excluidas]
        # Se consideran categoricas todas las columnas no numericas (texto o
        # `category`), independientemente de como pandas represente las cadenas.
        self.columnas_categoricas_ = [
            c for c in self.columnas_ if not pd.api.types.is_numeric_dtype(matriz[c])
        ]
        self.categorias_ = {
            c: sorted(matriz[c].dropna().astype(str).unique().tolist())
            for c in self.columnas_categoricas_
        }
        logger.info(
            "Constructor ajustado: %d predictores (%d categoricos)",
            len(self.columnas_),
            len(self.columnas_categoricas_),
        )
        return self

    def transformar(self, matriz: pd.DataFrame) -> pd.DataFrame:
        """Alinea una matriz al esquema fijado en `ajustar`."""
        if not self.columnas_:
            raise RuntimeError("El constructor no fue ajustado: llame primero a `ajustar`")
        faltantes = [c for c in self.columnas_ if c not in matriz.columns]
        if faltantes:
            raise ValueError(f"Faltan predictores en la matriz: {faltantes[:10]}")
        salida = matriz[self.columnas_].copy()
        for columna, categorias in self.categorias_.items():
            salida[columna] = pd.Categorical(salida[columna].astype(str), categories=categorias)
        for columna in salida.columns:
            if columna not in self.categorias_:
                salida[columna] = pd.to_numeric(salida[columna], errors="coerce").astype("float32")
        return salida

    def ajustar_transformar(self, matriz: pd.DataFrame) -> pd.DataFrame:
        """Atajo de `ajustar` seguido de `transformar`."""
        return self.ajustar(matriz).transformar(matriz)


def construir_objetivos(matriz: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Objetivos futuros `y_h = demanda(t + h)` para h = 1..horizonte.

    Las filas cuyo objetivo cae fuera del historico quedan como NaN y se
    descartan al entrenar cada horizonte.
    """
    if horizonte < 1:
        raise ValueError("El horizonte debe ser mayor o igual a 1")
    agrupado = matriz.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA]
    objetivos = {f"y_{h}": agrupado.shift(-h) for h in range(1, horizonte + 1)}
    return pd.DataFrame(objetivos, index=matriz.index)
