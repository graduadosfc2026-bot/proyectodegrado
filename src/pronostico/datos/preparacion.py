"""Transformacion de movimientos transaccionales en un panel de series.

El pipeline necesita un panel regular (SKU x periodo) sin huecos: los periodos
sin ventas son demanda cero, no datos faltantes. Ese detalle es critico en
repuestos, donde la mayoria de los periodos son ceros legitimos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utilidades.registro_log import obtener_logger
from . import esquema as esq

logger = obtener_logger(__name__)

COL_N_TRANSACCIONES = "n_transacciones"
COL_N_CLIENTES = "n_clientes"
COL_PRECIO_MEDIO = "precio_medio"
COL_DIAS_PROMOCION = "dias_promocion"
COL_DIAS_QUIEBRE = "dias_quiebre"
COL_DEMANDA_OBSERVADA = "demanda_observada"
COL_CENSURADO = "censurado"


def _inicio_periodo(fechas: pd.Series, frecuencia: str) -> pd.Series:
    """Lleva cada fecha al primer dia de su periodo (semana lunes o mes)."""
    periodos = fechas.dt.to_period(_periodo_pandas(frecuencia))
    return periodos.dt.start_time


def _periodo_pandas(frecuencia: str) -> str:
    """Traduce la frecuencia de configuracion al alias de `pandas.Period`."""
    normalizada = frecuencia.upper()
    # `pandas` nombra los periodos semanales por su dia de cierre: "W-SUN" es la
    # semana de lunes a domingo, que es la convencion de trabajo del proyecto.
    if normalizada in {"S", "W", "W-SUN", "W-MON", "SEMANAL"}:
        return "W-SUN"
    if normalizada in {"M", "MS", "MENSUAL"}:
        return "M"
    if normalizada in {"D", "DIARIO"}:
        return "D"
    raise ValueError(f"Frecuencia no soportada: {frecuencia}")


def frecuencia_offset(frecuencia: str) -> str:
    """Alias de `date_range` correspondiente a la frecuencia de trabajo."""
    periodo = _periodo_pandas(frecuencia)
    # El inicio de una semana "W-SUN" es un lunes: `date_range` la reproduce con
    # el ancla "W-MON".
    return {"W-SUN": "W-MON", "M": "MS", "D": "D"}[periodo]


def dias_por_periodo(frecuencia: str) -> float:
    """Cantidad media de dias que cubre un periodo de la frecuencia dada."""
    periodo = _periodo_pandas(frecuencia)
    return {"W-SUN": 7.0, "M": 30.44, "D": 1.0}[periodo]


def agregar_panel(
    movimientos: pd.DataFrame,
    frecuencia: str = "S",
    columna_objetivo: str = esq.COL_CANTIDAD,
) -> pd.DataFrame:
    """Agrega los movimientos a un panel SKU x periodo.

    Ademas de la demanda calcula variables de contexto (numero de operaciones,
    clientes distintos, precio medio, dias en promocion y dias con quiebre de
    stock) que luego se usan como predictores rezagados.
    """
    datos = esq.validar_movimientos(movimientos)
    datos = datos.assign(_periodo=_inicio_periodo(datos[esq.COL_FECHA], frecuencia))

    agregaciones: dict[str, pd.NamedAgg] = {
        esq.COL_DEMANDA: pd.NamedAgg(column=columna_objetivo, aggfunc="sum"),
    }
    panel = datos.groupby([esq.COL_SKU, "_periodo"], observed=True).agg(**agregaciones)

    # Contadores auxiliares calculados solo sobre las lineas con venta efectiva.
    con_venta = datos.loc[datos[columna_objetivo] > 0]
    if not con_venta.empty:
        auxiliares = con_venta.groupby([esq.COL_SKU, "_periodo"], observed=True).agg(
            **{
                COL_N_TRANSACCIONES: pd.NamedAgg(column=columna_objetivo, aggfunc="size"),
                COL_N_CLIENTES: pd.NamedAgg(
                    column=esq.COL_CLIENTE if esq.COL_CLIENTE in datos.columns else esq.COL_SKU,
                    aggfunc="nunique",
                ),
            }
        )
        panel = panel.join(auxiliares, how="left")

    if esq.COL_PRECIO in datos.columns:
        precios = (
            datos.loc[datos[esq.COL_PRECIO].notna()]
            .groupby([esq.COL_SKU, "_periodo"], observed=True)[esq.COL_PRECIO]
            .mean()
            .rename(COL_PRECIO_MEDIO)
        )
        panel = panel.join(precios, how="left")

    for columna_origen, columna_destino in (
        (esq.COL_PROMOCION, COL_DIAS_PROMOCION),
        (esq.COL_QUIEBRE, COL_DIAS_QUIEBRE),
    ):
        if columna_origen in datos.columns:
            marcas = (
                datos.assign(_v=pd.to_numeric(datos[columna_origen], errors="coerce").fillna(0))
                .groupby([esq.COL_SKU, "_periodo"], observed=True)["_v"]
                .apply(lambda s: float((s > 0).sum()))
                .rename(columna_destino)
            )
            panel = panel.join(marcas, how="left")

    panel = panel.reset_index().rename(columns={"_periodo": esq.COL_FECHA})
    return panel.sort_values([esq.COL_SKU, esq.COL_FECHA]).reset_index(drop=True)


def completar_grilla(
    panel: pd.DataFrame,
    frecuencia: str = "S",
    fecha_fin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rellena los periodos sin movimientos con demanda cero.

    Cada SKU arranca en su primer periodo con demanda registrada (antes de eso
    el repuesto no existia en el surtido) y termina en `fecha_fin`, por defecto
    el ultimo periodo observado en todo el panel.
    """
    if panel.empty:
        return panel.copy()

    offset = frecuencia_offset(frecuencia)
    fin_global = pd.Timestamp(fecha_fin) if fecha_fin is not None else panel[esq.COL_FECHA].max()

    partes = []
    for sku, grupo in panel.groupby(esq.COL_SKU, observed=True, sort=True):
        inicio = grupo[esq.COL_FECHA].min()
        if inicio > fin_global:
            continue
        grilla = pd.DataFrame(
            {esq.COL_FECHA: pd.date_range(inicio, fin_global, freq=offset)}
        )
        completo = grilla.merge(
            grupo.drop(columns=[esq.COL_SKU]), on=esq.COL_FECHA, how="left"
        )
        completo[esq.COL_SKU] = sku
        partes.append(completo)

    resultado = pd.concat(partes, ignore_index=True)
    columnas_conteo = [
        esq.COL_DEMANDA,
        COL_N_TRANSACCIONES,
        COL_N_CLIENTES,
        COL_DIAS_PROMOCION,
        COL_DIAS_QUIEBRE,
    ]
    for columna in columnas_conteo:
        if columna in resultado.columns:
            resultado[columna] = resultado[columna].fillna(0.0).astype(float)
    if COL_PRECIO_MEDIO in resultado.columns:
        # El precio no observado se arrastra desde el ultimo periodo con venta.
        resultado[COL_PRECIO_MEDIO] = resultado.groupby(esq.COL_SKU, observed=True)[
            COL_PRECIO_MEDIO
        ].ffill()

    orden = [esq.COL_SKU, esq.COL_FECHA] + [
        c for c in resultado.columns if c not in (esq.COL_SKU, esq.COL_FECHA)
    ]
    return resultado[orden].sort_values([esq.COL_SKU, esq.COL_FECHA]).reset_index(drop=True)


def marcar_censura(panel: pd.DataFrame, umbral_dias: float = 3.0) -> pd.DataFrame:
    """Marca los periodos con demanda potencialmente censurada por falta de stock.

    No se imputa la demanda perdida: se deja la marca disponible para que el
    modelo la use como variable y para poder excluir esos periodos de las
    metricas si el analista lo decide.
    """
    resultado = panel.copy()
    if COL_DIAS_QUIEBRE in resultado.columns:
        resultado[COL_CENSURADO] = (resultado[COL_DIAS_QUIEBRE] >= umbral_dias).astype(int)
    else:
        resultado[COL_CENSURADO] = 0
    return resultado


def filtrar_skus(
    panel: pd.DataFrame,
    min_periodos: int = 52,
    min_demanda_total: float = 12.0,
) -> pd.DataFrame:
    """Descarta SKU sin historia suficiente para entrenar y validar."""
    if panel.empty:
        return panel.copy()
    resumen = panel.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA].agg(
        periodos="size", total="sum"
    )
    validos = resumen.index[
        (resumen["periodos"] >= min_periodos) & (resumen["total"] >= min_demanda_total)
    ]
    descartados = resumen.shape[0] - len(validos)
    if descartados:
        logger.info(
            "Se descartan %d SKU por historia o volumen insuficiente (quedan %d)",
            descartados,
            len(validos),
        )
    return panel.loc[panel[esq.COL_SKU].isin(validos)].reset_index(drop=True)


def winsorizar_demanda(panel: pd.DataFrame, cuantil: float | None = 0.995) -> pd.DataFrame:
    """Recorta picos extremos de demanda por SKU al cuantil indicado.

    Amortigua pedidos atipicos (por ejemplo, una compra unica de un cliente
    grande) que de otro modo dominan el entrenamiento sin ser repetibles.
    """
    if cuantil is None or panel.empty:
        return panel.copy()
    resultado = panel.copy()
    topes = resultado.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA].transform(
        lambda s: s.quantile(cuantil)
    )
    topes = topes.where(topes > 0, resultado[esq.COL_DEMANDA])
    resultado[COL_DEMANDA_OBSERVADA] = resultado[esq.COL_DEMANDA]
    resultado[esq.COL_DEMANDA] = np.minimum(resultado[esq.COL_DEMANDA], topes)
    return resultado


def clasificar_abc_xyz(
    panel: pd.DataFrame,
    catalogo: pd.DataFrame | None = None,
    cortes_abc: tuple[float, float] = (0.8, 0.95),
    cortes_xyz: tuple[float, float] = (0.5, 1.0),
) -> pd.DataFrame:
    """Clasifica cada SKU por valor (ABC) y por variabilidad de demanda (XYZ).

    * ABC: participacion acumulada en el valor total de la demanda.
    * XYZ: coeficiente de variacion de la demanda por periodo.

    Se agrega ademas el par (ADI, CV2) de Syntetos-Boylan-Croston, que define
    el regimen de la serie: suave, intermitente, erratica o grumosa. Ese
    regimen determina que familia de modelos es apropiada para cada SKU.
    """
    if panel.empty:
        return pd.DataFrame()

    agrupado = panel.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA]
    resumen = pd.DataFrame(
        {
            "demanda_total": agrupado.sum(),
            "demanda_media": agrupado.mean(),
            "desvio": agrupado.std(ddof=0).fillna(0.0),
            "periodos": agrupado.size(),
            "periodos_con_demanda": agrupado.apply(lambda s: int((s > 0).sum())),
        }
    )
    resumen["cv"] = np.where(
        resumen["demanda_media"] > 0, resumen["desvio"] / resumen["demanda_media"], np.inf
    )
    # ADI: intervalo medio entre demandas. CV2: variabilidad del tamano del pedido.
    resumen["adi"] = np.where(
        resumen["periodos_con_demanda"] > 0,
        resumen["periodos"] / resumen["periodos_con_demanda"],
        np.inf,
    )
    tamanos = panel.loc[panel[esq.COL_DEMANDA] > 0].groupby(esq.COL_SKU, observed=True)[
        esq.COL_DEMANDA
    ]
    cv2 = (tamanos.std(ddof=0) / tamanos.mean()) ** 2
    resumen["cv2"] = cv2.reindex(resumen.index).fillna(0.0)

    resumen["regimen"] = np.select(
        [
            (resumen["adi"] < 1.32) & (resumen["cv2"] < 0.49),
            (resumen["adi"] >= 1.32) & (resumen["cv2"] < 0.49),
            (resumen["adi"] < 1.32) & (resumen["cv2"] >= 0.49),
        ],
        ["suave", "intermitente", "erratica"],
        default="grumosa",
    )

    valor_unitario = pd.Series(1.0, index=resumen.index)
    if catalogo is not None and esq.COL_COSTO in catalogo.columns:
        catalogo_valido = esq.validar_catalogo(catalogo).set_index(esq.COL_SKU)
        valor_unitario = (
            catalogo_valido[esq.COL_COSTO].reindex(resumen.index).fillna(1.0).astype(float)
        )
    resumen["valor_demanda"] = resumen["demanda_total"] * valor_unitario

    ordenado = resumen.sort_values("valor_demanda", ascending=False)
    total_valor = float(ordenado["valor_demanda"].sum())
    acumulado = (
        ordenado["valor_demanda"].cumsum() / total_valor if total_valor > 0 else
        pd.Series(np.linspace(0, 1, len(ordenado)), index=ordenado.index)
    )
    ordenado["clase_abc"] = np.select(
        [acumulado <= cortes_abc[0], acumulado <= cortes_abc[1]], ["A", "B"], default="C"
    )
    ordenado["clase_xyz"] = np.select(
        [ordenado["cv"] <= cortes_xyz[0], ordenado["cv"] <= cortes_xyz[1]],
        ["X", "Y"],
        default="Z",
    )
    ordenado["clase_abc_xyz"] = ordenado["clase_abc"] + ordenado["clase_xyz"]
    return ordenado.reset_index()


def preparar_panel(
    movimientos: pd.DataFrame,
    frecuencia: str = "S",
    columna_objetivo: str = esq.COL_CANTIDAD,
    min_periodos: int = 52,
    min_demanda_total: float = 12.0,
    winsorizar_cuantil: float | None = 0.995,
) -> pd.DataFrame:
    """Ejecuta la preparacion completa: agregar, completar, marcar y filtrar."""
    panel = agregar_panel(movimientos, frecuencia=frecuencia, columna_objetivo=columna_objetivo)
    panel = completar_grilla(panel, frecuencia=frecuencia)
    panel = marcar_censura(panel)
    panel = filtrar_skus(panel, min_periodos=min_periodos, min_demanda_total=min_demanda_total)
    panel = winsorizar_demanda(panel, cuantil=winsorizar_cuantil)
    logger.info(
        "Panel preparado: %d filas, %d SKU, periodos %s a %s",
        panel.shape[0],
        panel[esq.COL_SKU].nunique() if not panel.empty else 0,
        panel[esq.COL_FECHA].min().date() if not panel.empty else "-",
        panel[esq.COL_FECHA].max().date() if not panel.empty else "-",
    )
    return panel
