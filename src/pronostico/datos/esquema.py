"""Esquema canonico de los datos de entrada y utilidades de validacion.

El pipeline trabaja con dos tablas:

* ``movimientos``: una fila por linea de venta / pedido de repuesto.
* ``catalogo``:    una fila por SKU con sus atributos maestros.

Mantener el esquema en un unico lugar permite conectar la fuente real de la
empresa (ERP, sistema de facturacion) cambiando solo el mapeo de columnas.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------- movimientos
COL_FECHA = "fecha"
COL_SKU = "sku"
COL_CANTIDAD = "cantidad"
COL_PRECIO = "precio_unitario"
COL_CLIENTE = "cliente_id"
COL_CANAL = "canal"
COL_PROMOCION = "promocion"
COL_QUIEBRE = "quiebre_stock"

COLUMNAS_MOVIMIENTOS_OBLIGATORIAS = [COL_FECHA, COL_SKU, COL_CANTIDAD]
COLUMNAS_MOVIMIENTOS_OPCIONALES = [
    COL_PRECIO,
    COL_CLIENTE,
    COL_CANAL,
    COL_PROMOCION,
    COL_QUIEBRE,
]

# ------------------------------------------------------------------ catalogo
COL_DESCRIPCION = "descripcion"
COL_FAMILIA = "familia"
COL_MAQUINA = "maquina"
COL_CRITICIDAD = "criticidad"
COL_COSTO = "costo_unitario"
COL_PRECIO_LISTA = "precio_lista"
COL_LEAD_TIME = "lead_time_dias"
COL_ORIGEN = "origen_proveedor"
COL_MOQ = "lote_minimo"

COLUMNAS_CATALOGO_OBLIGATORIAS = [COL_SKU]
ATRIBUTOS_CATEGORICOS_SKU = [COL_FAMILIA, COL_MAQUINA, COL_CRITICIDAD, COL_ORIGEN]
ATRIBUTOS_NUMERICOS_SKU = [COL_COSTO, COL_PRECIO_LISTA, COL_LEAD_TIME, COL_MOQ]

# ------------------------------------------------------------ serie agregada
COL_DEMANDA = "demanda"
COL_HORIZONTE = "h"
COL_OBJETIVO = "y"
COL_PREDICCION = "prediccion"
COL_ORIGEN_BACKTEST = "origen"
COL_MODELO = "modelo"


class ErrorEsquema(ValueError):
    """Se lanza cuando una tabla de entrada no cumple el esquema esperado."""


def validar_movimientos(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza la tabla de movimientos de venta.

    Comprueba las columnas obligatorias, convierte tipos y elimina filas
    inutilizables (fecha o SKU nulos). Devuelve una copia normalizada.
    """
    faltantes = [c for c in COLUMNAS_MOVIMIENTOS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ErrorEsquema(
            f"Faltan columnas obligatorias en movimientos: {faltantes}. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    resultado = df.copy()
    resultado[COL_FECHA] = pd.to_datetime(resultado[COL_FECHA], errors="coerce")
    resultado[COL_SKU] = resultado[COL_SKU].astype(str).str.strip()
    resultado[COL_CANTIDAD] = pd.to_numeric(resultado[COL_CANTIDAD], errors="coerce")

    invalidas = resultado[COL_FECHA].isna() | (resultado[COL_SKU] == "") | resultado[
        COL_CANTIDAD
    ].isna()
    resultado = resultado.loc[~invalidas].reset_index(drop=True)

    if resultado.empty:
        raise ErrorEsquema("La tabla de movimientos quedo vacia tras la validacion")

    # Las devoluciones (cantidades negativas) no son demanda: se llevan a cero.
    resultado[COL_CANTIDAD] = resultado[COL_CANTIDAD].clip(lower=0)
    return resultado


def validar_catalogo(df: pd.DataFrame) -> pd.DataFrame:
    """Valida la tabla de catalogo de SKU y normaliza sus tipos."""
    faltantes = [c for c in COLUMNAS_CATALOGO_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ErrorEsquema(f"Faltan columnas obligatorias en catalogo: {faltantes}")

    resultado = df.copy()
    resultado[COL_SKU] = resultado[COL_SKU].astype(str).str.strip()
    duplicados = resultado[COL_SKU].duplicated()
    if duplicados.any():
        repetidos = resultado.loc[duplicados, COL_SKU].unique().tolist()[:5]
        raise ErrorEsquema(f"El catalogo tiene SKU duplicados, por ejemplo: {repetidos}")

    for columna in ATRIBUTOS_NUMERICOS_SKU:
        if columna in resultado.columns:
            resultado[columna] = pd.to_numeric(resultado[columna], errors="coerce")
    for columna in ATRIBUTOS_CATEGORICOS_SKU:
        if columna in resultado.columns:
            resultado[columna] = resultado[columna].astype(str).str.strip()
    return resultado
