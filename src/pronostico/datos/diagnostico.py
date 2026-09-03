"""Diagnostico de aptitud de los datos antes de entrenar.

Al conectar los datos reales de la empresa, la primera pregunta no es que tan
bueno es el modelo sino si los datos alcanzan para entrenarlo. Este modulo
responde eso: cuanta historia hay, cuantos SKU sobreviven a los filtros, que
tan intermitente es la demanda y que columnas opcionales faltan.
"""

from __future__ import annotations

import pandas as pd

from ..utilidades.registro_log import obtener_logger
from . import esquema as esq
from .preparacion import (
    COL_DIAS_QUIEBRE,
    agregar_panel,
    clasificar_abc_xyz,
    completar_grilla,
    dias_por_periodo,
)

logger = obtener_logger(__name__)


def diagnosticar(
    movimientos: pd.DataFrame,
    catalogo: pd.DataFrame | None = None,
    frecuencia: str = "S",
    min_periodos: int = 52,
    min_demanda_total: float = 12.0,
    periodo_estacional: int = 52,
    horizonte: int = 13,
) -> dict[str, object]:
    """Evalua si los datos alcanzan para entrenar y devuelve un informe.

    El informe separa los **hallazgos** (hechos medidos) de las **advertencias**
    (cosas que degradan el modelo pero no lo impiden) y los **bloqueantes**
    (impiden entrenar). No modifica los datos.
    """
    informe: dict[str, object] = {"advertencias": [], "bloqueantes": []}
    advertencias: list[str] = informe["advertencias"]  # type: ignore[assignment]
    bloqueantes: list[str] = informe["bloqueantes"]  # type: ignore[assignment]

    validos = esq.validar_movimientos(movimientos)
    descartadas = len(movimientos) - len(validos)
    if descartadas:
        advertencias.append(
            f"Se descartaron {descartadas:,} filas por fecha, SKU o cantidad invalidos"
        )

    panel = completar_grilla(agregar_panel(validos, frecuencia=frecuencia), frecuencia=frecuencia)

    resumen = panel.groupby(esq.COL_SKU, observed=True)[esq.COL_DEMANDA].agg(
        periodos="size", total="sum"
    )
    aptos = resumen.index[
        (resumen["periodos"] >= min_periodos) & (resumen["total"] >= min_demanda_total)
    ]

    informe["movimientos"] = len(validos)
    informe["skus_en_movimientos"] = int(validos[esq.COL_SKU].nunique())
    informe["fecha_inicio"] = str(validos[esq.COL_FECHA].min().date())
    informe["fecha_fin"] = str(validos[esq.COL_FECHA].max().date())
    informe["periodos_totales"] = int(panel[esq.COL_FECHA].nunique())
    informe["skus_aptos"] = int(len(aptos))
    informe["skus_descartados"] = int(len(resumen) - len(aptos))
    informe["proporcion_periodos_en_cero"] = float((panel[esq.COL_DEMANDA] == 0).mean())

    # --- Historia disponible --------------------------------------------------
    periodos = int(informe["periodos_totales"])
    if periodos < min_periodos:
        bloqueantes.append(
            f"Solo hay {periodos} periodos de historia y el filtro exige {min_periodos}. "
            f"Reduzca `datos.min_periodos_historia` o consiga mas historico."
        )
    if periodos < 2 * periodo_estacional:
        anos = periodos / periodo_estacional
        advertencias.append(
            f"Hay {anos:.1f} anos de historia. Con menos de 2 los rezagos estacionales "
            f"(rezago_{periodo_estacional}) quedan vacios y el modelo no puede aprender "
            f"la campana agricola; con 3 o mas el pronostico mejora sensiblemente."
        )
    if periodos < horizonte * 3:
        advertencias.append(
            f"La historia ({periodos} periodos) es corta frente al horizonte ({horizonte}): "
            f"el backtesting tendra pocos origenes y su medicion sera inestable."
        )
    if not len(aptos):
        bloqueantes.append(
            "Ningun SKU supera los filtros de historia y volumen minimos: "
            "revise `datos.min_periodos_historia` y `datos.min_demanda_total`."
        )

    # --- Columnas opcionales que aportan senal --------------------------------
    for columna, motivo in (
        (esq.COL_QUIEBRE, "sin ella la demanda perdida entra como demanda cero y el "
                          "modelo aprende a comprar de menos justo en los repuestos "
                          "que mas faltan"),
        (esq.COL_PRECIO, "sin ella no se puede modelar el efecto del precio ni de las "
                         "promociones"),
        (esq.COL_CLIENTE, "sin ella no se cuenta la cantidad de clientes distintos por "
                          "periodo"),
    ):
        if columna not in validos.columns:
            advertencias.append(f"Falta la columna opcional '{columna}': {motivo}")

    if COL_DIAS_QUIEBRE in panel.columns:
        proporcion = float((panel[COL_DIAS_QUIEBRE] > 0).mean())
        informe["proporcion_periodos_con_quiebre"] = proporcion
        if proporcion > 0.15:
            advertencias.append(
                f"El {proporcion:.0%} de los periodos tuvo quiebre de stock: la demanda "
                f"observada esta muy censurada y el pronostico tendera a quedarse corto."
            )

    # --- Catalogo -------------------------------------------------------------
    if catalogo is None:
        advertencias.append(
            "No hay catalogo de SKU: se pierden familia, maquina y criticidad como "
            "variables, y el plan de reposicion usara el lead time por defecto."
        )
    else:
        catalogo_valido = esq.validar_catalogo(catalogo)
        informe["skus_en_catalogo"] = int(len(catalogo_valido))
        sin_ficha = sorted(set(aptos) - set(catalogo_valido[esq.COL_SKU]))
        informe["skus_sin_ficha_en_catalogo"] = len(sin_ficha)
        if sin_ficha:
            advertencias.append(
                f"{len(sin_ficha)} SKU con ventas no estan en el catalogo "
                f"(por ejemplo {sin_ficha[:3]}): quedaran sin atributos."
            )
        for columna in (esq.COL_LEAD_TIME, esq.COL_COSTO, esq.COL_MOQ):
            if columna not in catalogo_valido.columns:
                advertencias.append(
                    f"El catalogo no trae '{columna}': el plan de reposicion usara un "
                    f"valor por defecto para esa variable."
                )
        if esq.COL_LEAD_TIME in catalogo_valido.columns:
            lead_max = float(catalogo_valido[esq.COL_LEAD_TIME].max())
            periodos_lead = lead_max / dias_por_periodo(frecuencia)
            informe["lead_time_maximo_dias"] = lead_max
            if periodos_lead > horizonte:
                advertencias.append(
                    f"El lead time mas largo del catalogo es de {lead_max:.0f} dias "
                    f"({periodos_lead:.0f} periodos) y el horizonte es de {horizonte}: "
                    f"suba `modelo.horizonte` para cubrirlo o esos SKU se extrapolaran."
                )

    # --- Regimen de demanda ---------------------------------------------------
    panel_apto = panel.loc[panel[esq.COL_SKU].isin(aptos)]
    if not panel_apto.empty:
        clases = clasificar_abc_xyz(panel_apto, catalogo)
        informe["regimenes"] = clases["regimen"].value_counts().to_dict()
        informe["clases_abc"] = clases["clase_abc"].value_counts().to_dict()
        intermitentes = float((clases["regimen"] != "suave").mean())
        informe["proporcion_no_suave"] = intermitentes
        if intermitentes > 0.7:
            advertencias.append(
                f"El {intermitentes:.0%} de los SKU tiene demanda no suave. Es normal en "
                f"repuestos, pero espere un WAPE alto en ese grupo y lea las metricas "
                f"por regimen, no solo el promedio global."
            )

    informe["apto_para_entrenar"] = not bloqueantes
    return informe


def imprimir_diagnostico(informe: dict[str, object]) -> None:
    """Muestra el informe de diagnostico en un formato legible en consola."""
    claves_texto = [
        ("movimientos", "Movimientos validos"),
        ("skus_en_movimientos", "SKU con movimientos"),
        ("skus_en_catalogo", "SKU en el catalogo"),
        ("fecha_inicio", "Primera fecha"),
        ("fecha_fin", "Ultima fecha"),
        ("periodos_totales", "Periodos de historia"),
        ("skus_aptos", "SKU aptos para entrenar"),
        ("skus_descartados", "SKU descartados por los filtros"),
        ("skus_sin_ficha_en_catalogo", "SKU sin ficha en el catalogo"),
        ("lead_time_maximo_dias", "Lead time maximo (dias)"),
    ]
    print("\n=== Diagnostico de los datos ===")
    for clave, etiqueta in claves_texto:
        if clave in informe:
            valor = informe[clave]
            texto = f"{valor:,}" if isinstance(valor, (int, float)) else str(valor)
            print(f"  {etiqueta:34s}: {texto}")
    for clave, etiqueta in (
        ("proporcion_periodos_en_cero", "Periodos sin demanda"),
        ("proporcion_periodos_con_quiebre", "Periodos con quiebre de stock"),
        ("proporcion_no_suave", "SKU de demanda no suave"),
    ):
        if clave in informe:
            print(f"  {etiqueta:34s}: {float(informe[clave]):.1%}")
    for clave, etiqueta in (("regimenes", "Regimen"), ("clases_abc", "Clase ABC")):
        if clave in informe:
            print(f"  {etiqueta:34s}: {informe[clave]}")

    for titulo, clave in (("Bloqueantes", "bloqueantes"), ("Advertencias", "advertencias")):
        mensajes = informe.get(clave) or []
        if mensajes:
            print(f"\n--- {titulo} ---")
            for mensaje in mensajes:  # type: ignore[union-attr]
                print(f"  * {mensaje}")

    veredicto = (
        "Los datos alcanzan para entrenar."
        if informe.get("apto_para_entrenar")
        else "Los datos NO alcanzan para entrenar: resuelva los bloqueantes."
    )
    print(f"\n{veredicto}")
