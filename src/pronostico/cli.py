"""Interfaz de linea de comandos del sistema de pronostico.

Uso tipico de punta a punta:

    python -m pronostico generar-datos
    python -m pronostico entrenar
    python -m pronostico predecir
    python -m pronostico reponer
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import Config, cargar_config
from .datos import esquema as esq
from .datos.diagnostico import diagnosticar, imprimir_diagnostico
from .datos.sintetico import generar_conjunto_datos
from .inventario.politica import resumen_politica
from .modelos.registro import modelos_disponibles
from .pipeline.entrenamiento import entrenar_pipeline
from .pipeline.prediccion import (
    cargar_artefacto,
    plan_de_reposicion,
    predecir_demanda,
    preparar_historia,
)
from .utilidades.persistencia import guardar_json, guardar_tabla, leer_tabla
from .utilidades.registro_log import configurar_logging, obtener_logger

logger = obtener_logger(__name__)

ARCHIVO_MOVIMIENTOS = "movimientos.csv"
ARCHIVO_CATALOGO = "catalogo_skus.csv"
ARCHIVO_PRONOSTICO = "pronostico.csv"
ARCHIVO_PLAN = "plan_reposicion.csv"
ARCHIVO_IMPORTANCIA = "importancia_variables.csv"
ARCHIVO_DIAGNOSTICO = "diagnostico_datos.json"


def _rutas_datos(config: Config, argumentos: argparse.Namespace) -> tuple[Path, Path]:
    """Resuelve las rutas de los archivos de entrada."""
    base = config.ruta_de("proyecto.directorio_datos_crudos")
    movimientos = Path(argumentos.movimientos) if argumentos.movimientos else base / ARCHIVO_MOVIMIENTOS
    catalogo = Path(argumentos.catalogo) if argumentos.catalogo else base / ARCHIVO_CATALOGO
    return movimientos, catalogo


def _cargar_entradas(config: Config, argumentos: argparse.Namespace):
    """Lee movimientos y catalogo, aplicando el mapeo de columnas configurado."""
    ruta_movimientos, ruta_catalogo = _rutas_datos(config, argumentos)
    mapeo = config.obtener("datos.mapeo_columnas") or {}

    # El mapeo se aplica antes de parsear fechas: la columna de fecha puede
    # llamarse distinto en el archivo de origen.
    movimientos = esq.aplicar_mapeo(leer_tabla(ruta_movimientos), mapeo)
    if esq.COL_FECHA in movimientos.columns:
        movimientos[esq.COL_FECHA] = pd.to_datetime(
            movimientos[esq.COL_FECHA], errors="coerce", format=config.obtener("datos.formato_fecha")
        )

    catalogo = None
    if ruta_catalogo.exists():
        catalogo = esq.aplicar_mapeo(leer_tabla(ruta_catalogo), mapeo)
    else:
        logger.warning(
            "No se encontro el catalogo en %s: se trabaja sin atributos de SKU", ruta_catalogo
        )
    return movimientos, catalogo


# --------------------------------------------------------------------- comandos
def comando_generar_datos(argumentos: argparse.Namespace, config: Config) -> int:
    """Genera el conjunto de datos sintetico de demostracion."""
    config.asegurar_directorios()
    seccion = config.seccion("simulacion")
    catalogo, movimientos = generar_conjunto_datos(
        n_skus=int(argumentos.n_skus or seccion.get("n_skus", 120)),
        fecha_inicio=str(seccion.get("fecha_inicio", "2019-01-01")),
        fecha_fin=str(seccion.get("fecha_fin", "2024-12-31")),
        semilla=int(config.obtener("proyecto.semilla", 42)),
        n_clientes=int(seccion.get("n_clientes", 60)),
        proporcion_intermitentes=float(seccion.get("proporcion_intermitentes", 0.55)),
        probabilidad_promocion=float(seccion.get("probabilidad_promocion", 0.05)),
        probabilidad_quiebre_stock=float(seccion.get("probabilidad_quiebre_stock", 0.03)),
    )
    destino = config.ruta_de("proyecto.directorio_datos_crudos")
    guardar_tabla(movimientos, destino / ARCHIVO_MOVIMIENTOS)
    guardar_tabla(catalogo, destino / ARCHIVO_CATALOGO)
    print(f"Movimientos: {destino / ARCHIVO_MOVIMIENTOS} ({len(movimientos):,} filas)")
    print(f"Catalogo:    {destino / ARCHIVO_CATALOGO} ({len(catalogo):,} SKU)")
    return 0


def comando_entrenar(argumentos: argparse.Namespace, config: Config) -> int:
    """Entrena, valida y guarda el modelo."""
    movimientos, catalogo = _cargar_entradas(config, argumentos)
    resultado = entrenar_pipeline(
        movimientos,
        catalogo,
        config,
        ejecutar_backtesting=not argumentos.sin_backtesting,
        guardar=True,
    )

    print("\n=== Resumen del entrenamiento ===")
    for clave, valor in resultado.metadatos.items():
        print(f"  {clave:20s}: {valor}")
    if resultado.metricas:
        columnas = [c for c in ("modelo", "wape", "mase", "mae", "rmse", "sesgo_relativo",
                                "tasa_llenado") if c in resultado.metricas["global"].columns]
        print("\n=== Comparacion de modelos (backtesting) ===")
        print(
            resultado.metricas["global"][columnas]
            .sort_values("wape")
            .to_string(index=False, float_format=lambda v: f"{v:,.4f}")
        )
    print(f"\nModelo guardado en: {resultado.ruta_modelo}")
    print(f"Reportes en:        {config.ruta_de('proyecto.directorio_reportes')}")
    return 0


def comando_predecir(argumentos: argparse.Namespace, config: Config) -> int:
    """Genera el pronostico con el modelo entrenado."""
    movimientos, _ = _cargar_entradas(config, argumentos)
    artefacto = cargar_artefacto(config, argumentos.modelo)
    historia = preparar_historia(movimientos, config)
    pronostico = predecir_demanda(artefacto, historia, horizonte=argumentos.horizonte)

    destino = config.ruta_de("proyecto.directorio_reportes") / ARCHIVO_PRONOSTICO
    guardar_tabla(pronostico, destino)
    print(f"\nPronostico guardado en: {destino}")
    print(pronostico.head(argumentos.filas).to_string(index=False))
    return 0


def comando_reponer(argumentos: argparse.Namespace, config: Config) -> int:
    """Calcula el plan de reposicion a partir del pronostico."""
    movimientos, _ = _cargar_entradas(config, argumentos)
    artefacto = cargar_artefacto(config, argumentos.modelo)
    historia = preparar_historia(movimientos, config)
    pronostico = predecir_demanda(artefacto, historia)

    stock_actual = None
    if argumentos.stock:
        tabla_stock = leer_tabla(argumentos.stock)
        columna_stock = [c for c in tabla_stock.columns if c != esq.COL_SKU][0]
        stock_actual = tabla_stock.set_index(esq.COL_SKU)[columna_stock]

    plan = plan_de_reposicion(
        artefacto, pronostico, config, stock_actual=stock_actual, metodo=argumentos.metodo
    )
    destino = config.ruta_de("proyecto.directorio_reportes") / ARCHIVO_PLAN
    guardar_tabla(plan, destino)

    print("\n=== Plan de reposicion ===")
    for clave, valor in resumen_politica(plan).items():
        print(f"  {clave:28s}: {valor:,.2f}")
    print(f"\nPlan guardado en: {destino}")
    print(plan.head(argumentos.filas).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return 0


def comando_diagnostico(argumentos: argparse.Namespace, config: Config) -> int:
    """Evalua si los datos alcanzan para entrenar, sin entrenar nada."""
    movimientos, catalogo = _cargar_entradas(config, argumentos)
    seccion = config.seccion("datos")
    informe = diagnosticar(
        movimientos,
        catalogo,
        frecuencia=str(seccion.get("frecuencia", "S")),
        min_periodos=int(seccion.get("min_periodos_historia", 52)),
        min_demanda_total=float(seccion.get("min_demanda_total", 12)),
        periodo_estacional=int(config.obtener("caracteristicas.periodo_estacional", 52)),
        horizonte=int(config.obtener("modelo.horizonte", 13)),
    )
    imprimir_diagnostico(informe)
    guardar_json(informe, config.ruta_de("proyecto.directorio_reportes") / ARCHIVO_DIAGNOSTICO)
    # Codigo 2 = los datos no alcanzan, distinto de un fallo de ejecucion.
    return 0 if informe.get("apto_para_entrenar") else 2


def comando_importancia(argumentos: argparse.Namespace, config: Config) -> int:
    """Calcula la importancia por permutacion de las variables del modelo."""
    movimientos, _ = _cargar_entradas(config, argumentos)
    artefacto = cargar_artefacto(config, argumentos.modelo)
    if not hasattr(artefacto.modelo, "importancia_permutacion"):
        raise TypeError(
            f"El modelo '{artefacto.nombre_modelo}' no expone importancia de variables"
        )
    historia = preparar_historia(movimientos, config)
    importancia = artefacto.modelo.importancia_permutacion(
        historia,
        horizonte=argumentos.horizonte or 1,
        n_repeticiones=argumentos.repeticiones,
        max_filas=argumentos.max_filas,
    )
    destino = config.ruta_de("proyecto.directorio_reportes") / ARCHIVO_IMPORTANCIA
    guardar_tabla(importancia, destino)
    print(f"\nImportancia guardada en: {destino}")
    print(importancia.head(argumentos.filas).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return 0


def comando_info(argumentos: argparse.Namespace, config: Config) -> int:
    """Muestra la configuracion vigente y los modelos disponibles."""
    print("Modelos disponibles:", ", ".join(modelos_disponibles()))
    print(f"Configuracion: {config.ruta}")
    for seccion in ("datos", "caracteristicas", "modelo", "validacion", "inventario"):
        print(f"\n[{seccion}]")
        for clave, valor in config.seccion(seccion).items():
            print(f"  {clave} = {valor}")
    return 0


# ------------------------------------------------------------------- argumentos
def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos."""
    # Las opciones comunes se declaran en un parser padre para que se acepten
    # tanto antes como despues del subcomando.
    #
    # `argparse.SUPPRESS` es imprescindible aqui: al procesar el subcomando,
    # argparse vuelca sobre el espacio de nombres principal todo lo que parseo
    # el subparser, incluidos sus valores por defecto. Sin SUPPRESS, escribir
    # `pronostico --config X entrenar` terminaria con `config = None`, porque el
    # subparser lo pisaria. Con SUPPRESS la clave simplemente no existe cuando la
    # opcion no se uso, y `_completar_opciones_comunes` pone el valor por defecto
    # despues de parsear.
    #
    # Por el mismo motivo no se usa `parser.set_defaults`: ese metodo reescribe
    # el atributo `default` de las acciones que coinciden, y `parents=` comparte
    # esas acciones con los subparsers, con lo que anularia el SUPPRESS.
    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Ruta del archivo YAML de configuracion",
    )
    comunes.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Muestra el detalle de la ejecucion",
    )
    comunes.add_argument(
        "--movimientos", default=argparse.SUPPRESS, help="CSV de movimientos de venta"
    )
    comunes.add_argument(
        "--catalogo", default=argparse.SUPPRESS, help="CSV del catalogo de SKU"
    )

    parser = argparse.ArgumentParser(
        prog="pronostico",
        parents=[comunes],
        description=(
            "Sistema de pronostico de demanda para repuestos de maquinaria "
            "agroindustrial."
        ),
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    p_datos = subcomandos.add_parser(
        "generar-datos",
        parents=[comunes],
        help="Genera datos sinteticos de demostracion",
    )
    p_datos.add_argument("--n-skus", type=int, help="Cantidad de SKU a simular")
    p_datos.set_defaults(funcion=comando_generar_datos)

    p_entrenar = subcomandos.add_parser(
        "entrenar", parents=[comunes], help="Entrena y valida el modelo"
    )
    p_entrenar.add_argument(
        "--sin-backtesting",
        action="store_true",
        help="Entrena directamente sin comparar contra las referencias",
    )
    p_entrenar.set_defaults(funcion=comando_entrenar)

    p_predecir = subcomandos.add_parser(
        "predecir", parents=[comunes], help="Genera el pronostico de demanda"
    )
    p_predecir.add_argument("--modelo", help="Ruta del artefacto entrenado")
    p_predecir.add_argument("--horizonte", type=int, help="Periodos a pronosticar")
    p_predecir.add_argument("--filas", type=int, default=15, help="Filas a mostrar")
    p_predecir.set_defaults(funcion=comando_predecir)

    p_reponer = subcomandos.add_parser(
        "reponer", parents=[comunes], help="Calcula el plan de reposicion"
    )
    p_reponer.add_argument("--modelo", help="Ruta del artefacto entrenado")
    p_reponer.add_argument("--stock", help="CSV con las existencias actuales por SKU")
    p_reponer.add_argument(
        "--metodo",
        choices=["parametrico", "cuantil"],
        help="Metodo de calculo del stock de seguridad",
    )
    p_reponer.add_argument("--filas", type=int, default=15, help="Filas a mostrar")
    p_reponer.set_defaults(funcion=comando_reponer)

    p_diagnostico = subcomandos.add_parser(
        "diagnostico",
        parents=[comunes],
        help="Evalua si los datos alcanzan para entrenar, sin entrenar",
    )
    p_diagnostico.set_defaults(funcion=comando_diagnostico)

    p_importancia = subcomandos.add_parser(
        "importancia",
        parents=[comunes],
        help="Calcula la importancia por permutacion de las variables",
    )
    p_importancia.add_argument("--modelo", help="Ruta del artefacto entrenado")
    p_importancia.add_argument(
        "--horizonte", type=int, help="Horizonte a analizar (por defecto 1)"
    )
    p_importancia.add_argument(
        "--repeticiones", type=int, default=3, help="Permutaciones por variable"
    )
    p_importancia.add_argument(
        "--max-filas", type=int, default=5000, help="Muestra maxima de filas"
    )
    p_importancia.add_argument("--filas", type=int, default=20, help="Filas a mostrar")
    p_importancia.set_defaults(funcion=comando_importancia)

    p_info = subcomandos.add_parser(
        "info", parents=[comunes], help="Muestra la configuracion vigente"
    )
    p_info.set_defaults(funcion=comando_info)
    return parser


# Opciones comunes y su valor cuando no se usan (ver `construir_parser`).
OPCIONES_COMUNES: dict[str, object] = {
    "config": None,
    "verbose": False,
    "movimientos": None,
    "catalogo": None,
}


def _completar_opciones_comunes(argumentos: argparse.Namespace) -> argparse.Namespace:
    """Rellena las opciones comunes que argparse omitio por `SUPPRESS`."""
    for destino, por_defecto in OPCIONES_COMUNES.items():
        if not hasattr(argumentos, destino):
            setattr(argumentos, destino, por_defecto)
    return argumentos


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la linea de comandos."""
    parser = construir_parser()
    argumentos = _completar_opciones_comunes(parser.parse_args(argv))
    configurar_logging(logging.DEBUG if argumentos.verbose else logging.INFO)

    try:
        config = cargar_config(argumentos.config)
        pd.set_option("display.width", 160)
        return int(argumentos.funcion(argumentos, config))
    except Exception as error:  # noqa: BLE001 - la CLI reporta el fallo al usuario
        logger.error("%s: %s", type(error).__name__, error)
        if argumentos.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
