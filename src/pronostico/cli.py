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
from .utilidades.persistencia import guardar_tabla, leer_tabla
from .utilidades.registro_log import configurar_logging, obtener_logger

logger = obtener_logger(__name__)

ARCHIVO_MOVIMIENTOS = "movimientos.csv"
ARCHIVO_CATALOGO = "catalogo_skus.csv"
ARCHIVO_PRONOSTICO = "pronostico.csv"
ARCHIVO_PLAN = "plan_reposicion.csv"
ARCHIVO_IMPORTANCIA = "importancia_variables.csv"


def _rutas_datos(config: Config, argumentos: argparse.Namespace) -> tuple[Path, Path]:
    """Resuelve las rutas de los archivos de entrada."""
    base = config.ruta_de("proyecto.directorio_datos_crudos")
    movimientos = Path(argumentos.movimientos) if argumentos.movimientos else base / ARCHIVO_MOVIMIENTOS
    catalogo = Path(argumentos.catalogo) if argumentos.catalogo else base / ARCHIVO_CATALOGO
    return movimientos, catalogo


def _cargar_entradas(config: Config, argumentos: argparse.Namespace):
    """Lee movimientos y catalogo desde disco."""
    ruta_movimientos, ruta_catalogo = _rutas_datos(config, argumentos)
    movimientos = leer_tabla(ruta_movimientos, columnas_fecha=[esq.COL_FECHA])
    catalogo = leer_tabla(ruta_catalogo) if ruta_catalogo.exists() else None
    if catalogo is None:
        logger.warning("No se encontro el catalogo en %s: se entrena sin atributos de SKU", ruta_catalogo)
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
    parser = argparse.ArgumentParser(
        prog="pronostico",
        description=(
            "Sistema de pronostico de demanda para repuestos de maquinaria "
            "agroindustrial."
        ),
    )
    parser.add_argument("--config", help="Ruta del archivo YAML de configuracion")
    parser.add_argument(
        "--verbose", action="store_true", help="Muestra el detalle de la ejecucion"
    )
    parser.add_argument("--movimientos", help="CSV de movimientos de venta")
    parser.add_argument("--catalogo", help="CSV del catalogo de SKU")

    subcomandos = parser.add_subparsers(dest="comando", required=True)

    p_datos = subcomandos.add_parser(
        "generar-datos", help="Genera datos sinteticos de demostracion"
    )
    p_datos.add_argument("--n-skus", type=int, help="Cantidad de SKU a simular")
    p_datos.set_defaults(funcion=comando_generar_datos)

    p_entrenar = subcomandos.add_parser("entrenar", help="Entrena y valida el modelo")
    p_entrenar.add_argument(
        "--sin-backtesting",
        action="store_true",
        help="Entrena directamente sin comparar contra las referencias",
    )
    p_entrenar.set_defaults(funcion=comando_entrenar)

    p_predecir = subcomandos.add_parser("predecir", help="Genera el pronostico de demanda")
    p_predecir.add_argument("--modelo", help="Ruta del artefacto entrenado")
    p_predecir.add_argument("--horizonte", type=int, help="Periodos a pronosticar")
    p_predecir.add_argument("--filas", type=int, default=15, help="Filas a mostrar")
    p_predecir.set_defaults(funcion=comando_predecir)

    p_reponer = subcomandos.add_parser("reponer", help="Calcula el plan de reposicion")
    p_reponer.add_argument("--modelo", help="Ruta del artefacto entrenado")
    p_reponer.add_argument("--stock", help="CSV con las existencias actuales por SKU")
    p_reponer.add_argument(
        "--metodo",
        choices=["parametrico", "cuantil"],
        help="Metodo de calculo del stock de seguridad",
    )
    p_reponer.add_argument("--filas", type=int, default=15, help="Filas a mostrar")
    p_reponer.set_defaults(funcion=comando_reponer)

    p_importancia = subcomandos.add_parser(
        "importancia", help="Calcula la importancia por permutacion de las variables"
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

    p_info = subcomandos.add_parser("info", help="Muestra la configuracion vigente")
    p_info.set_defaults(funcion=comando_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la linea de comandos."""
    parser = construir_parser()
    argumentos = parser.parse_args(argv)
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
