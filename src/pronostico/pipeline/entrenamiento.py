"""Pipeline de entrenamiento: de los datos crudos al modelo productivo.

Pasos:

1. Preparar el panel de demanda y clasificar los SKU (ABC-XYZ y regimen).
2. Comparar el modelo principal contra las referencias por origen movil.
3. Reentrenar el modelo ganador con toda la historia disponible.
4. Guardar el artefacto entrenado y los reportes de validacion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..caracteristicas.constructor import ConstructorCaracteristicas
from ..config import Config
from ..datos import esquema as esq
from ..datos.preparacion import clasificar_abc_xyz, preparar_panel
from ..evaluacion.backtesting import ejecutar_backtest, elegir_mejor_modelo, resumir_backtest
from ..inventario.politica import error_por_sku_desde_backtest
from ..modelos.base import ModeloPronostico
from ..modelos.registro import crear_modelo
from ..utilidades.persistencia import guardar_json, guardar_modelo, guardar_tabla
from ..utilidades.registro_log import obtener_logger

logger = obtener_logger(__name__)

NOMBRE_ARTEFACTO = "modelo_pronostico.joblib"


@dataclass
class ResultadoEntrenamiento:
    """Todo lo que produce un entrenamiento completo."""

    modelo: ModeloPronostico
    nombre_modelo: str
    panel: pd.DataFrame
    clasificacion: pd.DataFrame
    resultados_backtest: pd.DataFrame
    metricas: dict[str, pd.DataFrame]
    ruta_modelo: Path | None = None
    metadatos: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtefactoModelo:
    """Objeto serializado que se lleva a produccion."""

    modelo: ModeloPronostico
    nombre_modelo: str
    catalogo: pd.DataFrame | None
    configuracion: dict[str, Any]
    metricas_validacion: dict[str, Any]
    error_por_sku: pd.Series | None = None


def _constructor_desde_config(config: Config) -> ConstructorCaracteristicas:
    """Arma el constructor de caracteristicas segun la configuracion."""
    seccion = config.seccion("caracteristicas")
    return ConstructorCaracteristicas(
        rezagos=list(seccion.get("rezagos", [1, 2, 3, 4, 8, 13, 26, 52])),
        ventanas_moviles=list(seccion.get("ventanas_moviles", [4, 8, 13, 26, 52])),
        periodo_estacional=int(seccion.get("periodo_estacional", 52)),
        ordenes_fourier=int(seccion.get("ordenes_fourier", 3)),
        usar_calendario_agricola=bool(seccion.get("usar_calendario_agricola", True)),
        usar_atributos_sku=bool(seccion.get("usar_atributos_sku", True)),
    )


def construir_fabricas(
    config: Config,
    incluir_referencias: bool = True,
    incluir_cuantiles: bool = True,
) -> dict[str, Any]:
    """Devuelve las fabricas de modelos a comparar en el backtesting.

    Se usan fabricas (funciones sin argumentos) y no instancias porque cada
    origen del backtesting necesita un modelo recien inicializado.

    Args:
        incluir_cuantiles: si es False el modelo principal solo estima la media
            condicional. Entrenar los cuantiles multiplica el costo por la
            cantidad de niveles, y la comparacion entre modelos del backtesting
            se hace sobre el pronostico puntual.
    """
    frecuencia = str(config.obtener("datos.frecuencia", "S"))
    periodo = int(config.obtener("caracteristicas.periodo_estacional", 52))
    horizonte = int(config.obtener("modelo.horizonte", 13))
    nombre_principal = str(config.obtener("modelo.nombre", "gbrt_global"))
    hiperparametros = dict(config.seccion("modelo").get("hiperparametros", {}))
    cuantiles = list(config.obtener("modelo.cuantiles", []) or []) if incluir_cuantiles else []
    semilla = int(config.obtener("proyecto.semilla", 42))
    decaimiento = float(config.obtener("modelo.decaimiento_temporal", 1.0))

    def fabrica_principal() -> ModeloPronostico:
        return crear_modelo(
            nombre_principal,
            horizonte=horizonte,
            cuantiles=cuantiles,
            hiperparametros=hiperparametros,
            constructor=_constructor_desde_config(config),
            decaimiento_temporal=decaimiento,
            semilla=semilla,
            frecuencia=frecuencia,
            periodo_estacional=periodo,
        )

    fabricas: dict[str, Any] = {nombre_principal: fabrica_principal}
    if incluir_referencias:
        for nombre in config.obtener("modelo.referencias", []) or []:
            if nombre == nombre_principal:
                continue
            fabricas[nombre] = (
                lambda n=nombre: crear_modelo(
                    n, frecuencia=frecuencia, periodo_estacional=periodo
                )
            )
    return fabricas


def entrenar_pipeline(
    movimientos: pd.DataFrame,
    catalogo: pd.DataFrame | None,
    config: Config,
    ejecutar_backtesting: bool = True,
    guardar: bool = True,
) -> ResultadoEntrenamiento:
    """Ejecuta el pipeline completo de entrenamiento y devuelve sus resultados."""
    seccion_datos = config.seccion("datos")
    panel = preparar_panel(
        movimientos,
        frecuencia=str(seccion_datos.get("frecuencia", "S")),
        columna_objetivo=str(seccion_datos.get("columna_objetivo", esq.COL_CANTIDAD)),
        min_periodos=int(seccion_datos.get("min_periodos_historia", 52)),
        min_demanda_total=float(seccion_datos.get("min_demanda_total", 12)),
        winsorizar_cuantil=seccion_datos.get("winsorizar_cuantil", 0.995),
    )
    if panel.empty:
        raise RuntimeError("No quedaron SKU con historia suficiente para entrenar")

    seccion_inv = config.seccion("inventario")
    clasificacion = clasificar_abc_xyz(
        panel,
        catalogo,
        cortes_abc=tuple(seccion_inv.get("cortes_abc", (0.8, 0.95))),
        cortes_xyz=tuple(seccion_inv.get("cortes_xyz", (0.5, 1.0))),
    )

    horizonte = int(config.obtener("modelo.horizonte", 13))
    periodo = int(config.obtener("caracteristicas.periodo_estacional", 52))
    nombre_principal = str(config.obtener("modelo.nombre", "gbrt_global"))

    resultados_backtest = pd.DataFrame()
    metricas: dict[str, pd.DataFrame] = {}
    nombre_ganador = nombre_principal

    if ejecutar_backtesting:
        seccion_val = config.seccion("validacion")
        evaluar_cuantiles = bool(seccion_val.get("evaluar_cuantiles", False))
        cuantiles_backtest = (
            list(config.obtener("modelo.cuantiles", []) or []) if evaluar_cuantiles else None
        )
        resultados_backtest = ejecutar_backtest(
            panel,
            fabricas=construir_fabricas(config, incluir_cuantiles=evaluar_cuantiles),
            horizonte=horizonte,
            catalogo=catalogo,
            n_origenes=int(seccion_val.get("n_origenes", 6)),
            paso=int(seccion_val.get("paso_origenes", 4)),
            periodos_reservados=int(seccion_val.get("periodos_prueba", 0)),
            cuantiles=cuantiles_backtest,
        )
        metricas = resumir_backtest(
            resultados_backtest, panel, periodo_estacional=periodo, clases=clasificacion
        )
        metrica_seleccion = str(config.obtener("validacion.metrica_seleccion", "wape"))
        nombre_ganador, valor = elegir_mejor_modelo(metricas["global"], metrica_seleccion)
        logger.info(
            "Modelo ganador del backtesting: %s (%s = %.4f)",
            nombre_ganador,
            metrica_seleccion,
            valor,
        )
        if nombre_ganador != nombre_principal:
            logger.warning(
                "La referencia '%s' supera al modelo principal '%s': se despliega la "
                "referencia y se recomienda revisar las variables del modelo",
                nombre_ganador,
                nombre_principal,
            )

    # Reentrenamiento final con toda la historia disponible.
    fabricas = construir_fabricas(config)
    modelo = fabricas[nombre_ganador]()
    modelo.entrenar(panel, catalogo)

    resultado = ResultadoEntrenamiento(
        modelo=modelo,
        nombre_modelo=nombre_ganador,
        panel=panel,
        clasificacion=clasificacion,
        resultados_backtest=resultados_backtest,
        metricas=metricas,
        metadatos={
            "skus": int(panel[esq.COL_SKU].nunique()),
            "periodos": int(panel[esq.COL_FECHA].nunique()),
            "fecha_inicio": str(panel[esq.COL_FECHA].min().date()),
            "fecha_fin": str(panel[esq.COL_FECHA].max().date()),
            "horizonte": horizonte,
            "modelo_principal": nombre_principal,
            "modelo_desplegado": nombre_ganador,
        },
    )

    if guardar:
        resultado.ruta_modelo = _guardar_resultados(resultado, catalogo, config)
    return resultado


def _guardar_resultados(
    resultado: ResultadoEntrenamiento, catalogo: pd.DataFrame | None, config: Config
) -> Path:
    """Persiste el artefacto del modelo y las tablas del reporte de validacion."""
    config.asegurar_directorios()
    directorio_modelos = config.ruta_de("proyecto.directorio_modelos")
    directorio_reportes = config.ruta_de("proyecto.directorio_reportes")

    error_por_sku = None
    if not resultado.resultados_backtest.empty:
        error_por_sku = error_por_sku_desde_backtest(
            resultado.resultados_backtest, resultado.nombre_modelo
        )

    artefacto = ArtefactoModelo(
        modelo=resultado.modelo,
        nombre_modelo=resultado.nombre_modelo,
        catalogo=catalogo,
        configuracion=config.datos,
        metricas_validacion=(
            resultado.metricas["global"].to_dict("records") if resultado.metricas else {}
        ),
        error_por_sku=error_por_sku,
    )
    ruta_modelo = guardar_modelo(artefacto, directorio_modelos / NOMBRE_ARTEFACTO)

    guardar_tabla(resultado.clasificacion, directorio_reportes / "clasificacion_skus.csv")
    for nombre, tabla in resultado.metricas.items():
        guardar_tabla(tabla, directorio_reportes / f"metricas_{nombre}.csv")
    if not resultado.resultados_backtest.empty:
        guardar_tabla(
            resultado.resultados_backtest, directorio_reportes / "backtest_detalle.csv"
        )
    guardar_json(resultado.metadatos, directorio_reportes / "metadatos_entrenamiento.json")

    logger.info("Artefacto guardado en %s", ruta_modelo)
    return ruta_modelo
