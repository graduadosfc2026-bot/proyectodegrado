"""Pipeline de prediccion: del modelo guardado al plan de compras."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..datos import esquema as esq
from ..datos.preparacion import preparar_panel
from ..inventario.politica import calcular_politica
from ..modelos.arbol import etiqueta_cuantil
from ..utilidades.persistencia import cargar_modelo
from ..utilidades.registro_log import obtener_logger
from .entrenamiento import NOMBRE_ARTEFACTO, ArtefactoModelo

logger = obtener_logger(__name__)


def cargar_artefacto(config: Config, ruta: str | Path | None = None) -> ArtefactoModelo:
    """Carga el artefacto de modelo entrenado desde disco."""
    destino = (
        Path(ruta)
        if ruta is not None
        else config.ruta_de("proyecto.directorio_modelos") / NOMBRE_ARTEFACTO
    )
    artefacto = cargar_modelo(destino)
    if not isinstance(artefacto, ArtefactoModelo):
        raise TypeError(f"El archivo {destino} no contiene un artefacto de modelo valido")
    logger.info("Artefacto cargado: modelo '%s' desde %s", artefacto.nombre_modelo, destino)
    return artefacto


def preparar_historia(movimientos: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Prepara el panel de historia con los mismos criterios del entrenamiento."""
    seccion = config.seccion("datos")
    return preparar_panel(
        movimientos,
        frecuencia=str(seccion.get("frecuencia", "S")),
        columna_objetivo=str(seccion.get("columna_objetivo", esq.COL_CANTIDAD)),
        min_periodos=int(seccion.get("min_periodos_historia", 52)),
        min_demanda_total=float(seccion.get("min_demanda_total", 12)),
        winsorizar_cuantil=seccion.get("winsorizar_cuantil", 0.995),
    )


def predecir_demanda(
    artefacto: ArtefactoModelo,
    historia: pd.DataFrame,
    horizonte: int | None = None,
    cuantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Genera el pronostico para todos los SKU desde el fin de la historia."""
    config = Config(datos=artefacto.configuracion)
    horizonte_final = int(horizonte or config.obtener("modelo.horizonte", 13))
    niveles = cuantiles if cuantiles is not None else list(
        config.obtener("modelo.cuantiles", []) or []
    )

    if niveles:
        pronostico = artefacto.modelo.predecir_cuantiles(historia, horizonte_final, niveles)
    else:
        pronostico = artefacto.modelo.predecir(historia, horizonte_final)

    pronostico[esq.COL_MODELO] = artefacto.nombre_modelo
    logger.info(
        "Pronostico generado: %d SKU x %d periodos",
        pronostico[esq.COL_SKU].nunique(),
        horizonte_final,
    )
    return pronostico


def plan_de_reposicion(
    artefacto: ArtefactoModelo,
    pronostico: pd.DataFrame,
    config: Config | None = None,
    stock_actual: pd.Series | None = None,
    metodo: str | None = None,
) -> pd.DataFrame:
    """Convierte el pronostico en la politica de reposicion por SKU."""
    configuracion = config or Config(datos=artefacto.configuracion)
    seccion = configuracion.seccion("inventario")
    nivel_servicio = float(seccion.get("nivel_servicio", 0.95))

    # Si el modelo estima el cuantil del nivel de servicio se usa ese metodo,
    # que no supone normalidad del error; si no, se cae al parametrico.
    metodo_final = metodo
    if metodo_final is None:
        tiene_cuantil = etiqueta_cuantil(nivel_servicio) in pronostico.columns
        metodo_final = "cuantil" if tiene_cuantil else "parametrico"

    return calcular_politica(
        pronostico,
        catalogo=artefacto.catalogo,
        error_por_sku=artefacto.error_por_sku,
        nivel_servicio=nivel_servicio,
        lead_time_dias_por_defecto=float(seccion.get("lead_time_dias_por_defecto", 21)),
        periodo_revision_dias=float(seccion.get("periodo_revision_dias", 7)),
        frecuencia=str(configuracion.obtener("datos.frecuencia", "S")),
        stock_actual=stock_actual,
        metodo=metodo_final,
    )
