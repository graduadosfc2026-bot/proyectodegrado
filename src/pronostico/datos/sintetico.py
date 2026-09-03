"""Generador de datos sinteticos de demanda de repuestos agroindustriales.

Sirve para dos cosas:

1. Permitir que el proyecto sea ejecutable de punta a punta sin depender de
   datos confidenciales de la empresa.
2. Actuar como banco de pruebas con "verdad conocida": la demanda se construye
   a partir de componentes explicitos (estacionalidad de campana, tendencia,
   precio, promociones, intermitencia) que luego los modelos deben recuperar.

El calendario agricola modelado corresponde al Cono Sur (hemisferio sur):

* Siembra gruesa (soja / maiz): septiembre - diciembre.
* Cosecha gruesa:               marzo - julio.
* Siembra fina (trigo / cebada): mayo - julio.
* Cosecha fina:                 noviembre - diciembre.

La demanda de repuestos se adelanta a la ventana de uso de cada maquina porque
el productor repara antes de entrar a la campana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utilidades.registro_log import obtener_logger
from . import esquema as esq

logger = obtener_logger(__name__)

# Perfil estacional mensual (indice 0 = enero) por tipo de maquina.
# Valores multiplicativos alrededor de 1.0.
PERFILES_ESTACIONALES: dict[str, list[float]] = {
    # Preparacion en enero-febrero y uso intenso en la cosecha gruesa.
    "cosechadora": [1.35, 1.70, 1.75, 1.50, 1.15, 0.95, 0.80, 0.65, 0.60, 0.70, 1.05, 1.25],
    # Se acondiciona antes de la siembra gruesa (sep-dic) y de la fina (may-jul).
    "sembradora": [0.65, 0.70, 0.80, 1.05, 1.30, 1.20, 0.90, 1.25, 1.70, 1.65, 1.10, 0.80],
    # Aplicaciones concentradas en el verano del cultivo de gruesa.
    "pulverizadora": [1.55, 1.45, 1.20, 0.85, 0.70, 0.60, 0.65, 0.85, 1.10, 1.35, 1.50, 1.50],
    # Uso durante todo el ano, con repunte en las ventanas de campana.
    "tractor": [1.15, 1.20, 1.15, 1.05, 1.00, 0.90, 0.85, 0.90, 1.05, 1.10, 1.10, 1.10],
    # Logistica de grano: acompana la cosecha.
    "tolva_acoplado": [1.20, 1.35, 1.60, 1.45, 1.10, 0.85, 0.70, 0.65, 0.70, 0.85, 1.20, 1.35],
}

# Familias de repuestos con su rango de costo unitario y su rotacion tipica.
FAMILIAS: dict[str, dict[str, float | str]] = {
    "filtros":               {"costo_medio": 18_000,  "rotacion": "alta"},
    "correas_y_mangueras":   {"costo_medio": 42_000,  "rotacion": "alta"},
    "rodamientos":           {"costo_medio": 65_000,  "rotacion": "media"},
    "cuchillas_y_puas":      {"costo_medio": 31_000,  "rotacion": "alta"},
    "hidraulica":            {"costo_medio": 210_000, "rotacion": "baja"},
    "embrague_y_transmision":{"costo_medio": 380_000, "rotacion": "baja"},
    "neumaticos":            {"costo_medio": 520_000, "rotacion": "baja"},
    "electrico_y_sensores":  {"costo_medio": 145_000, "rotacion": "media"},
    "lubricantes":           {"costo_medio": 26_000,  "rotacion": "alta"},
    "chapa_y_carroceria":    {"costo_medio": 95_000,  "rotacion": "baja"},
}

MAQUINAS = list(PERFILES_ESTACIONALES.keys())
CANALES = ["mostrador", "taller_propio", "mayorista", "ecommerce"]
CRITICIDADES = ["alta", "media", "baja"]
ORIGENES = ["nacional", "importado"]

# Feriados nacionales aproximados (mes, dia) con actividad comercial nula.
FERIADOS_FIJOS = [(1, 1), (5, 1), (5, 25), (6, 20), (7, 9), (12, 8), (12, 25)]


def _factor_estacional_diario(fechas: pd.DatetimeIndex, maquina: str) -> np.ndarray:
    """Interpola el perfil mensual de una maquina a una curva diaria suave.

    Se usa una serie de Fourier ajustada al perfil mensual para evitar los
    saltos artificiales del primer dia de cada mes.
    """
    perfil = np.asarray(PERFILES_ESTACIONALES[maquina], dtype=float)
    # Posicion angular del centro de cada mes dentro del ano.
    angulos_mes = 2.0 * np.pi * (np.arange(12) + 0.5) / 12.0
    diseno_mes = np.column_stack(
        [np.ones(12)]
        + [f(k * angulos_mes) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    )
    coeficientes, *_ = np.linalg.lstsq(diseno_mes, perfil, rcond=None)

    dia_del_ano = np.asarray(fechas.dayofyear, dtype=float)
    longitud = np.where(np.asarray(fechas.is_leap_year, dtype=bool), 366.0, 365.0)
    angulos_dia = 2.0 * np.pi * (dia_del_ano - 0.5) / longitud
    diseno_dia = np.column_stack(
        [np.ones(len(fechas))]
        + [f(k * angulos_dia) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    )
    return np.clip(diseno_dia @ coeficientes, 0.15, None)


def _factor_dia_semana(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Actividad comercial por dia de la semana (domingo cerrado)."""
    pesos = np.array([1.05, 1.00, 1.00, 1.00, 1.10, 0.55, 0.05])  # lunes..domingo
    factores = pesos[np.asarray(fechas.dayofweek, dtype=int)]
    es_feriado = np.array(
        [(f.month, f.day) in FERIADOS_FIJOS for f in fechas], dtype=bool
    )
    factores = np.where(es_feriado, 0.05, factores)
    return factores


def _muestrear_binomial_negativa(
    rng: np.random.Generator, medias: np.ndarray, dispersion: float
) -> np.ndarray:
    """Muestra una binomial negativa con media `medias` y sobredispersion dada.

    Parametrizacion media-varianza: var = mu + mu^2 / r, con r = `dispersion`.
    Valores altos de `dispersion` se acercan a una Poisson.
    """
    medias = np.clip(medias, 1e-9, None)
    probabilidades = dispersion / (dispersion + medias)
    return rng.negative_binomial(dispersion, probabilidades).astype(float)


def generar_catalogo(n_skus: int, rng: np.random.Generator) -> pd.DataFrame:
    """Construye el catalogo maestro de repuestos."""
    familias = list(FAMILIAS.keys())
    filas = []
    for indice in range(n_skus):
        familia = familias[indice % len(familias)]
        maquina = MAQUINAS[int(rng.integers(len(MAQUINAS)))]
        origen = "importado" if rng.random() < 0.4 else "nacional"
        costo_base = float(FAMILIAS[familia]["costo_medio"])
        costo = float(np.round(costo_base * np.exp(rng.normal(0.0, 0.45)), 2))
        margen = float(rng.uniform(1.35, 2.10))
        lead_time = int(rng.integers(45, 121) if origen == "importado" else rng.integers(5, 26))
        filas.append(
            {
                esq.COL_SKU: f"RP-{indice + 1:04d}",
                esq.COL_DESCRIPCION: f"{familia.replace('_', ' ')} para {maquina}",
                esq.COL_FAMILIA: familia,
                esq.COL_MAQUINA: maquina,
                esq.COL_CRITICIDAD: str(
                    rng.choice(CRITICIDADES, p=[0.35, 0.45, 0.20])
                ),
                esq.COL_COSTO: costo,
                esq.COL_PRECIO_LISTA: float(np.round(costo * margen, 2)),
                esq.COL_LEAD_TIME: lead_time,
                esq.COL_ORIGEN: origen,
                esq.COL_MOQ: int(rng.choice([1, 1, 2, 5, 10])),
            }
        )
    return pd.DataFrame(filas)


def generar_movimientos(
    catalogo: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
    rng: np.random.Generator,
    n_clientes: int = 60,
    proporcion_intermitentes: float = 0.55,
    probabilidad_promocion: float = 0.05,
    probabilidad_quiebre_stock: float = 0.03,
) -> pd.DataFrame:
    """Genera el historico transaccional de demanda para todo el catalogo.

    Cada SKU combina:

    * un nivel base propio (lognormal),
    * la estacionalidad de campana de su maquina,
    * una tendencia suave de largo plazo,
    * elasticidad al precio con promociones ocasionales,
    * intermitencia (una fraccion de los SKU son de baja rotacion),
    * quiebres de stock que censuran la demanda observada.

    La columna `quiebre_stock` marca los dias censurados: es informacion que en
    la practica permite corregir la demanda observada antes de entrenar.
    """
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    n_dias = len(fechas)
    tiempo_normalizado = np.linspace(0.0, 1.0, n_dias)
    factor_semana = _factor_dia_semana(fechas)
    clientes = [f"CLI-{i + 1:04d}" for i in range(n_clientes)]
    # Cache de la curva estacional: solo hay una por tipo de maquina.
    estacionalidad = {m: _factor_estacional_diario(fechas, m) for m in MAQUINAS}

    registros: list[pd.DataFrame] = []
    for fila in catalogo.itertuples(index=False):
        rotacion = str(FAMILIAS[getattr(fila, esq.COL_FAMILIA)]["rotacion"])
        es_intermitente = rng.random() < proporcion_intermitentes

        if es_intermitente:
            nivel_base = float(np.exp(rng.normal(-2.4, 0.7)))   # ~0.02 - 0.3 u/dia
            dispersion = float(rng.uniform(0.25, 0.9))          # muy sobredisperso
        else:
            escala = {"alta": 0.9, "media": 0.2, "baja": -0.5}[rotacion]
            nivel_base = float(np.exp(rng.normal(escala, 0.6)))  # ~0.5 - 5 u/dia
            dispersion = float(rng.uniform(1.5, 6.0))

        # Tendencia de largo plazo: parque de maquinas creciendo o SKU en salida.
        tendencia = np.exp(rng.normal(0.0, 0.35) * tiempo_normalizado)
        # Ciclo lento adicional (clima, precio de los granos, ciclo economico).
        fase = rng.uniform(0, 2 * np.pi)
        ciclo = 1.0 + 0.18 * np.sin(2 * np.pi * tiempo_normalizado * rng.uniform(1.0, 2.5) + fase)

        # Promociones: bloques cortos de descuento sobre el precio de lista.
        en_promocion = np.zeros(n_dias, dtype=bool)
        dia = 0
        while dia < n_dias:
            if rng.random() < probabilidad_promocion:
                largo = int(rng.integers(7, 22))
                en_promocion[dia : dia + largo] = True
                dia += largo
            dia += 7
        descuento = np.where(en_promocion, rng.uniform(0.10, 0.28), 0.0)
        elasticidad = float(rng.uniform(1.0, 2.2))
        factor_precio = (1.0 - descuento) ** (-elasticidad)

        intensidad = (
            nivel_base
            * estacionalidad[getattr(fila, esq.COL_MAQUINA)]
            * tendencia
            * ciclo
            * factor_semana
            * factor_precio
        )
        demanda_real = _muestrear_binomial_negativa(rng, intensidad, dispersion)

        # Quiebres de stock: rachas donde no se puede vender (demanda censurada).
        quiebre = np.zeros(n_dias, dtype=bool)
        dia = 0
        while dia < n_dias:
            if rng.random() < probabilidad_quiebre_stock:
                largo = int(rng.integers(3, 18))
                quiebre[dia : dia + largo] = True
                dia += largo
            dia += 14
        demanda_observada = np.where(quiebre, 0.0, demanda_real)

        activos = np.flatnonzero(demanda_observada > 0)
        if activos.size == 0:
            continue

        precio_lista = float(getattr(fila, esq.COL_PRECIO_LISTA))
        registros.append(
            pd.DataFrame(
                {
                    esq.COL_FECHA: fechas[activos],
                    esq.COL_SKU: getattr(fila, esq.COL_SKU),
                    esq.COL_CANTIDAD: demanda_observada[activos],
                    esq.COL_PRECIO: np.round(
                        precio_lista
                        * (1.0 - descuento[activos])
                        * rng.normal(1.0, 0.02, activos.size),
                        2,
                    ),
                    esq.COL_CLIENTE: rng.choice(clientes, activos.size),
                    esq.COL_CANAL: rng.choice(CANALES, activos.size, p=[0.45, 0.2, 0.25, 0.1]),
                    esq.COL_PROMOCION: en_promocion[activos].astype(int),
                    esq.COL_QUIEBRE: 0,
                }
            )
        )

        # Se registran tambien los dias de quiebre para poder corregir la serie.
        dias_quiebre = np.flatnonzero(quiebre)
        if dias_quiebre.size:
            registros.append(
                pd.DataFrame(
                    {
                        esq.COL_FECHA: fechas[dias_quiebre],
                        esq.COL_SKU: getattr(fila, esq.COL_SKU),
                        esq.COL_CANTIDAD: 0.0,
                        esq.COL_PRECIO: np.nan,
                        esq.COL_CLIENTE: None,
                        esq.COL_CANAL: None,
                        esq.COL_PROMOCION: en_promocion[dias_quiebre].astype(int),
                        esq.COL_QUIEBRE: 1,
                    }
                )
            )

    if not registros:
        raise RuntimeError("La simulacion no genero ningun movimiento de demanda")

    movimientos = pd.concat(registros, ignore_index=True)
    return movimientos.sort_values([esq.COL_FECHA, esq.COL_SKU]).reset_index(drop=True)


def generar_conjunto_datos(
    n_skus: int = 120,
    fecha_inicio: str = "2019-01-01",
    fecha_fin: str = "2024-12-31",
    semilla: int = 42,
    n_clientes: int = 60,
    proporcion_intermitentes: float = 0.55,
    probabilidad_promocion: float = 0.05,
    probabilidad_quiebre_stock: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera catalogo y movimientos sinteticos listos para el pipeline."""
    rng = np.random.default_rng(semilla)
    catalogo = generar_catalogo(n_skus, rng)
    movimientos = generar_movimientos(
        catalogo,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        rng=rng,
        n_clientes=n_clientes,
        proporcion_intermitentes=proporcion_intermitentes,
        probabilidad_promocion=probabilidad_promocion,
        probabilidad_quiebre_stock=probabilidad_quiebre_stock,
    )
    logger.info(
        "Datos sinteticos generados: %d SKU, %d movimientos, %s a %s",
        catalogo.shape[0],
        movimientos.shape[0],
        movimientos[esq.COL_FECHA].min().date(),
        movimientos[esq.COL_FECHA].max().date(),
    )
    return catalogo, movimientos
