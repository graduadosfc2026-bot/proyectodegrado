"""Variables de calendario y de campana agricola.

La demanda de repuestos agroindustriales no depende del calendario comercial
sino del calendario productivo: el productor repara antes de entrar a la
ventana de siembra o de cosecha. Estas variables le dan al modelo la posicion
del periodo dentro de ese ciclo, de forma continua y sin saltos artificiales.

Las ventanas corresponden al Cono Sur (hemisferio sur). Para adaptar el modelo
a otra region basta con redefinir `CAMPANAS`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nombre de campana -> (dia del ano de inicio, dia del ano de fin).
# Las ventanas que cruzan el fin de ano se expresan con inicio > fin.
CAMPANAS: dict[str, tuple[int, int]] = {
    "siembra_gruesa": (274, 355),   # 1 de octubre  - 21 de diciembre
    "cosecha_gruesa": (60, 196),    # 1 de marzo    - 15 de julio
    "siembra_fina": (135, 212),     # 15 de mayo    - 31 de julio
    "cosecha_fina": (319, 10),      # 15 de noviembre - 10 de enero
}

DIAS_ANO = 365.25


def _dia_del_ano(fechas: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Dia del ano como arreglo de punto flotante."""
    indice = pd.DatetimeIndex(pd.Series(fechas).to_numpy())
    return np.asarray(indice.dayofyear, dtype=float)


def _distancia_circular(dias: np.ndarray, referencia: float) -> np.ndarray:
    """Distancia en dias entre `dias` y `referencia` sobre un ciclo anual."""
    bruta = np.abs(dias - referencia)
    return np.minimum(bruta, DIAS_ANO - bruta)


def _centro_y_ancho(inicio: int, fin: int) -> tuple[float, float]:
    """Centro y semiancho (en dias) de una ventana anual, incluso si cruza enero."""
    largo = (fin - inicio) % DIAS_ANO
    centro = (inicio + largo / 2.0) % DIAS_ANO
    return centro, max(largo / 2.0, 1.0)


def intensidad_campanas(fechas: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Intensidad continua (0 a 1) de cada campana agricola para cada fecha.

    En lugar de una variable binaria "estoy o no en campana" se usa una campana
    gaussiana centrada en la ventana: da un maximo en el pico de actividad y
    decae suavemente hacia los bordes, que es como se comporta la demanda real.
    """
    dias = _dia_del_ano(fechas)
    columnas: dict[str, np.ndarray] = {}
    for nombre, (inicio, fin) in CAMPANAS.items():
        centro, semiancho = _centro_y_ancho(inicio, fin)
        distancia = _distancia_circular(dias, centro)
        sigma = semiancho / 1.6
        columnas[f"campana_{nombre}"] = np.exp(-0.5 * (distancia / sigma) ** 2)
        # Distancia normalizada al pico: informa la "antelacion" a la ventana.
        columnas[f"dist_pico_{nombre}"] = distancia / (DIAS_ANO / 2.0)
    columnas["campana_total"] = np.sum(
        [v for k, v in columnas.items() if k.startswith("campana_")], axis=0
    )
    return pd.DataFrame(columnas, index=pd.RangeIndex(len(dias)))


def terminos_fourier(
    fechas: pd.Series | pd.DatetimeIndex, ordenes: int = 3, prefijo: str = "anual"
) -> pd.DataFrame:
    """Terminos de Fourier de la estacionalidad anual.

    Capturan la forma del ciclo anual con pocos parametros y, a diferencia de
    las variables ficticias por mes, no introducen discontinuidades entre
    periodos contiguos.
    """
    dias = _dia_del_ano(fechas)
    angulo = 2.0 * np.pi * dias / DIAS_ANO
    columnas: dict[str, np.ndarray] = {}
    for k in range(1, ordenes + 1):
        columnas[f"{prefijo}_sin_{k}"] = np.sin(k * angulo)
        columnas[f"{prefijo}_cos_{k}"] = np.cos(k * angulo)
    return pd.DataFrame(columnas, index=pd.RangeIndex(len(dias)))


def caracteristicas_calendario(
    fechas: pd.Series | pd.DatetimeIndex,
    ordenes_fourier: int = 3,
    usar_calendario_agricola: bool = True,
) -> pd.DataFrame:
    """Bloque completo de variables de calendario para una serie de fechas."""
    indice = pd.DatetimeIndex(pd.Series(fechas).to_numpy())
    base = pd.DataFrame(
        {
            "mes": np.asarray(indice.month, dtype=float),
            "trimestre": np.asarray(indice.quarter, dtype=float),
            "semana_del_ano": np.asarray(indice.isocalendar().week, dtype=float),
            "dia_del_ano": np.asarray(indice.dayofyear, dtype=float),
            "ano": np.asarray(indice.year, dtype=float),
        },
        index=pd.RangeIndex(len(indice)),
    )
    bloques = [base, terminos_fourier(indice, ordenes=ordenes_fourier)]
    if usar_calendario_agricola:
        bloques.append(intensidad_campanas(indice))
    resultado = pd.concat(bloques, axis=1)
    resultado.index = pd.Index(range(len(indice)))
    return resultado
