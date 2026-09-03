"""Fixtures compartidas por las pruebas."""

from __future__ import annotations

import pandas as pd
import pytest

from pronostico.config import Config
from pronostico.datos.preparacion import preparar_panel
from pronostico.datos.sintetico import generar_conjunto_datos


@pytest.fixture(scope="session")
def datos_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conjunto pequeno de catalogo y movimientos, suficiente para las pruebas."""
    return generar_conjunto_datos(
        n_skus=12,
        fecha_inicio="2021-01-01",
        fecha_fin="2023-12-31",
        semilla=7,
        n_clientes=15,
    )


@pytest.fixture(scope="session")
def catalogo(datos_sinteticos) -> pd.DataFrame:
    return datos_sinteticos[0]


@pytest.fixture(scope="session")
def movimientos(datos_sinteticos) -> pd.DataFrame:
    return datos_sinteticos[1]


@pytest.fixture(scope="session")
def panel(movimientos) -> pd.DataFrame:
    """Panel semanal preparado, con el filtro de historia relajado."""
    return preparar_panel(movimientos, min_periodos=40, min_demanda_total=5)


@pytest.fixture()
def config_prueba() -> Config:
    """Configuracion minima y rapida para las pruebas de integracion."""
    return Config(
        datos={
            "proyecto": {"semilla": 7},
            "datos": {
                "frecuencia": "S",
                "columna_objetivo": "cantidad",
                "min_periodos_historia": 40,
                "min_demanda_total": 5,
                "winsorizar_cuantil": 0.995,
            },
            "caracteristicas": {
                "rezagos": [1, 2, 4, 13],
                "ventanas_moviles": [4, 13],
                "periodo_estacional": 52,
                "ordenes_fourier": 2,
                "usar_calendario_agricola": True,
                "usar_atributos_sku": True,
            },
            "modelo": {
                "nombre": "gbrt_global",
                "horizonte": 4,
                "cuantiles": [0.9],
                "referencias": ["naive", "sba"],
                "hiperparametros": {"max_iter": 30, "max_bins": 32, "early_stopping": False},
            },
            "validacion": {
                "n_origenes": 2,
                "paso_origenes": 4,
                "periodos_prueba": 4,
                "metrica_seleccion": "wape",
                "evaluar_cuantiles": False,
            },
            "inventario": {
                "nivel_servicio": 0.95,
                "lead_time_dias_por_defecto": 21,
                "periodo_revision_dias": 7,
                "cortes_abc": [0.8, 0.95],
                "cortes_xyz": [0.5, 1.0],
            },
        }
    )
