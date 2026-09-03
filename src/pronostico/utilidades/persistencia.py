"""Guardado y lectura de artefactos (modelos, tablas y reportes JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def marca_de_tiempo() -> str:
    """Marca de tiempo UTC apta para nombres de archivo."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def guardar_modelo(objeto: Any, ruta: str | Path) -> Path:
    """Serializa un objeto entrenado con joblib."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objeto, destino)
    return destino


def cargar_modelo(ruta: str | Path) -> Any:
    """Deserializa un objeto guardado con `guardar_modelo`."""
    origen = Path(ruta)
    if not origen.exists():
        raise FileNotFoundError(f"No existe el artefacto de modelo: {origen}")
    return joblib.load(origen)


def guardar_tabla(df: pd.DataFrame, ruta: str | Path) -> Path:
    """Guarda un DataFrame en CSV (UTF-8, sin indice)."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, encoding="utf-8")
    return destino


def leer_tabla(ruta: str | Path, columnas_fecha: list[str] | None = None) -> pd.DataFrame:
    """Lee un CSV parseando las columnas de fecha indicadas."""
    origen = Path(ruta)
    if not origen.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {origen}")
    return pd.read_csv(origen, parse_dates=columnas_fecha or [], encoding="utf-8")


def guardar_json(contenido: Any, ruta: str | Path) -> Path:
    """Guarda un diccionario o lista como JSON legible."""
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False, default=str)
    return destino
