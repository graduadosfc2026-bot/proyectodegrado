"""Registro de modelos disponibles y su construccion por nombre."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .arbol import GBRTGlobal
from .base import ModeloPronostico
from .referencia import (
    SBA,
    TSB,
    Croston,
    MediaEstacional,
    MediaMovil,
    Naive,
    NaiveEstacional,
)

CONSTRUCTORES: dict[str, Callable[..., ModeloPronostico]] = {
    "naive": Naive,
    "naive_estacional": NaiveEstacional,
    "media_movil": MediaMovil,
    "media_estacional": MediaEstacional,
    "croston": Croston,
    "sba": SBA,
    "tsb": TSB,
    "gbrt_global": GBRTGlobal,
}

MODELOS_REFERENCIA = [
    "naive",
    "naive_estacional",
    "media_movil",
    "media_estacional",
    "croston",
    "sba",
    "tsb",
]


def modelos_disponibles() -> list[str]:
    """Nombres de todos los modelos registrados."""
    return sorted(CONSTRUCTORES)


def crear_modelo(nombre: str, **parametros: Any) -> ModeloPronostico:
    """Instancia un modelo por su nombre registrado.

    Args:
        nombre: clave del registro, por ejemplo "gbrt_global" o "sba".
        **parametros: argumentos del constructor del modelo. Los que no
            correspondan al modelo pedido se descartan silenciosamente para
            poder pasar la configuracion completa del proyecto.
    """
    if nombre not in CONSTRUCTORES:
        raise KeyError(
            f"Modelo desconocido: '{nombre}'. Disponibles: {modelos_disponibles()}"
        )
    constructor = CONSTRUCTORES[nombre]
    firma = inspect.signature(constructor.__init__)
    acepta_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in firma.parameters.values()
    )
    if acepta_kwargs:
        # Los modelos por serie heredan `frecuencia` y `periodo_estacional`.
        permitidos = set(firma.parameters) | {"frecuencia", "periodo_estacional"}
    else:
        permitidos = set(firma.parameters)
    filtrados = {k: v for k, v in parametros.items() if k in permitidos}
    return constructor(**filtrados)
