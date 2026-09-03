"""Configuracion centralizada del logging del proyecto."""

from __future__ import annotations

import logging
import sys

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configurado = False


def configurar_logging(nivel: int | str = logging.INFO) -> None:
    """Instala un handler de consola unico para todo el paquete."""
    global _configurado
    raiz = logging.getLogger("pronostico")
    if _configurado:
        raiz.setLevel(nivel)
        return
    manejador = logging.StreamHandler(stream=sys.stderr)
    manejador.setFormatter(logging.Formatter(_FORMATO, datefmt="%H:%M:%S"))
    raiz.addHandler(manejador)
    raiz.setLevel(nivel)
    raiz.propagate = False
    _configurado = True


def obtener_logger(nombre: str) -> logging.Logger:
    """Devuelve un logger hijo del logger raiz del proyecto."""
    configurar_logging()
    if nombre.startswith("pronostico"):
        return logging.getLogger(nombre)
    return logging.getLogger(f"pronostico.{nombre}")
