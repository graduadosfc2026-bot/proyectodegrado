"""Carga y acceso tipado a la configuracion del proyecto."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

RUTA_CONFIG_POR_DEFECTO = Path("configuracion/config.yaml")


def raiz_proyecto() -> Path:
    """Directorio raiz del repositorio (dos niveles arriba de este archivo)."""
    return Path(__file__).resolve().parents[2]


def _fusionar(base: dict, extra: Mapping[str, Any]) -> dict:
    """Fusiona `extra` sobre `base` de forma recursiva sin mutar los originales."""
    resultado = copy.deepcopy(base)
    for clave, valor in extra.items():
        if isinstance(valor, Mapping) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


@dataclass
class Config:
    """Contenedor de la configuracion con acceso por ruta punteada.

    Ejemplo:
        >>> cfg = Config({"modelo": {"horizonte": 13}})
        >>> cfg.obtener("modelo.horizonte")
        13
    """

    datos: dict = field(default_factory=dict)
    ruta: Path | None = None

    # ------------------------------------------------------------------ acceso
    def obtener(self, ruta_clave: str, por_defecto: Any = None) -> Any:
        """Devuelve el valor en `ruta_clave` ("seccion.subclave") o `por_defecto`."""
        actual: Any = self.datos
        for parte in ruta_clave.split("."):
            if not isinstance(actual, Mapping) or parte not in actual:
                return por_defecto
            actual = actual[parte]
        return actual

    def requerir(self, ruta_clave: str) -> Any:
        """Igual que `obtener` pero falla si la clave no existe."""
        centinela = object()
        valor = self.obtener(ruta_clave, centinela)
        if valor is centinela:
            raise KeyError(f"Falta la clave de configuracion '{ruta_clave}'")
        return valor

    def seccion(self, nombre: str) -> dict:
        """Devuelve una seccion completa como diccionario (vacio si no existe)."""
        valor = self.obtener(nombre, {})
        return dict(valor) if isinstance(valor, Mapping) else {}

    def con_sobrescrituras(self, sobrescrituras: Mapping[str, Any]) -> "Config":
        """Nueva configuracion con valores sobrescritos por ruta punteada."""
        datos = copy.deepcopy(self.datos)
        for ruta_clave, valor in sobrescrituras.items():
            partes = ruta_clave.split(".")
            nodo = datos
            for parte in partes[:-1]:
                nodo = nodo.setdefault(parte, {})
            nodo[partes[-1]] = valor
        return Config(datos=datos, ruta=self.ruta)

    # ------------------------------------------------------------------- rutas
    def ruta_de(self, ruta_clave: str) -> Path:
        """Resuelve una ruta de la configuracion respecto de la raiz del proyecto."""
        valor = Path(str(self.requerir(ruta_clave)))
        return valor if valor.is_absolute() else raiz_proyecto() / valor

    def asegurar_directorios(self) -> None:
        """Crea los directorios de trabajo declarados en la seccion `proyecto`."""
        for clave in (
            "proyecto.directorio_datos_crudos",
            "proyecto.directorio_datos_procesados",
            "proyecto.directorio_modelos",
            "proyecto.directorio_reportes",
        ):
            if self.obtener(clave) is not None:
                self.ruta_de(clave).mkdir(parents=True, exist_ok=True)


def cargar_config(
    ruta: str | os.PathLike[str] | None = None,
    sobrescrituras: Mapping[str, Any] | None = None,
) -> Config:
    """Carga la configuracion YAML del proyecto.

    Args:
        ruta: archivo YAML a leer. Si es None se usa `configuracion/config.yaml`
            en la raiz del repositorio.
        sobrescrituras: valores a sobrescribir, indexados por ruta punteada.
    """
    ruta_final = Path(ruta) if ruta is not None else raiz_proyecto() / RUTA_CONFIG_POR_DEFECTO
    if not ruta_final.exists():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {ruta_final}")
    with ruta_final.open("r", encoding="utf-8") as archivo:
        datos = yaml.safe_load(archivo) or {}
    if not isinstance(datos, dict):
        raise ValueError(f"La configuracion {ruta_final} debe ser un mapeo YAML")
    if sobrescrituras:
        datos = _fusionar(datos, sobrescrituras)
    return Config(datos=datos, ruta=ruta_final)
