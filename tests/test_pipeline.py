"""Pruebas de integracion del pipeline completo."""

from __future__ import annotations

import pandas as pd
import pytest

from pronostico.config import Config, cargar_config
from pronostico.datos import esquema as esq
from pronostico.pipeline.entrenamiento import construir_fabricas, entrenar_pipeline
from pronostico.pipeline.prediccion import (
    cargar_artefacto,
    plan_de_reposicion,
    predecir_demanda,
    preparar_historia,
)


def test_configuracion_del_repositorio_es_valida():
    config = cargar_config()
    assert config.obtener("modelo.horizonte") >= 1
    assert config.obtener("datos.frecuencia") in {"S", "MS", "M"}
    assert 0.5 <= config.obtener("inventario.nivel_servicio") < 1.0


def test_acceso_y_sobrescritura_de_configuracion():
    config = Config(datos={"modelo": {"horizonte": 13}})
    assert config.obtener("modelo.horizonte") == 13
    assert config.obtener("modelo.inexistente", "def") == "def"
    with pytest.raises(KeyError):
        config.requerir("modelo.inexistente")
    modificada = config.con_sobrescrituras({"modelo.horizonte": 4, "nuevo.valor": 1})
    assert modificada.obtener("modelo.horizonte") == 4
    assert modificada.obtener("nuevo.valor") == 1
    assert config.obtener("modelo.horizonte") == 13  # el original no se modifica


def test_las_fabricas_incluyen_principal_y_referencias(config_prueba):
    fabricas = construir_fabricas(config_prueba)
    assert set(fabricas) == {"gbrt_global", "naive", "sba"}
    sin_cuantiles = construir_fabricas(config_prueba, incluir_cuantiles=False)
    assert sin_cuantiles["gbrt_global"]().cuantiles == []


def test_pipeline_completo_entrena_predice_y_repone(
    movimientos, catalogo, config_prueba, tmp_path
):
    config = config_prueba.con_sobrescrituras(
        {
            "proyecto.directorio_modelos": str(tmp_path / "modelos"),
            "proyecto.directorio_reportes": str(tmp_path / "reportes"),
            "proyecto.directorio_datos_crudos": str(tmp_path / "crudos"),
            "proyecto.directorio_datos_procesados": str(tmp_path / "procesados"),
        }
    )

    resultado = entrenar_pipeline(movimientos, catalogo, config)
    assert resultado.ruta_modelo is not None and resultado.ruta_modelo.exists()
    assert not resultado.metricas["global"].empty
    assert resultado.metadatos["skus"] == resultado.panel[esq.COL_SKU].nunique()
    assert (tmp_path / "reportes" / "clasificacion_skus.csv").exists()
    assert (tmp_path / "reportes" / "metricas_global.csv").exists()

    artefacto = cargar_artefacto(config)
    assert artefacto.nombre_modelo == resultado.nombre_modelo
    assert artefacto.error_por_sku is not None

    historia = preparar_historia(movimientos, config)
    pronostico = predecir_demanda(artefacto, historia)
    horizonte = config.obtener("modelo.horizonte")
    assert len(pronostico) == historia[esq.COL_SKU].nunique() * horizonte
    assert (pronostico[esq.COL_PREDICCION] >= 0).all()
    assert pronostico["fecha_objetivo"].min() > historia[esq.COL_FECHA].max()

    plan = plan_de_reposicion(artefacto, pronostico, config)
    assert len(plan) == historia[esq.COL_SKU].nunique()
    assert (plan["punto_reorden"] >= plan["demanda_esperada_reposicion"] - 1e-9).all()
    assert (plan["cantidad_sugerida"] >= 0).all()


def test_entrenamiento_sin_backtesting_es_mas_directo(
    movimientos, catalogo, config_prueba, tmp_path
):
    config = config_prueba.con_sobrescrituras(
        {
            "proyecto.directorio_modelos": str(tmp_path / "modelos"),
            "proyecto.directorio_reportes": str(tmp_path / "reportes"),
            "proyecto.directorio_datos_crudos": str(tmp_path / "crudos"),
            "proyecto.directorio_datos_procesados": str(tmp_path / "procesados"),
        }
    )
    resultado = entrenar_pipeline(
        movimientos, catalogo, config, ejecutar_backtesting=False, guardar=False
    )
    assert resultado.metricas == {}
    assert resultado.resultados_backtest.empty
    assert resultado.nombre_modelo == "gbrt_global"


def test_falla_con_movimientos_vacios(config_prueba):
    vacios = pd.DataFrame(columns=[esq.COL_FECHA, esq.COL_SKU, esq.COL_CANTIDAD])
    with pytest.raises(Exception):
        entrenar_pipeline(vacios, None, config_prueba, guardar=False)
