"""Pruebas de la interfaz de linea de comandos."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from pronostico.cli import main


@pytest.fixture()
def entorno_cli(tmp_path, config_prueba, movimientos, catalogo):
    """Prepara un directorio de trabajo completo con datos y configuracion."""
    for subdirectorio in ("crudos", "procesados", "modelos", "reportes"):
        (tmp_path / subdirectorio).mkdir()
    movimientos.to_csv(tmp_path / "crudos" / "movimientos.csv", index=False)
    catalogo.to_csv(tmp_path / "crudos" / "catalogo_skus.csv", index=False)

    datos = dict(config_prueba.datos)
    datos["proyecto"] = {
        "semilla": 7,
        "directorio_datos_crudos": str(tmp_path / "crudos"),
        "directorio_datos_procesados": str(tmp_path / "procesados"),
        "directorio_modelos": str(tmp_path / "modelos"),
        "directorio_reportes": str(tmp_path / "reportes"),
    }
    ruta_config = tmp_path / "config.yaml"
    ruta_config.write_text(yaml.safe_dump(datos), encoding="utf-8")
    return tmp_path, str(ruta_config)


def test_las_opciones_comunes_funcionan_en_cualquier_orden(entorno_cli):
    """--config y --movimientos deben aceptarse antes y despues del subcomando."""
    tmp_path, ruta_config = entorno_cli
    ruta_movimientos = str(tmp_path / "crudos" / "movimientos.csv")
    assert main(["--config", ruta_config, "diagnostico", "--movimientos", ruta_movimientos]) == 0
    assert main(["--config", ruta_config, "--movimientos", ruta_movimientos, "diagnostico"]) == 0


def test_comando_diagnostico_reporta_datos_aptos(entorno_cli, capsys):
    _, ruta_config = entorno_cli
    assert main(["--config", ruta_config, "diagnostico"]) == 0
    assert "Diagnostico de los datos" in capsys.readouterr().out


def test_el_mapeo_de_columnas_permite_leer_una_exportacion_del_erp(
    tmp_path, config_prueba, movimientos, catalogo
):
    """Con el mapeo, un CSV con nombres y formato de fecha propios funciona igual."""
    for subdirectorio in ("crudos", "modelos", "reportes", "procesados"):
        (tmp_path / subdirectorio).mkdir()

    erp = movimientos.rename(
        columns={"fecha": "FEC_COMP", "sku": "COD_ART", "cantidad": "CANT_FACT"}
    )
    erp["FEC_COMP"] = pd.to_datetime(erp["FEC_COMP"]).dt.strftime("%d/%m/%Y")
    erp.to_csv(tmp_path / "crudos" / "movimientos.csv", index=False)
    catalogo.rename(columns={"sku": "COD_ART"}).to_csv(
        tmp_path / "crudos" / "catalogo_skus.csv", index=False
    )

    datos = dict(config_prueba.datos)
    datos["datos"] = {
        **datos["datos"],
        "mapeo_columnas": {"FEC_COMP": "fecha", "COD_ART": "sku", "CANT_FACT": "cantidad"},
        "formato_fecha": "%d/%m/%Y",
    }
    datos["proyecto"] = {
        "semilla": 7,
        "directorio_datos_crudos": str(tmp_path / "crudos"),
        "directorio_datos_procesados": str(tmp_path / "procesados"),
        "directorio_modelos": str(tmp_path / "modelos"),
        "directorio_reportes": str(tmp_path / "reportes"),
    }
    ruta_config = tmp_path / "config.yaml"
    ruta_config.write_text(yaml.safe_dump(datos), encoding="utf-8")

    assert main(["--config", str(ruta_config), "diagnostico"]) == 0
    assert main(["--config", str(ruta_config), "entrenar"]) == 0
    assert (tmp_path / "modelos" / "modelo_pronostico.joblib").exists()


def test_comando_info(entorno_cli, capsys):
    _, ruta_config = entorno_cli
    assert main(["--config", ruta_config, "info"]) == 0
    assert "Modelos disponibles" in capsys.readouterr().out


def test_flujo_entrenar_predecir_reponer(entorno_cli):
    tmp_path, ruta_config = entorno_cli
    assert main(["--config", ruta_config, "entrenar"]) == 0
    assert (tmp_path / "modelos" / "modelo_pronostico.joblib").exists()

    assert main(["--config", ruta_config, "predecir", "--filas", "3"]) == 0
    pronostico = pd.read_csv(tmp_path / "reportes" / "pronostico.csv")
    assert not pronostico.empty

    assert main(["--config", ruta_config, "reponer", "--filas", "3"]) == 0
    plan = pd.read_csv(tmp_path / "reportes" / "plan_reposicion.csv")
    assert "punto_reorden" in plan.columns


def test_error_controlado_si_faltan_los_datos(tmp_path, capsys):
    ruta_config = tmp_path / "config.yaml"
    ruta_config.write_text(
        yaml.safe_dump({"proyecto": {"directorio_datos_crudos": str(tmp_path / "vacio")}}),
        encoding="utf-8",
    )
    # No hay movimientos: la CLI debe fallar con codigo 1 en vez de reventar.
    assert main(["--config", str(ruta_config), "predecir"]) == 1
