# Atajos de desarrollo del proyecto.

VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: instalar datos entrenar predecir reponer importancia test limpiar todo

instalar:      ## Crea el entorno virtual e instala el proyecto
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

datos:         ## Genera el conjunto de datos sintetico de demostracion
	$(PY) -m pronostico generar-datos

entrenar:      ## Entrena, valida contra las referencias y guarda el modelo
	$(PY) -m pronostico entrenar

predecir:      ## Genera el pronostico con el modelo entrenado
	$(PY) -m pronostico predecir

reponer:       ## Calcula el plan de reposicion
	$(PY) -m pronostico reponer

importancia:   ## Calcula la importancia por permutacion de las variables
	$(PY) -m pronostico importancia

test:          ## Corre la suite de pruebas
	$(PY) -m pytest -q

todo: datos entrenar predecir reponer  ## Ejecuta el flujo completo

limpiar:       ## Borra datos generados y artefactos
	rm -rf datos/crudos/*.csv datos/procesados/*.csv
	rm -rf artefactos/modelos/*.joblib artefactos/reportes/*.csv artefactos/reportes/*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
