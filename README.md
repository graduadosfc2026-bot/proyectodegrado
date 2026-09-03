# Pronostico de demanda de repuestos de maquinaria agroindustrial

Sistema de inteligencia artificial, entrenable de punta a punta, para pronosticar
la demanda de repuestos de una empresa de maquinaria agroindustrial y traducir
ese pronostico en una politica concreta de reposicion de inventario.

El proyecto no se queda en "predecir una serie de tiempo": resuelve los tres
problemas que hacen dificil este dominio en particular.

| Problema del dominio | Como lo resuelve el proyecto |
| --- | --- |
| La mayoria de los repuestos son de **demanda intermitente**: semanas enteras en cero y pedidos esporadicos. | Perdida de Poisson, modelo global entrenado con todos los SKU a la vez, y comparacion obligatoria contra Croston, SBA y TSB. |
| La demanda no sigue el calendario comercial sino el **calendario de campana** (siembra y cosecha). | Variables continuas de intensidad de campana agricola y terminos de Fourier anuales. |
| Un pronostico sin decision no sirve: hay que decidir **cuanto comprar y cuando**. | Cuantiles de demanda, stock de seguridad, punto de reorden y cantidad sugerida por SKU respetando lote minimo y lead time. |

---

## Inicio rapido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pronostico generar-datos   # datos sinteticos de demostracion
python -m pronostico entrenar        # valida, compara modelos y guarda el ganador
python -m pronostico predecir        # pronostico por SKU y periodo
python -m pronostico reponer         # plan de compras a partir del pronostico
python -m pronostico importancia     # que variables usa el modelo
```

Sin argumentos, todos los comandos leen `configuracion/config.yaml`, toman los
datos de `datos/crudos/` y escriben los resultados en `artefactos/`.

---

## Que hace cada comando

### `generar-datos`

Simula un historico realista (por defecto 120 SKU y 6 anos) con estacionalidad
de campana, tendencia, promociones, elasticidad al precio, intermitencia y
quiebres de stock. Sirve para probar el sistema completo sin datos reales y como
banco de pruebas de "verdad conocida": la demanda se arma con componentes
explicitos que los modelos deben recuperar.

Produce `datos/crudos/movimientos.csv` y `datos/crudos/catalogo_skus.csv`.

### `entrenar`

1. Agrega los movimientos a un panel semanal SKU x periodo **sin huecos** (un
   periodo sin ventas es un cero, no un dato faltante).
2. Clasifica cada SKU por valor (ABC), por variabilidad (XYZ) y por regimen de
   demanda (suave, intermitente, erratica, grumosa).
3. Corre un **backtesting de origen movil**: elige varios origenes, entrena solo
   con lo anterior a cada uno y pronostica el horizonte completo.
4. Compara el modelo principal contra todas las referencias y elige el ganador
   por WAPE.
5. Reentrena el ganador con toda la historia y guarda el artefacto.

### `predecir`

Carga el artefacto y genera el pronostico de los proximos `horizonte` periodos
para cada SKU, con la media condicional y los cuantiles configurados.

### `importancia`

Calcula la importancia por permutacion de cada variable: cuanto empeora el error
al mezclar aleatoriamente esa columna. Es la lectura de negocio del modelo.

### `reponer`

Convierte el pronostico en la decision de compra: demanda esperada durante el
ciclo de reposicion, stock de seguridad para el nivel de servicio objetivo,
punto de reorden y cantidad sugerida. Con `--stock existencias.csv` descuenta
las existencias actuales y devuelve directamente cuanto pedir.

---

## El modelo

El modelo principal (`gbrt_global`) es un ensamble de arboles con impulso de
gradiente (`HistGradientBoostingRegressor`) con cuatro decisiones de diseno:

**Modelo global.** Un unico modelo entrenado con todos los SKU juntos, no uno por
serie. Los repuestos comparten estacionalidad de campana y comportamiento por
familia; un modelo global aprende de los SKU con historia larga y transfiere ese
conocimiento a los de baja rotacion, donde una serie individual no alcanza para
estimar nada.

**Estrategia directa multi-horizonte.** Un estimador por horizonte `h = 1..H`, en
lugar de realimentar el propio pronostico. Evita la acumulacion de error de los
metodos recursivos y deja que cada horizonte use su propia combinacion de
variables (a 1 semana manda el nivel reciente; a 13 semanas, la campana).

**Perdida de Poisson.** La demanda de repuestos son conteos no negativos con
muchos ceros. La perdida de Poisson modela exactamente eso y garantiza
predicciones positivas, a diferencia del error cuadratico.

**Cuantiles nativos.** Ademas de la media condicional se estiman los cuantiles
altos con perdida pinball. Son los que dimensionan el stock de seguridad: para
decidir cuanto comprar importa el escenario malo, no el promedio.

### Variables (unos 80 predictores)

| Bloque | Contenido |
| --- | --- |
| Historia | Rezagos 1 a 52, medias / desvios / maximos moviles, proporcion de ceros, impulso corto contra largo. |
| Intermitencia | Periodos sin demanda, tamano medio de pedido, frecuencia de demanda, ADI acumulado. |
| Calendario agricola | Intensidad continua de siembra gruesa, cosecha gruesa, siembra fina y cosecha fina, mas distancia a cada pico. |
| Estacionalidad | Terminos de Fourier anuales, mes, trimestre, semana del ano. |
| Contexto comercial | Precio medio y precio relativo, dias en promocion, dias con quiebre de stock, transacciones y clientes distintos. |
| Transversal | Demanda media de la familia y de la maquina en el mismo periodo, y la razon del SKU contra ese agregado. |
| Atributos del SKU | Familia, maquina, criticidad, origen del proveedor, costo, lead time, lote minimo. |

Todas las variables de la fila con origen `t` se calculan **unicamente con datos
disponibles hasta `t`**. Hay una prueba automatica dedicada a verificarlo
(`test_no_hay_fuga_de_informacion_desde_el_futuro`).

### Modelos de referencia

Ningun modelo de aprendizaje automatico se justifica si no le gana a estos:

| Modelo | Idea |
| --- | --- |
| `naive` | Repite el ultimo valor. |
| `naive_estacional` | Repite la misma semana de la campana anterior. |
| `media_movil` | Media de las ultimas 13 semanas. |
| `media_estacional` | Perfil estacional historico ajustado al nivel reciente. |
| `croston` | Estandar de la industria para demanda intermitente. |
| `sba` | Croston con la correccion de sesgo de Syntetos-Boylan. |
| `tsb` | Teunter-Syntetos-Babai: actualiza la probabilidad de demanda tambien en los ceros. |

---

## Validacion

**Origen movil, nunca particion aleatoria.** Una particion aleatoria entrena con
el futuro y produce metricas irreproducibles en produccion. Aqui la validacion
imita la operacion real: se fija un origen, se entrena solo con lo anterior y se
pronostica el horizonte completo; se repite sobre varios origenes.

**Metricas.** En demanda intermitente las metricas clasicas enganan: el MAPE es
indefinido cuando la demanda real es cero (la mayoria de los periodos) y el RMSE
premia a quien pronostica siempre cero. Por eso las metricas principales son:

- **WAPE** — error absoluto sobre demanda total. Definido con ceros, se lee como
  "porcentaje de unidades mal pronosticadas". Es la metrica de seleccion.
- **MASE** — error relativo al del modelo ingenuo estacional. Menor que 1 =
  mejor que repetir la campana anterior.
- **Sesgo relativo** — sobre o subestimacion sistematica; una desviacion
  persistente se traduce en sobrestock o en faltantes.
- **Tasa de llenado** — fraccion de la demanda real que se habria podido
  atender comprando exactamente lo pronosticado.
- **Pinball y cobertura** — calibracion de los cuantiles, que es lo que valida
  el dimensionamiento del stock de seguridad.

Los reportes se desagregan por horizonte, por origen, por SKU, por clase ABC y
por regimen de demanda, porque el error medio global esconde que los SKU
grumosos se comportan de manera muy distinta a los suaves.

---

## Resultados sobre el conjunto de demostracion

Backtesting de 4 origenes moviles, horizonte de 13 semanas, 120 SKU y 6 anos de
historia. Metrica de seleccion: WAPE (menor es mejor).

| Modelo | WAPE | MASE | MAE | RMSE | Sesgo relativo | Tasa de llenado |
| --- | --- | --- | --- | --- | --- | --- |
| **`gbrt_global`** | **0.402** | **0.733** | **1.99** | **4.23** | **-1.9%** | 0.790 |
| `naive_estacional` | 0.520 | 0.890 | 2.57 | 5.29 | -6.4% | 0.708 |
| `sba` | 0.539 | 0.859 | 2.66 | 5.30 | +23.8% | 0.850 |
| `croston` | 0.571 | 0.892 | 2.82 | 5.58 | +30.3% | 0.866 |
| `media_movil` | 0.614 | 0.921 | 3.03 | 6.06 | +33.9% | 0.863 |
| `naive` | 0.630 | 0.996 | 3.11 | 6.84 | +22.0% | 0.795 |

El modelo global reduce el WAPE un 23% frente a la mejor referencia y es el
unico practicamente insesgado: Croston y SBA logran buena tasa de llenado pero
a costa de sobrestimar entre un 24% y un 30%, es decir, comprando de mas.

**La ventaja crece con el horizonte**, que es donde importa para comprar
importados con lead times largos:

| WAPE por horizonte | h=1 | h=4 | h=8 | h=13 |
| --- | --- | --- | --- | --- |
| `gbrt_global` | 0.338 | 0.358 | 0.371 | 0.451 |
| `naive_estacional` | 0.482 | 0.478 | 0.463 | 0.594 |
| `sba` | 0.381 | 0.451 | 0.532 | 0.680 |

A una semana, SBA es competitivo; a trece semanas, su error casi duplica al del
modelo global. Los metodos de Croston proyectan una tasa constante y no tienen
manera de anticipar la campana.

**Por regimen de demanda** (WAPE, mejor modelo por fila en negrita):

| Regimen | SKU | `gbrt_global` | `naive_estacional` | `sba` |
| --- | --- | --- | --- | --- |
| Suave | 51 | **0.351** | 0.462 | 0.493 |
| Erratica | 3 | **0.574** | 0.778 | 0.728 |
| Grumosa | 3 | **0.775** | 0.956 | 0.851 |
| Intermitente | 63 | **1.067** | 1.273 | 1.134 |

Una lectura honesta del ultimo renglon: en los SKU intermitentes **todos** los
modelos superan un WAPE de 1, es decir, el error acumulado supera a la demanda
total del periodo. Es lo esperable cuando la demanda son unos pocos pedidos
aislados: el momento exacto es practicamente impredecible. Para esos repuestos
el valor del sistema no esta en acertar la semana sino en estimar bien el
**nivel** y su incertidumbre, que es justamente lo que consume la politica de
inventario a traves de los cuantiles. Por eso el reporte se desagrega por
regimen: promediar todo en una sola cifra esconde esta diferencia.

### Que variables usa el modelo

Importancia por permutacion sobre el horizonte de 1 semana (`python -m
pronostico importancia`):

| # | Variable | Lectura |
| --- | --- | --- |
| 1 | `tamano_medio_pedido` | Cuanto se pide cuando se pide: separa el tamano del pedido de su frecuencia, igual que Croston pero aprendido. |
| 2-3 | `media_movil_8`, `media_movil_4` | Nivel reciente de demanda. |
| 4 | `demanda_media_maquina` | Senal de mercado: como se mueve el resto de los repuestos de esa maquina en el mismo periodo. |
| 5 | `dias_quiebre` | Los quiebres de stock censuran la demanda observada; el modelo aprende a corregir ese sesgo. |
| 7 | `maquina` | Cosechadora, sembradora o pulverizadora: define el perfil de campana. |
| 20+ | `campana_total`, `anual_cos_3`, ... | La estacionalidad de campana aporta despues del nivel, y sobre todo en los horizontes largos. |

Que `dias_quiebre` aparezca entre las cinco primeras variables confirma la
recomendacion de mas abajo: si la empresa no registra los quiebres, el sistema
pierde una de sus senales mas utiles.

---

## De pronostico a decision de compra

```
demanda de reposicion = suma del pronostico durante (lead time + periodo de revision)
stock de seguridad    = z(nivel de servicio) x desvio del error x raiz(periodos del ciclo)
punto de reorden      = demanda de reposicion + stock de seguridad
cantidad sugerida     = redondeo al lote minimo de (punto de reorden - stock actual)
```

El stock de seguridad se puede calcular de dos formas:

- **Parametrica** (por defecto sin cuantiles): usa el desvio del **error de
  pronostico medido en el backtesting**, no la variabilidad de la demanda. Es la
  diferencia entre dimensionar contra la incertidumbre real del modelo y
  dimensionar contra el ruido historico.
- **Por cuantiles** (por defecto si el modelo los estima): usa la brecha entre
  el cuantil alto y la media que estima el propio modelo. No supone normalidad
  del error, lo que importa en repuestos con demanda grumosa, donde la
  distribucion es muy asimetrica. La brecha por periodo se agrega con la raiz
  del numero de periodos del ciclo, no sumando los cuantiles: **el cuantil de
  la suma no es la suma de los cuantiles**, y sumarlos sobredimensiona el stock.

---

## Estructura del repositorio

```
configuracion/config.yaml        Toda la parametrizacion del sistema
src/pronostico/
  config.py                      Carga de configuracion con acceso por ruta punteada
  cli.py                         Interfaz de linea de comandos
  datos/
    esquema.py                   Esquema canonico y validacion de las tablas de entrada
    sintetico.py                 Generador de datos de demostracion
    preparacion.py               Panel regular, censura, filtros, ABC-XYZ y regimen
  caracteristicas/
    calendario.py                Campana agricola y estacionalidad de Fourier
    constructor.py               Matriz de predictores y objetivos multi-horizonte
  modelos/
    base.py                      Interfaz comun de todos los modelos
    referencia.py                Naive, estacional, media movil, Croston, SBA, TSB
    arbol.py                     Modelo global de gradient boosting (principal)
    registro.py                  Fabrica de modelos por nombre
  evaluacion/
    metricas.py                  WAPE, MASE, pinball, cobertura, tasa de llenado
    backtesting.py               Validacion por origen movil y comparacion de modelos
  inventario/
    politica.py                  Stock de seguridad, punto de reorden, cantidad a pedir
  pipeline/
    entrenamiento.py             Pipeline completo de entrenamiento
    prediccion.py                Pipeline de prediccion y plan de reposicion
tests/                           Pruebas unitarias y de integracion
```

---

## Usar los datos reales de la empresa

El generador sintetico es solo para demostracion. Para conectar el sistema a los
datos reales alcanza con exportar dos CSV con el esquema de
`src/pronostico/datos/esquema.py`:

**`movimientos.csv`** — una fila por linea de venta o pedido.

| Columna | Obligatoria | Descripcion |
| --- | --- | --- |
| `fecha` | si | Fecha del movimiento. |
| `sku` | si | Codigo del repuesto. |
| `cantidad` | si | Unidades. Las devoluciones (negativas) se llevan a cero. |
| `precio_unitario` | no | Precio efectivo de venta. |
| `cliente_id` | no | Permite contar clientes distintos por periodo. |
| `canal` | no | Mostrador, taller, mayorista, etc. |
| `promocion` | no | 1 si el movimiento ocurrio en promocion. |
| `quiebre_stock` | no | 1 si ese dia no habia stock (demanda censurada). |

**`catalogo_skus.csv`** — una fila por SKU: `sku` (obligatoria), `familia`,
`maquina`, `criticidad`, `costo_unitario`, `precio_lista`, `lead_time_dias`,
`origen_proveedor`, `lote_minimo`.

Luego:

```bash
python -m pronostico entrenar --movimientos ruta/a/movimientos.csv \
                              --catalogo ruta/a/catalogo_skus.csv
```

Dos recomendaciones al pasar a datos reales:

1. **Registrar los quiebres de stock.** Sin esa columna, la demanda perdida
   entra al modelo como demanda cero y el sistema aprende a comprar de menos
   justo en los repuestos que mas faltan.
2. **Revisar el calendario agricola.** Las ventanas de `CAMPANAS` en
   `caracteristicas/calendario.py` corresponden al Cono Sur. Para otra region o
   para otros cultivos hay que redefinirlas.

---

## Configuracion

Todo se controla desde `configuracion/config.yaml`. Los parametros que mas
cambian el resultado:

| Parametro | Efecto |
| --- | --- |
| `datos.frecuencia` | `S` semanal o `MS` mensual. Al pasar a mensual hay que ajustar `periodo_estacional` a 12 y los rezagos. |
| `modelo.horizonte` | Periodos a pronosticar. Debe cubrir el lead time mas largo del catalogo. |
| `modelo.cuantiles` | Cuantiles a estimar. Cada uno multiplica el tiempo de entrenamiento. |
| `validacion.n_origenes` | Mas origenes dan una medicion mas estable y tardan mas. |
| `validacion.evaluar_cuantiles` | Evalua tambien la calibracion de los cuantiles en el backtesting. |
| `inventario.nivel_servicio` | Nivel de servicio objetivo. De 0.95 a 0.99 el stock de seguridad crece cerca de un 40%. |

---

## Pruebas

```bash
pytest              # suite completa
pytest -k fuga      # la prueba de no fuga de informacion temporal
```

La suite cubre metricas, preparacion de datos, ausencia de fuga temporal,
modelos de referencia contra su definicion analitica, backtesting, politica de
inventario y el pipeline completo de punta a punta.
