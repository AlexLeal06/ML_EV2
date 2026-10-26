# Predicción de precios de viviendas en Ames, Iowa

Proyecto de Machine Learning para el Caso B de MLY1101. Estima el precio de venta (`SalePrice`) de una vivienda usando sus atributos físicos y de transacción. La solución funciona como apoyo para una estimación inicial; no reemplaza una tasación profesional.

## Problema de negocio

Una empresa inmobiliaria necesita una referencia inicial y consistente para revisar el valor de viviendas comparables. Estimar el precio con datos históricos permite priorizar tasaciones y detectar propiedades que requieren revisión más detallada.

La pregunta que responde el proyecto es: **¿qué precio de venta histórico se puede estimar para una vivienda de Ames dadas sus características observables?**

## Objetivos y métricas de evaluación

1. Preparar el dataset y documentar calidad, faltantes y patrones relevantes.
2. Comparar Ridge, árbol de decisión y Random Forest en datos no vistos.
3. Identificar perfiles de viviendas con K-means, sin usar `SalePrice` para formar los grupos.
4. Guardar una solución reproducible para generar predicciones nuevas.

Las siguientes son **métricas técnicas del modelo**, no indicadores de impacto comercial medido. MAE, RMSE y R² se calculan para el Random Forest seleccionado en el conjunto de test reservado. Silhouette corresponde al análisis K-means con `k=2`; no mide el error de predicción del precio.

| Métrica técnica | Qué mide | Resultado observado e interpretación | Criterio de éxito |
|---|---|---|---|
| **MAE (USD)** | Promedio de las diferencias absolutas entre precios reales y predichos. | **USD 15.670** en el test del Random Forest. En ese conjunto, el error absoluto promedio fue de aproximadamente USD 15.670; no significa que cada predicción tenga ese error. | Definir con el negocio el error promedio máximo aceptable para el uso previsto. |
| **RMSE (USD)** | Raíz del promedio de los errores al cuadrado; da mayor peso a errores absolutos grandes. | **USD 26.572** en el test del Random Forest. Es mayor que el MAE porque penaliza con más fuerza los errores grandes; no indica por sí solo que estos correspondan a viviendas de alto valor. | Acordar un máximo tolerable considerando el costo de los errores grandes. |
| **R²** | Compara la variabilidad de precios explicada por el modelo con la variabilidad observada. | **0,912** en el test: el modelo explica aproximadamente el 91,2% de la variabilidad observada en ese conjunto. No es el porcentaje de predicciones correctas. | Interpretarlo junto con MAE y RMSE; no usarlo como único criterio de aprobación. |
| **Silhouette** | Evalúa cohesión de cada grupo y separación respecto de otros; su rango es de -1 a 1. | **0,315** para K-means con `k=2`, lo que sugiere una separación moderada entre perfiles. Evalúa el clustering, no la precisión del precio predicho. | Interpretarlo junto con los perfiles y su utilidad para el negocio; no existe un umbral universal que garantice clusters útiles. |

Los criterios de éxito aún no se han acordado con una inmobiliaria ni validado en un proceso real. Por ello, los resultados se presentan como evaluación técnica del dataset, y cualquier meta comercial queda **por definir con el negocio**.

Como KPI de negocio propuesto para un futuro piloto, se plantea medir el **porcentaje de estimaciones dentro de un margen de error porcentual aceptable**: `100 × viviendas con error porcentual absoluto dentro del margen acordado / total de viviendas evaluadas`. El margen y la meta deben acordarse con el negocio antes de interpretar este indicador; no se reporta como resultado actual porque todavía no existe ese acuerdo ni una validación en operación.

## Fuente de datos

`data/raw/Ames_Iowa_Housing_Dataset.csv` contiene 2.930 ventas residenciales de Ames, Iowa, entre 2006 y 2010, con 82 columnas. Cada fila es una venta y las variables describen ubicación, superficie, calidad, antigüedad y condiciones de la vivienda.

## Flujo de datos y preparación

```text
data/raw/Ames_Iowa_Housing_Dataset.csv
                 │ validación de SalePrice y duplicados
                 ▼
data/interim/ames_validated.csv
                 │ exclusión de identificadores Order y PID
                 ▼
data/processed/ames_modeling.csv
                 │ split 80/20; selección por CV en train; transformaciones ajustadas dentro de cada fold
                 ▼
models/preprocessor.joblib + models/best_model.joblib
```

- **raw:** archivo original entregado; no se modifica.
- **interim:** registros con `SalePrice` válido y sin duplicados exactos. Se usa para EDA y clustering.
- **processed:** tabla de modelamiento sin `Order` ni `PID`. Mantiene nulos de manera intencional: imputar, escalar o codificar aquí antes de separar train y test provocaría fuga de datos.
- **models:** el preprocesador ajustado guarda la imputación, escalamiento y codificación; el estimador se guarda separado. `src/predict.py` carga ambos en el orden correcto.

Los nulos numéricos se imputan con mediana y las categorías faltantes se reemplazan por `Sin dato`. La ausencia puede significar que una propiedad no tiene esa característica o que el dato es desconocido; el modelo conserva esa señal, aunque no diferencia ambas causas.

## Análisis exploratorio (EDA)

La ejecución actual no detecta duplicados ni precios nulos. `SalePrice` tiene mediana de USD 160.000 y va de USD 12.789 a USD 755.000, con sesgo hacia valores altos. Las asociaciones numéricas más altas con el precio son `Overall Qual`, `Gr Liv Area`, `Garage Cars` y `Year Built`. Estas relaciones son descriptivas, no causales.

Los faltantes se concentran en atributos poco frecuentes, como `Pool QC`, `Misc Feature`, `Alley` y `Fence`. El notebook genera las tablas y figuras de EDA en `reports/`, organizadas por tipo de resultado.

## Metodología CRISP-DM

1. **Comprensión del negocio:** definir la estimación de precio como apoyo a tasación y priorización.
2. **Comprensión de los datos:** revisar estructura, calidad, faltantes, distribución de precios, correlaciones y barrios.
3. **Preparación:** crear las capas interim y processed; excluir IDs; mantener nulos para imputarlos dentro del entrenamiento.
4. **Modelamiento:** comparar Ridge, árbol de decisión y Random Forest mediante validación cruzada de cinco particiones sobre el 80% de entrenamiento; aplicar K-means a ocho variables físicas estandarizadas.
5. **Evaluación:** elegir el modelo por RMSE promedio de validación cruzada, evaluar ese modelo una sola vez en el 20% de test reservado y compararlo con una baseline; revisar predicción versus valor real, residuos y MAE por barrio; seleccionar k con codo y silhouette.
6. **Uso:** guardar el mejor tipo de modelo y el preprocesador ajustados con todos los datos etiquetados.

## Resultados

Comparación de modelos por validación cruzada de cinco particiones dentro del conjunto de entrenamiento (80%). Solo el modelo seleccionado y la baseline se evalúan en test (20% reservado), con semilla 42:

| Modelo | RMSE CV promedio (USD) | Desv. estándar RMSE CV (USD) | MAE test (USD) | RMSE test (USD) | R² test |
|---|---:|---:|---:|---:|---:|
| Random Forest (seleccionado) | 26.373 | 5.500 | 15.670 | 26.572 | 0,912 |
| Ridge | 29.657 | 9.504 | — | — | — |
| Árbol de decisión | 38.228 | 4.904 | — | — | — |
| Baseline mediana | — | — | 63.838 | 94.323 | -0,110 |

La validación cruzada selecciona el modelo con menor RMSE promedio sin consultar el conjunto de test. Las métricas de test se reportan únicamente para el modelo seleccionado y la baseline. Si el RMSE es mayor que el MAE, algunos errores grandes tienen peso relevante, por lo que las propiedades de alto valor deben revisarse manualmente.

Para K-means, `k=2` alcanza el silhouette más alto (0,315), una separación moderada. Un segmento reúne viviendas más nuevas, de mayor calidad y precio mediano; el otro agrupa viviendas más antiguas, más pequeñas y de menor precio mediano. Son perfiles exploratorios, no reglas para fijar un precio individual.

## Reportes generados

La carpeta `reports/` separa cada tipo de evidencia para evitar una carpeta plana con archivos mezclados:

```text
reports/
├── figures/   gráficos de EDA, modelos, residuos y clustering
├── metrics/   métricas del modelo y resumen de ejecución
└── tables/    calidad de datos, asociaciones y perfiles de clusters
```

El notebook genera estos archivos. Las evidencias centrales son `metrics/model_metrics.csv`, `figures/02_model_comparison.png`, `figures/03_test_predictions_residuals.png`, `figures/04_kmeans_selection.png` y `tables/05_cluster_profiles.csv`.

## Estructura del repositorio

```text
config/       Parámetros y rutas del proyecto
data/raw/     Dataset original
data/interim/ Dataset validado para EDA y clustering
data/processed/ Dataset sin IDs, listo para separar train/test
notebooks/    Notebook reproducible de principio a fin
src/          Funciones de preparación, entrenamiento y predicción
models/       Preprocesador y estimador serializados
reports/      Evidencia organizada de la ejecución
README.md     Informe técnico reproducible
```

## Reproducibilidad

Desde la raíz del proyecto:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m notebook
```

En Jupyter, abre `notebooks/procesamiento.ipynb` y ejecuta todas las celdas de arriba hacia abajo. El notebook crea `interim`, `processed`, los reportes y los artefactos de `models/` sin requerir ajustes de rutas.

También puedes entrenar desde consola:

```bash
python src/train.py
```

Para generar predicciones, entrega un `DataFrame` con las columnas de la vivienda a `make_prediction` desde `src/predict.py`. La función carga el preprocesador y el estimador guardados.

## Límites, ética y uso responsable

Los datos son históricos y de una sola ciudad. Características como el barrio pueden reflejar desigualdades socioeconómicas. El notebook presenta MAE por barrio como diagnóstico exploratorio para grupos con al menos 20 casos de test; una diferencia no prueba por sí sola discriminación. No se debe usar este modelo para conceder crédito, determinar acceso a vivienda ni fijar precios automáticamente sin validación temporal, geográfica, de sesgo y revisión humana.
