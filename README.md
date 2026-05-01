# cloud-engine

Motor de scoring crediticio basado en datos del BCRA. Incluye:
- Pipeline de preprocessing a partir de archivos oficiales (deudores.txt y 24DSF.txt).
- Entrenamiento de un modelo MLP para predecir `score_crediticio`.
- Utilidades de inferencia y adaptacion de JSON de la API BCRA.

El objetivo es generar un score continuo entre 0.0 y 1.0 que represente el perfil
crediticio esperado a 6 meses, evitando leakage temporal.

## Estructura del repositorio

- `src/preprocessing/`:
  - Construccion del dataset final (features + target).
  - Balanceo o muestreo aleatorio del dataset para entrenamiento.
- `src/model/`:
  - Entrenamiento del modelo.
  - Script de prediccion sobre CSV.
- `src/query/`:
  - Adaptador de JSON de la API BCRA a CSV de entrada para el modelo.
- `data/`:
  - Inputs crudos del BCRA (no versionado).
- `data_processed/`:
  - Salidas intermedias de preprocessing (no versionado).
- `artifacts/`:
  - Artefactos del modelo entrenado (no versionado).

Documentacion adicional:
- `agents/SISTEMA_ACTUAL.md`: detalle del esquema y columnas del dataset.

## Requisitos

- Python 3.12+
- Dependencias definidas en `pyproject.toml`

Instalacion (si usas `uv`):

```bash
uv sync
```

## Datos de entrada

El pipeline espera los archivos del BCRA en:

- `data/202602DEUDORES/deudores.txt` (o `deudores_test.txt`)
- `data/24DSF202602/24DSF.txt` (obligatorio)

Los nombres de carpeta pueden variar, pero hoy el script usa esos paths
por default. Si cambian, hay que ajustar `src/preprocessing/build_dataset.py`.

## Pipeline de preprocessing (dataset final)

Genera `data_processed/dataset_final.csv` con:

- Features actuales: agregadas desde `deudores.txt`.
- Features temporales: calculadas con meses 7..24 del 24DSF.
- Target real `score_crediticio`: derivado de meses 1..6 del 24DSF.

Ejecutar:

```bash
uv run python -m preprocessing.build_dataset
```

Salida esperada:

- `data_processed/dataset_final.csv`

## Reduccion del dataset para entrenamiento

Para reducir el dataset y entrenar mas rapido, usar:

```bash
uv run python -m preprocessing.balance_dataset --strategy balance
```

Opciones utiles:

- `--strategy balance|random`: balancea por bins o muestrea uniforme.
- `--max-total-rows 2000000`: tope total de filas.
- `--output data_processed/dataset_train_balanced.csv`: ruta de salida.

## Entrenamiento del modelo

El entrenamiento guarda artefactos en `artifacts/` (no versionado). Por defecto
usa `data_processed/dataset_train_balanced.csv`.

```bash
uv run python src/model/train_model.py
```

Para entrenar con otro dataset:

```bash
uv run python src/model/train_model.py --dataset /ruta/a/tu_dataset.csv
```

Artefactos generados:

- `artifacts/modelo_crediticio.keras`
- `artifacts/scaler.joblib`
- `artifacts/feature_columns.json`
- `artifacts/feature_fill_values.json`
- `artifacts/train_metrics.json`

## Prediccion sobre CSV

```bash
uv run python src/model/predict.py \
  --input /ruta/a/input.csv \
  --output /ruta/a/predicciones.csv
```

El CSV debe tener el mismo esquema de features que `dataset_final.csv`.

## Prediccion desde JSON de la API BCRA

Si tenes un JSON de la API BCRA (endpoint Historicas), podes convertirlo a CSV
de entrada con:

```bash
uv run python src/query/json_to_model_input.py
```

Por defecto lee `data/query/query.json` (o `query.JSON`) y guarda
`data/query/query_procesed.csv`.

Luego:

```bash
uv run python src/model/predict.py \
  --input data/query/query_procesed.csv \
  --output data/query/predicciones.csv
```

## Notas de produccion

- `artifacts/` no se versiona (estan en `.gitignore`).
- `data/` y `data_processed/` tampoco se versionan.
- `features_desde_api()` en `src/preprocessing/load_data.py` transforma JSON de la API
  BCRA a features compatibles con el modelo.

## Licencia

No especificada.
