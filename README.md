# cloud-engine

Motor de preprocessing para construir un dataset de scoring crediticio BCRA.

## Estado actual

El pipeline vigente genera un dataset final con target continuo `score_crediticio` (`0.0` a `1.0`), evitando leakage temporal:

- Target: meses `m01..m06` del `24DSF.txt`
- Features temporales: meses `m07..m24`
- Features actuales: agregadas desde `deudores.txt`

La documentación detallada del sistema y de todas las columnas exportadas está en:

- `agents/SISTEMA_ACTUAL.md`

## Cómo correr el pipeline

Desde `src`:

```bash
uv run python -m preprocessing.build_dataset
```

Salida esperada:

- `src/data_processed/dataset_final.csv`

## Entrenamiento del modelo

El entrenamiento guarda los artefactos en `artifacts/` (no versionado).
Por defecto usa `data_processed/dataset_train_balanced.csv` como dataset.

```bash
uv run python src/model/train_model.py
```

Para indicar otro dataset:

```bash
uv run python src/model/train_model.py --dataset /ruta/a/tu_dataset.csv
```

## Prediccion

El script de inferencia usa por defecto los artefactos en `artifacts/`.

```bash
uv run python src/model/predict.py --input /ruta/a/input.csv --output /ruta/a/predicciones.csv
```

## Inputs esperados

- `src/data/202602DEUDORES/deudores.txt` (o `deudores_test.txt`)
- `src/data/24DSF202602/24DSF.txt` (obligatorio)
