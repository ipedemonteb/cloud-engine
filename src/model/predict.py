"""
Utilidad de inferencia para modelo de scoring crediticio.

Carga:
- modelo Keras entrenado
- scaler de entrenamiento
- columnas de features usadas en train

Permite predecir sobre un CSV con esquema de dataset_final.csv
(ignorando nro_id/score_crediticio si existen).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from tensorflow.keras.models import load_model


ID_COLUMN = "nro_id"
TARGET_COLUMN = "score_crediticio"
CATEGORICAL_COLUMNS = ["actividad"]


def parse_args() -> argparse.Namespace:
    default_artifacts_dir = Path(__file__).resolve().parent / "artifacts"

    parser = argparse.ArgumentParser(description="Predice score_crediticio con modelo entrenado")
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument(
        "--model",
        default=str(default_artifacts_dir / "modelo_crediticio.keras"),
        help="Ruta del modelo Keras",
    )
    parser.add_argument(
        "--scaler",
        default=str(default_artifacts_dir / "scaler.joblib"),
        help="Ruta del scaler",
    )
    parser.add_argument(
        "--columns",
        default=str(default_artifacts_dir / "feature_columns.json"),
        help="Ruta del JSON de columnas de features",
    )
    parser.add_argument(
        "--output",
        default=str(default_artifacts_dir / "predicciones.csv"),
        help="Ruta de salida del CSV con predicciones",
    )
    return parser.parse_args()


def _read_feature_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("feature_columns.json tiene formato invalido")
    return data


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    ids = df[ID_COLUMN] if ID_COLUMN in df.columns else None

    X = df.copy()
    for col in [ID_COLUMN, TARGET_COLUMN]:
        if col in X.columns:
            X = X.drop(columns=[col])

    if "actividad" not in X.columns:
        X["actividad"] = "desconocido"

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS)
    return X, ids


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    scaler_path = Path(args.scaler).expanduser().resolve()
    columns_path = Path(args.columns).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in [input_path, model_path, scaler_path, columns_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontro archivo requerido: {path}")

    df = pd.read_csv(input_path)
    X, ids = _build_features(df)

    feature_columns = _read_feature_columns(columns_path)
    X = X.reindex(columns=feature_columns, fill_value=0)

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0).reshape(-1)

    out = pd.DataFrame({"score_crediticio_pred": preds})
    if ids is not None:
        out.insert(0, ID_COLUMN, ids.values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")


if __name__ == "__main__":
    main()
