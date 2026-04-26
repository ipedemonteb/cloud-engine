"""
Pipeline minimo de entrenamiento para modelo de scoring crediticio.

Implementa los pasos de agents/MODEL.md:
1) Carga dataset_final.csv
2) Separa X / y
3) train/test split
4) One-hot de actividad
5) Escalado con StandardScaler
6) MLP simple (Keras)
7) Entrenamiento
8) Evaluacion
9) Guardado de artefactos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_VALIDATION_SPLIT = 0.2

TARGET_COLUMN = "score_crediticio"
ID_COLUMN = "nro_id"
CATEGORICAL_COLUMNS = ["actividad"]


def _default_dataset_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data_processed" / "dataset_final.csv"


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path).expanduser().resolve()

    return _default_dataset_path()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {ID_COLUMN, TARGET_COLUMN, *CATEGORICAL_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"El dataset no tiene columnas requeridas: {', '.join(missing)}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("El dataset contiene NaN en score_crediticio")

    out_of_range = (~df[TARGET_COLUMN].between(0.0, 1.0)).sum()
    if out_of_range:
        raise ValueError(
            f"El dataset contiene {out_of_range:,} filas con score_crediticio fuera de [0.0, 1.0]"
        )


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def _build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(input_dim,)),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_feature_columns(columns: list[str], output_path: Path) -> None:
    _ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    default_artifacts_dir = Path(__file__).resolve().parent / "artifacts"

    parser = argparse.ArgumentParser(description="Entrena un MLP para score_crediticio")
    parser.add_argument("--dataset", default=None, help="Ruta a dataset_final.csv")
    parser.add_argument(
        "--model-output",
        default=str(default_artifacts_dir / "modelo_crediticio.keras"),
        help="Ruta de salida del modelo Keras",
    )
    parser.add_argument(
        "--scaler-output",
        default=str(default_artifacts_dir / "scaler.joblib"),
        help="Ruta de salida del StandardScaler",
    )
    parser.add_argument(
        "--columns-output",
        default=str(default_artifacts_dir / "feature_columns.json"),
        help="Ruta de salida de columnas de entrenamiento",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro dataset_final.csv en: {dataset_path}")

    print(f"Cargando dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    _validate_columns(df)

    X, y = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLUMNS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLUMNS)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = _build_model(X_train_scaled.shape[1])
    model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
    )

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test loss (mse): {loss:.6f}")
    print(f"Test MAE: {mae:.6f}")

    preds = model.predict(X_test_scaled, verbose=0)
    print(f"Predicciones ejemplo (primeras 5): {preds[:5].reshape(-1).round(4).tolist()}")

    model_output = Path(args.model_output).expanduser().resolve()
    scaler_output = Path(args.scaler_output).expanduser().resolve()
    columns_output = Path(args.columns_output).expanduser().resolve()

    _ensure_parent_dir(model_output)
    model.save(model_output)

    _ensure_parent_dir(scaler_output)
    joblib.dump(scaler, scaler_output)

    _save_feature_columns(feature_columns, columns_output)

    print(f"Modelo guardado en: {model_output}")
    print(f"Scaler guardado en: {scaler_output}")
    print(f"Columnas guardadas en: {columns_output}")


if __name__ == "__main__":
    main()
