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
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
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
SCORE_BINS = [0.0, 0.25, 0.50, 0.75, 1.000001]
SCORE_BIN_LABELS = ["b0_0_025", "b1_025_050", "b2_050_075", "b3_075_100"]


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


def _save_metrics(metrics: dict, output_path: Path) -> None:
    _ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)


def _to_bins(values: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, bins=SCORE_BINS, right=False) - 1
    return np.clip(idx, 0, len(SCORE_BIN_LABELS) - 1)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def _metrics_por_bin(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    bins_true = _to_bins(y_true)
    filas: list[dict] = []

    for i, label in enumerate(SCORE_BIN_LABELS):
        mask = bins_true == i
        n = int(mask.sum())
        if n == 0:
            filas.append({
                "bin": label,
                "rows": 0,
                "mae": None,
                "rmse": None,
            })
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        mse = mean_squared_error(y_t, y_p)
        filas.append({
            "bin": label,
            "rows": n,
            "mae": float(mean_absolute_error(y_t, y_p)),
            "rmse": float(np.sqrt(mse)),
        })

    return filas


def _bin_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_bin = _to_bins(y_true)
    y_pred_bin = _to_bins(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        labels=list(range(len(SCORE_BIN_LABELS))),
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true_bin,
        y_pred_bin,
        labels=list(range(len(SCORE_BIN_LABELS))),
    )

    por_clase = []
    for i, label in enumerate(SCORE_BIN_LABELS):
        por_clase.append({
            "bin": label,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    return {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "macro_f1": float(
            f1_score(
                y_true_bin,
                y_pred_bin,
                labels=list(range(len(SCORE_BIN_LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_bin,
                y_pred_bin,
                labels=list(range(len(SCORE_BIN_LABELS))),
                average="weighted",
                zero_division=0,
            )
        ),
        "by_bin": por_clase,
        "confusion_matrix": cm.astype(int).tolist(),
        "bin_labels": SCORE_BIN_LABELS,
    }


def _print_metricas(nombre: str, metricas: dict) -> None:
    print(f"{nombre}:")
    print(f"  MAE      : {metricas['mae']:.6f}")
    print(f"  RMSE     : {metricas['rmse']:.6f}")
    print(f"  MSE      : {metricas['mse']:.6f}")
    print(f"  R2       : {metricas['r2']:.6f}")
    print(f"  MedAE    : {metricas['median_ae']:.6f}")


def _print_metricas_por_bin(metricas_por_bin: list[dict]) -> None:
    print("Metricas por bin (segun y_true):")
    for row in metricas_por_bin:
        if row["rows"] == 0:
            print(f"  {row['bin']}: 0 filas")
            continue
        print(
            f"  {row['bin']}: filas={row['rows']:,} "
            f"MAE={row['mae']:.6f} RMSE={row['rmse']:.6f}"
        )


def _print_metricas_bins_clasificacion(metricas: dict) -> None:
    print("Clasificacion por bins (0.25):")
    print(f"  Accuracy   : {metricas['accuracy']:.6f}")
    print(f"  Macro F1   : {metricas['macro_f1']:.6f}")
    print(f"  Weighted F1: {metricas['weighted_f1']:.6f}")
    print("  Por bin:")
    for row in metricas["by_bin"]:
        print(
            f"    {row['bin']}: support={row['support']:,} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}"
        )


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
    parser.add_argument(
        "--metrics-output",
        default=str(default_artifacts_dir / "train_metrics.json"),
        help="Ruta de salida de metricas de entrenamiento/evaluacion",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro dataset_final.csv en: {dataset_path}")

    io_start = time.perf_counter()
    print(f"Cargando dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    _validate_columns(df)
    io_seconds = time.perf_counter() - io_start

    X, y = _prepare_features(df)
    y_bins = _to_bins(y.to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_bins,
    )

    prep_start = time.perf_counter()
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLUMNS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLUMNS)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    prep_seconds = time.perf_counter() - prep_start

    train_start = time.perf_counter()
    model = _build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
    )
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    y_test_np = y_test.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float)

    preds_test = model.predict(X_test_scaled, verbose=0).reshape(-1)
    preds_train = model.predict(X_train_scaled, verbose=0).reshape(-1)
    eval_seconds = time.perf_counter() - eval_start

    baseline_pred = np.full_like(y_test_np, fill_value=float(np.mean(y_train_np)), dtype=float)

    train_metrics = _regression_metrics(y_train_np, preds_train)
    test_metrics = _regression_metrics(y_test_np, preds_test)
    baseline_metrics = _regression_metrics(y_test_np, baseline_pred)
    test_metrics_by_bin = _metrics_por_bin(y_test_np, preds_test)
    test_bin_clf = _bin_classification_metrics(y_test_np, preds_test)

    print()
    _print_metricas("Train", train_metrics)
    print()
    _print_metricas("Test", test_metrics)
    print()
    _print_metricas("Baseline (pred constante = media train)", baseline_metrics)
    print()
    _print_metricas_por_bin(test_metrics_by_bin)
    print()
    _print_metricas_bins_clasificacion(test_bin_clf)
    print(f"Predicciones ejemplo (primeras 5): {preds_test[:5].round(4).tolist()}")

    model_output = Path(args.model_output).expanduser().resolve()
    scaler_output = Path(args.scaler_output).expanduser().resolve()
    columns_output = Path(args.columns_output).expanduser().resolve()
    metrics_output = Path(args.metrics_output).expanduser().resolve()

    _ensure_parent_dir(model_output)
    model.save(model_output)

    _ensure_parent_dir(scaler_output)
    joblib.dump(scaler, scaler_output)

    _save_feature_columns(feature_columns, columns_output)

    train_history = history.history
    metrics_payload = {
        "dataset": str(dataset_path),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "features_after_encoding": int(X_train_scaled.shape[1]),
        "config": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "score_bins": SCORE_BINS,
            "score_bin_labels": SCORE_BIN_LABELS,
        },
        "timings_seconds": {
            "load_dataset": round(io_seconds, 4),
            "prepare_features": round(prep_seconds, 4),
            "train": round(train_seconds, 4),
            "evaluate": round(eval_seconds, 4),
            "total": round(time.perf_counter() - total_start, 4),
        },
        "keras_history": {
            "loss": [float(v) for v in train_history.get("loss", [])],
            "mae": [float(v) for v in train_history.get("mae", [])],
            "val_loss": [float(v) for v in train_history.get("val_loss", [])],
            "val_mae": [float(v) for v in train_history.get("val_mae", [])],
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "test_metrics_by_bin": test_metrics_by_bin,
        "test_bin_classification": test_bin_clf,
    }
    _save_metrics(metrics_payload, metrics_output)

    print(f"Modelo guardado en: {model_output}")
    print(f"Scaler guardado en: {scaler_output}")
    print(f"Columnas guardadas en: {columns_output}")
    print(f"Metricas guardadas en: {metrics_output}")


if __name__ == "__main__":
    main()
