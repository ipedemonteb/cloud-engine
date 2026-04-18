"""
reprocess_targets.py
--------------------
Recalcula targets sobre un dataset ya generado, sin rehacer el pipeline
completo de carga y feature engineering.

Uso desde la raiz del repo:

    uv run python src/reprocess_targets.py

Opcionalmente se pueden pasar overrides de reglas en un JSON:

    uv run python src/reprocess_targets.py \
        --params data_processed/target_params.example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from targets import (
    MIN_FEATURES_FOR_TARGETS,
    TARGET_COLUMNS,
    generar_targets,
    get_default_target_params,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET_DEFAULT_PATH = ROOT / 'data_processed' / 'dataset_final.csv'
OUTPUT_DEFAULT_PATH = ROOT / 'data_processed' / 'dataset_final_retarget.csv'


def _resolver_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _validar_columnas(df: pd.DataFrame, esperadas: list[str], nombre_df: str) -> None:
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"{nombre_df} no tiene columnas esperadas: {', '.join(faltantes)}")


def _cargar_overrides(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError('El archivo de parametros debe ser un objeto JSON (dict).')

    return payload


def _contar_cambios(df_anterior: pd.DataFrame, df_nuevo: pd.DataFrame) -> dict[str, int]:
    cambios: dict[str, int] = {}
    for col in TARGET_COLUMNS:
        if col not in df_anterior.columns:
            continue
        antes = df_anterior[col]
        despues = df_nuevo[col]
        cambios[col] = int((antes != despues).sum())
    return cambios


def _armar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Reprocesa targets sobre un CSV ya generado.',
    )
    parser.add_argument(
        '--input',
        default=str(DATASET_DEFAULT_PATH.relative_to(ROOT)),
        help='CSV de entrada con features y (opcionalmente) targets actuales.',
    )
    parser.add_argument(
        '--output',
        default=str(OUTPUT_DEFAULT_PATH.relative_to(ROOT)),
        help='CSV de salida con targets recalculados.',
    )
    parser.add_argument(
        '--params',
        default=None,
        help='JSON con overrides de reglas de target.',
    )
    parser.add_argument(
        '--show-defaults',
        action='store_true',
        help='Imprime el JSON de parametros por defecto y termina.',
    )
    return parser


def main() -> None:
    parser = _armar_parser()
    args = parser.parse_args()

    if args.show_defaults:
        print(json.dumps(get_default_target_params(), indent=2, ensure_ascii=True))
        return

    input_path = _resolver_path(args.input)
    output_path = _resolver_path(args.output)
    params_path = _resolver_path(args.params) if args.params else None

    if not input_path.exists():
        raise FileNotFoundError(f'No existe el archivo de entrada: {input_path}')

    if params_path is not None and not params_path.exists():
        raise FileNotFoundError(f'No existe el archivo de parametros: {params_path}')

    print('=' * 60)
    print('  Reproceso de targets (sin regenerar features)')
    print('=' * 60)
    print(f'Input : {input_path}')
    print(f'Output: {output_path}')
    print(f"Params: {params_path if params_path is not None else 'defaults'}")

    df_input = pd.read_csv(input_path)
    _validar_columnas(df_input, MIN_FEATURES_FOR_TARGETS, 'dataset de entrada')

    overrides = _cargar_overrides(params_path)

    features_cols = [c for c in df_input.columns if c not in TARGET_COLUMNS]
    df_features = df_input[features_cols].copy()

    df_retarget = generar_targets(df_features, params=overrides)

    columnas_salida = features_cols + TARGET_COLUMNS
    _validar_columnas(df_retarget, columnas_salida, 'dataset reprocesado')
    df_retarget = df_retarget[columnas_salida]

    cambios = _contar_cambios(df_input, df_retarget)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_retarget.to_csv(output_path, index=False)

    print('\nArchivo guardado correctamente.')
    print(f'  Filas   : {len(df_retarget):,}')
    print(f'  Columnas: {len(df_retarget.columns)}')
    if cambios:
        print('  Cambios vs targets previos:')
        for col in TARGET_COLUMNS:
            if col in cambios:
                print(f'    {col}: {cambios[col]:,} filas modificadas')


if __name__ == '__main__':
    main()
