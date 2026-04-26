"""
reprocess_targets.py
--------------------
Reconstruye el dataset final de scoring con el pipeline actualizado.

Uso desde la raiz del repo:

    uv run python src/reprocess_targets.py

Opcionalmente se pueden pasar paths de input/output:

    uv run python src/reprocess_targets.py \
        --deudores data/202602DEUDORES/deudores.txt \
        --dsf24 data/24DSF202602/24DSF.txt \
        --output data_processed/dataset_final_retarget.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_dataset import (
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    PERSONA_HUMANA_ONLY,
    TARGETS,
    TEMPORALES_ESPERADAS,
    _resolver_path_preferido,
    _validar_columnas,
    _validar_rangos_temporales,
)
from features import combinar_features
from load_data import cargar_24dsf, cargar_deudores
from targets import generar_targets


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DEFAULT_DEUDORES_DIR = DATA_DIR / '202602DEUDORES'
DEFAULT_DSF24_DIR = DATA_DIR / '24DSF202602'

DATASET_DEFAULT_PATH = ROOT / 'data_processed' / 'dataset_final.csv'
OUTPUT_DEFAULT_PATH = ROOT / 'data_processed' / 'dataset_final_retarget.csv'


def _resolver_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _armar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Reconstruye dataset de scoring con score real del 24DSF.',
    )
    parser.add_argument(
        '--deudores',
        default=None,
        help='Path a deudores.txt (si no se pasa, usa ruta por defecto con fallback a test).',
    )
    parser.add_argument(
        '--dsf24',
        default=None,
        help='Path a 24DSF.txt (obligatorio para generar target real).',
    )
    parser.add_argument(
        '--output',
        default=str(OUTPUT_DEFAULT_PATH.relative_to(ROOT)),
        help='CSV de salida reconstruido.',
    )
    parser.add_argument(
        '--baseline',
        default=str(DATASET_DEFAULT_PATH.relative_to(ROOT)),
        help='CSV baseline para comparar cambios (opcional).',
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='No comparar contra baseline aunque exista.',
    )
    return parser


def _comparar_con_baseline(path_baseline: Path, df_nuevo: pd.DataFrame) -> None:
    if not path_baseline.exists():
        print(f"Baseline no encontrado: {path_baseline}")
        return

    df_base = pd.read_csv(path_baseline)
    if 'nro_id' not in df_base.columns:
        print('Baseline no tiene columna nro_id; se omite comparacion.')
        return

    base_cols = set(df_base.columns)
    new_cols = set(df_nuevo.columns)
    solo_base = sorted(base_cols - new_cols)
    solo_nuevo = sorted(new_cols - base_cols)

    print('\nComparacion contra baseline:')
    if solo_base:
        print('  Columnas solo en baseline: ' + ', '.join(solo_base))
    if solo_nuevo:
        print('  Columnas solo en nuevo:    ' + ', '.join(solo_nuevo))
    if not solo_base and not solo_nuevo:
        print('  Columnas: sin diferencias')

    comunes = [c for c in df_nuevo.columns if c in df_base.columns]
    if not comunes:
        return

    base_idx = df_base.set_index('nro_id')
    new_idx = df_nuevo.set_index('nro_id')
    comunes_ids = base_idx.index.intersection(new_idx.index)
    print(f"  CUITs comunes para comparar: {len(comunes_ids):,}")
    if len(comunes_ids) == 0:
        return

    target = TARGETS[0]
    if target in comunes:
        cambios_target = int(
            (
                base_idx.loc[comunes_ids, target]
                != new_idx.loc[comunes_ids, target]
            ).sum()
        )
        print(f"  Filas con target cambiado ({target}): {cambios_target:,}")


def main() -> None:
    parser = _armar_parser()
    args = parser.parse_args()

    output_path = _resolver_path(args.output)
    baseline_path = _resolver_path(args.baseline)

    if args.deudores:
        deudores_path = _resolver_path(args.deudores)
    else:
        deudores_path = _resolver_path_preferido(
            DEFAULT_DEUDORES_DIR / 'deudores.txt',
            DEFAULT_DEUDORES_DIR / 'deudores_test.txt',
        )

    if args.dsf24:
        dsf24_path = _resolver_path(args.dsf24)
    else:
        dsf24_path = DEFAULT_DSF24_DIR / '24DSF.txt'

    if deudores_path is None or not deudores_path.exists():
        raise FileNotFoundError(
            f'No se encontro deudores.txt ni deudores_test.txt en {DEFAULT_DEUDORES_DIR}'
        )

    if not dsf24_path.exists():
        raise FileNotFoundError(
            f'No se encontro 24DSF.txt en {DEFAULT_DSF24_DIR}. Sin ese archivo no hay target real.'
        )

    print('=' * 60)
    print('  Reconstruccion dataset final (scoring crediticio)')
    print('=' * 60)
    print(f'Archivo deudores: {deudores_path}')
    print(f'Archivo 24DSF:    {dsf24_path}')
    print(f'Output:           {output_path}')
    print(
        'Filtro personas humanas: '
        + ('activo (20/23/24/27)' if PERSONA_HUMANA_ONLY else 'desactivado')
    )

    df_actuales = cargar_deudores(
        deudores_path,
        persona_humana_only=PERSONA_HUMANA_ONLY,
    )
    _validar_columnas(
        df_actuales,
        [
            'nro_id', 'situacion', 'prestamos_total', 'dias_atraso_max',
            'tiene_garantia_a', 'ratio_cobertura', 'refinanciado',
            'proceso_judicial', 'recategorizado', 'irrecuperable',
            'cant_entidades', 'actividad',
        ],
        'df_actuales',
    )

    if df_actuales['nro_id'].duplicated().any():
        raise ValueError('df_actuales tiene CUITs duplicados; se esperaba una fila por nro_id')

    df_temporales, df_target = cargar_24dsf(
        dsf24_path,
        persona_humana_only=PERSONA_HUMANA_ONLY,
    )
    _validar_columnas(df_temporales, ['nro_id', *TEMPORALES_ESPERADAS], 'df_temporales')
    _validar_columnas(df_target, ['nro_id', 'score_crediticio'], 'df_target')
    _validar_rangos_temporales(df_temporales)

    if df_temporales['nro_id'].duplicated().any():
        raise ValueError('df_temporales tiene CUITs duplicados; se esperaba una fila por nro_id')
    if df_target['nro_id'].duplicated().any():
        raise ValueError('df_target tiene CUITs duplicados; se esperaba una fila por nro_id')

    invalid_target = (~df_target['score_crediticio'].between(0.0, 1.0)).sum()
    if invalid_target:
        raise ValueError(
            f"df_target contiene {invalid_target:,} filas con score_crediticio fuera de [0.0, 1.0]"
        )
    if df_target['score_crediticio'].isna().any():
        raise ValueError('df_target contiene valores NaN en score_crediticio')

    df_features = combinar_features(df_actuales, df_temporales)
    df_final = generar_targets(df_features, df_target)
    _validar_columnas(df_final, TARGETS, 'df_final')

    columnas_finales = ['nro_id'] + FEATURES_NUMERICAS + FEATURES_CATEGORICAS + TARGETS
    _validar_columnas(df_final, columnas_finales, 'df_final (columnas exportables)')
    df_export = df_final[columnas_finales]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_csv(output_path, index=False)

    print('\nArchivo guardado correctamente.')
    print(f'  Filas   : {len(df_export):,}')
    print(f'  Columnas: {len(df_export.columns)}')

    if not args.skip_baseline:
        _comparar_con_baseline(baseline_path, df_export)


if __name__ == '__main__':
    main()
