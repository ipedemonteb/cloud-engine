"""
targets.py
----------
Genera las variables target a partir de las features ya construidas.

El módulo ahora permite inyectar parámetros en runtime para recalcular
targets sin reconstruir todo el pipeline de features.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMNS = ['monto_sugerido', 'cuotas_cod', 'cuotas_valor']

MIN_FEATURES_FOR_TARGETS = [
    'situacion',
    'prestamos_total',
    'dias_atraso_max',
    'proceso_judicial',
    'irrecuperable',
]


DEFAULT_TARGET_PARAMS: dict[str, Any] = {
    # Exclusión: perfiles que reciben monto=0 y cuotas=0
    'situacion_exclusion': 4,
    'dias_atraso_exclusion': 90,
    'excluir_judicial': True,
    'excluir_irrecuperable': True,
    # Monto sugerido
    'factor_monto_por_situacion': {
        1: 0.80,
        2: 0.50,
        3: 0.20,
    },
    'bonus_estabilidad_max': 0.20,
    'meses_historico_base': 24,
    # Formato JSON friendly para overrides
    'penalidad_atraso': [
        {'desde': 0, 'hasta': 0, 'penalidad': 0.00},
        {'desde': 1, 'hasta': 30, 'penalidad': 0.10},
        {'desde': 31, 'hasta': 90, 'penalidad': 0.25},
        {'desde': 91, 'hasta': 999, 'penalidad': 0.50},
    ],
    # Cuotas sugeridas
    'cuotas_valores': {
        0: 0,
        1: 3,
        2: 12,
        3: 24,
    },
    'cuotas_sit_mala_umbral': 6,
    'cuotas_racha_umbral': 6,
}


def get_default_target_params() -> dict[str, Any]:
    """Retorna una copia editable de los parámetros por defecto."""
    return copy.deepcopy(DEFAULT_TARGET_PARAMS)


def _validar_columnas(df: pd.DataFrame, esperadas: list[str], nombre_df: str) -> None:
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"{nombre_df} no tiene columnas esperadas: {', '.join(faltantes)}")


def _merge_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = get_default_target_params()
    if not overrides:
        return params

    desconocidas = set(overrides) - set(DEFAULT_TARGET_PARAMS)
    if desconocidas:
        raise ValueError(
            "Parámetros de target desconocidos: "
            + ', '.join(sorted(desconocidas))
        )

    for key, value in overrides.items():
        if key in {'factor_monto_por_situacion', 'cuotas_valores'}:
            merged = dict(params[key])
            merged.update(value)
            params[key] = merged
        else:
            params[key] = value

    return params


def _normalizar_dict_int(raw: dict[Any, Any], nombre: str) -> dict[int, Any]:
    normalizado: dict[int, Any] = {}
    for key, value in raw.items():
        try:
            key_int = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{nombre}: clave inválida {key!r}; debe ser int") from exc
        normalizado[key_int] = value
    return normalizado


def _normalizar_penalidad_atraso(raw: Any) -> list[tuple[int, int, float]]:
    penalidades: list[tuple[int, int, float]] = []

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                lo, hi = key
            elif isinstance(key, str) and '-' in key:
                lo, hi = key.split('-', maxsplit=1)
            else:
                raise ValueError(
                    "penalidad_atraso en dict requiere claves tipo (desde,hasta) o 'desde-hasta'"
                )
            penalidades.append((int(lo), int(hi), float(value)))

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("penalidad_atraso en lista requiere items tipo dict")
            if {'desde', 'hasta', 'penalidad'} - set(item):
                raise ValueError(
                    "Cada item de penalidad_atraso debe tener: desde, hasta, penalidad"
                )
            penalidades.append((
                int(item['desde']),
                int(item['hasta']),
                float(item['penalidad']),
            ))
    else:
        raise ValueError("penalidad_atraso debe ser list o dict")

    if not penalidades:
        raise ValueError('penalidad_atraso no puede quedar vacío')

    penalidades = sorted(penalidades, key=lambda x: x[0])
    for lo, hi, _ in penalidades:
        if lo > hi:
            raise ValueError(f'Rango inválido en penalidad_atraso: ({lo}, {hi})')

    return penalidades


def normalizar_target_params(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Mezcla defaults + overrides y normaliza tipos.

    Retorna una configuración interna lista para cálculo.
    """
    params = _merge_params(overrides)

    factor_por_sit = _normalizar_dict_int(params['factor_monto_por_situacion'], 'factor_monto_por_situacion')
    cuotas_valores = _normalizar_dict_int(params['cuotas_valores'], 'cuotas_valores')

    for sit in (1, 2, 3):
        if sit not in factor_por_sit:
            raise ValueError(f'factor_monto_por_situacion debe incluir la clave {sit}')
        factor_por_sit[sit] = float(factor_por_sit[sit])

    for cod in (0, 1, 2, 3):
        if cod not in cuotas_valores:
            raise ValueError(f'cuotas_valores debe incluir la clave {cod}')
        cuotas_valores[cod] = int(cuotas_valores[cod])

    config = {
        'situacion_exclusion': int(params['situacion_exclusion']),
        'dias_atraso_exclusion': int(params['dias_atraso_exclusion']),
        'excluir_judicial': bool(params['excluir_judicial']),
        'excluir_irrecuperable': bool(params['excluir_irrecuperable']),
        'factor_monto_por_situacion': factor_por_sit,
        'bonus_estabilidad_max': float(params['bonus_estabilidad_max']),
        'meses_historico_base': int(params['meses_historico_base']),
        'penalidad_atraso': _normalizar_penalidad_atraso(params['penalidad_atraso']),
        'cuotas_valores': cuotas_valores,
        'cuotas_sit_mala_umbral': float(params['cuotas_sit_mala_umbral']),
        'cuotas_racha_umbral': float(params['cuotas_racha_umbral']),
    }

    if config['meses_historico_base'] <= 0:
        raise ValueError('meses_historico_base debe ser > 0')

    return config


def _penalidad_por_atraso(dias: int, penalidad_atraso: list[tuple[int, int, float]]) -> float:
    """Devuelve la penalidad correspondiente a los días de atraso."""
    for lo, hi, penalidad in penalidad_atraso:
        if lo <= dias <= hi:
            return penalidad
    return penalidad_atraso[-1][2]


def _calcular_monto(row: pd.Series, tiene_historico: bool, cfg: dict[str, Any]) -> float:
    sit = int(row['situacion'])
    atraso = int(row['dias_atraso_max'])

    if sit >= cfg['situacion_exclusion']:
        return 0.0
    if cfg['excluir_judicial'] and int(row.get('proceso_judicial', 0) or 0) == 1:
        return 0.0
    if cfg['excluir_irrecuperable'] and int(row.get('irrecuperable', 0) or 0) == 1:
        return 0.0
    if atraso > cfg['dias_atraso_exclusion']:
        return 0.0

    prestamos = float(row['prestamos_total'])
    if tiene_historico and not pd.isna(row.get('monto_max_24m', np.nan)):
        base = max(prestamos, float(row['monto_max_24m']))
    else:
        base = prestamos

    if base == 0:
        return 0.0

    factor = cfg['factor_monto_por_situacion'].get(sit, 0.0)

    bonus = 0.0
    if tiene_historico and not pd.isna(row.get('meses_en_sit1', np.nan)):
        bonus = (
            float(row['meses_en_sit1'])
            / cfg['meses_historico_base']
            * cfg['bonus_estabilidad_max']
        )

    penalidad = _penalidad_por_atraso(atraso, cfg['penalidad_atraso'])
    monto = base * max(0.0, factor + bonus - penalidad)
    return round(monto, 1)


def _calcular_cuotas_cod(row: pd.Series, monto: float, cfg: dict[str, Any]) -> int:
    if monto == 0:
        return 0

    sit = int(row['situacion'])
    racha = float(row.get('racha_sit1_actual', 0) or 0)
    sit_mala = float(row.get('meses_sit_mala', 0) or 0)

    if sit >= cfg['situacion_exclusion']:
        return 0

    if sit == 3 or sit_mala >= cfg['cuotas_sit_mala_umbral']:
        return 1

    if sit == 1 and racha >= cfg['cuotas_racha_umbral']:
        return 3

    return 2


def generar_targets(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Agrega las columnas target al DataFrame de features.

    Parámetros:
        df: DataFrame con features ya construidas.
        params: overrides opcionales sobre DEFAULT_TARGET_PARAMS.
    """
    _validar_columnas(df, MIN_FEATURES_FOR_TARGETS, 'df (features para targets)')
    cfg = normalizar_target_params(params)

    print('Generando targets...')

    tiene_historico = (
        'meses_en_sit1' in df.columns
        and df['meses_en_sit1'].notna().any()
    )

    df = df.copy()

    df['monto_sugerido'] = df.apply(
        lambda row: _calcular_monto(row, tiene_historico, cfg),
        axis=1,
    )
    df['cuotas_cod'] = df.apply(
        lambda row: _calcular_cuotas_cod(row, row['monto_sugerido'], cfg),
        axis=1,
    )
    df['cuotas_valor'] = df['cuotas_cod'].map(cfg['cuotas_valores'])

    total = len(df)
    if total == 0:
        print('  Dataset vacío; no hay targets para reportar.')
        return df

    con_prestamo = int((df['monto_sugerido'] > 0).sum())
    print(f"  Con préstamo sugerido : {con_prestamo:,} ({con_prestamo/total*100:.1f}%)")
    print(f"  Sin préstamo (monto=0): {total-con_prestamo:,} ({(total-con_prestamo)/total*100:.1f}%)")
    print('  Distribución de cuotas:')
    for cod, valor in cfg['cuotas_valores'].items():
        n = int((df['cuotas_cod'] == cod).sum())
        print(f"    clase {cod} ({valor:2d} cuotas): {n:6,} ({n/total*100:.1f}%)")

    media = df.loc[df['monto_sugerido'] > 0, 'monto_sugerido'].mean()
    if pd.isna(media):
        print('  Monto promedio (con préstamo): 0.0 miles de $')
    else:
        print(f"  Monto promedio (con préstamo): {media:.1f} miles de $")

    return df
