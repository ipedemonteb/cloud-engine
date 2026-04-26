"""
features.py
-----------
Normaliza y valida las features generadas por load_data.py,
asegurando una fila por CUIT y un contrato estable para el pipeline.

Features del estado actual (de deudores.txt):
    situacion, prestamos_total, dias_atraso_max, tiene_garantia_a,
    ratio_cobertura, refinanciado, proceso_judicial, recategorizado,
    cant_entidades, actividad

Features temporales (de 24DSF.txt):
    meses_en_sit1, meses_sit_mala, peor_situacion_24m,
    tendencia_situacion, racha_sit1_actual, variacion_monto_12m,
    monto_promedio_24m, monto_max_24m, meses_con_deuda
"""

import pandas as pd
import numpy as np


# ── Features del estado actual (deudores.txt) ────────────────────────────────

def build_features_actuales(df_deudores: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y normaliza las features actuales (deudores.txt) ya agregadas.
    El DataFrame esperado viene de cargar_deudores(), con una fila por CUIT.

    Retorna un DataFrame con una fila por CUIT.
    """
    print("Construyendo features actuales...")

    columnas_esperadas = [
        'nro_id', 'situacion', 'prestamos_total', 'garantias_pref_a',
        'garantias_pref_b', 'dias_atraso_max', 'proceso_judicial',
        'refinanciado', 'recategorizado', 'irrecuperable',
        'cant_entidades', 'actividad',
    ]
    faltantes = [c for c in columnas_esperadas if c not in df_deudores.columns]
    if faltantes:
        raise ValueError(
            "df_deudores no tiene las columnas esperadas para features actuales: "
            + ", ".join(faltantes)
        )

    features = df_deudores[columnas_esperadas].copy()

    if features['nro_id'].duplicated().any():
        print("  ⚠ Se detectaron CUITs duplicados en features actuales; consolidando...")

        def primera_actividad(serie: pd.Series) -> str:
            conocidas = serie[serie != 'desconocido']
            return str(conocidas.iloc[0]) if len(conocidas) > 0 else 'desconocido'

        features = (
            features
            .groupby('nro_id', as_index=False)
            .agg({
                'situacion': 'max',
                'prestamos_total': 'sum',
                'garantias_pref_a': 'sum',
                'garantias_pref_b': 'sum',
                'dias_atraso_max': 'max',
                'proceso_judicial': 'max',
                'refinanciado': 'max',
                'recategorizado': 'max',
                'irrecuperable': 'max',
                'cant_entidades': 'max',
                'actividad': primera_actividad,
            })
        )

    features['tiene_garantia_a'] = (features['garantias_pref_a'] > 0).astype(int)
    features['ratio_cobertura'] = np.where(
        features['prestamos_total'] > 0,
        features['garantias_pref_a'] / features['prestamos_total'],
        0.0,
    )

    features = features[
        [
            'nro_id', 'situacion', 'prestamos_total', 'dias_atraso_max',
            'garantias_pref_a', 'garantias_pref_b', 'tiene_garantia_a',
            'ratio_cobertura', 'refinanciado', 'proceso_judicial',
            'recategorizado', 'irrecuperable', 'cant_entidades', 'actividad',
        ]
    ]

    print(f"  → {len(features):,} CUITs con features actuales")
    return features


# ── Features temporales (24DSF.txt) ─────────────────────────────────────────

def build_features_temporales(df_24dsf: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y normaliza las features temporales (24DSF) ya agregadas.
    El DataFrame esperado viene de cargar_24dsf(), con una fila por CUIT,
    y features calculadas solo con meses 7-24 (mes 7 es el mas reciente
    disponible como feature).

    Retorna un DataFrame con una fila por CUIT.
    """
    print("Construyendo features temporales...")

    columnas_esperadas = [
        'nro_id',
        'meses_en_sit1',
        'meses_sit_mala',
        'peor_situacion_24m',
        'tendencia_situacion',
        'racha_sit1_actual',
        'variacion_monto_12m',
        'monto_promedio_24m',
        'monto_max_24m',
        'meses_con_deuda',
    ]
    faltantes = [c for c in columnas_esperadas if c not in df_24dsf.columns]
    if faltantes:
        raise ValueError(
            "df_24dsf no tiene las columnas esperadas para features temporales: "
            + ", ".join(faltantes)
        )

    features_temp = df_24dsf[columnas_esperadas].copy()

    if features_temp['nro_id'].duplicated().any():
        print("  ⚠ Se detectaron CUITs duplicados en features temporales; consolidando...")
        features_temp = (
            features_temp
            .groupby('nro_id', as_index=False)
            .agg({
                'meses_en_sit1': 'min',
                'meses_sit_mala': 'max',
                'peor_situacion_24m': 'max',
                'tendencia_situacion': 'max',
                'racha_sit1_actual': 'min',
                'variacion_monto_12m': 'mean',
                'monto_promedio_24m': 'mean',
                'monto_max_24m': 'max',
                'meses_con_deuda': 'max',
            })
        )

    print(f"  → {len(features_temp):,} CUITs con features temporales")
    return features_temp


# ── Join de ambas fuentes ────────────────────────────────────────────────────

def combinar_features(
    df_actuales: pd.DataFrame,
    df_temporales: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Une las features actuales con las temporales por CUIT.
    Los CUITs que no aparecen en 24DSF quedan con NaN en las features temporales
    (pueden existir si el 24DSF es de un período anterior).
    """
    print("Combinando features actuales y temporales...")

    # Si no hay 24DSF todavía, dejamos las temporales en NaN
    cols_temporales = [
        'meses_en_sit1', 'meses_sit_mala', 'peor_situacion_24m',
        'tendencia_situacion', 'racha_sit1_actual', 'variacion_monto_12m',
        'monto_promedio_24m', 'monto_max_24m', 'meses_con_deuda',
    ]

    if df_temporales is None:
        df = df_actuales.copy()
        for col in cols_temporales:
            if col not in df.columns:
                df[col] = np.nan
    else:
        faltantes = [c for c in ['nro_id', *cols_temporales] if c not in df_temporales.columns]
        if faltantes:
            raise ValueError(
                "df_temporales no tiene las columnas esperadas: "
                + ", ".join(faltantes)
            )

        df_temporales = df_temporales[['nro_id', *cols_temporales]].drop_duplicates(subset='nro_id', keep='first')
        df = df_actuales.merge(df_temporales, on='nro_id', how='left')

    sin_historico = df[cols_temporales[0]].isna().sum()
    if sin_historico > 0:
        print(f"  ⚠ {sin_historico:,} CUITs sin historial en 24DSF (features temporales en NaN)")

    print(f"  → Dataset combinado: {len(df):,} filas x {len(df.columns)} columnas")
    return df
