"""
features.py
-----------
Toma los DataFrames crudos de load_data.py y construye el dataset
de features agregadas — una fila por CUIT.

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
    Agrupa por CUIT tomando el peor/mayor valor entre todas las entidades
    que reportan a esa persona.

    Retorna un DataFrame con una fila por CUIT.
    """
    print("Construyendo features actuales...")

    grp = df_deudores.groupby('nro_id')

    features = pd.DataFrame(index=grp.groups.keys())
    features.index.name = 'nro_id'

    # Situación: peor entre todas las entidades
    features['situacion'] = grp['situacion'].max()

    # Deuda total: suma de todos los bancos
    features['prestamos_total'] = grp['prestamos'].sum()

    # Peor atraso registrado
    features['dias_atraso_max'] = grp['dias_atraso'].max()

    # Garantías
    features['garantias_pref_a']  = grp['garantias_pref_a'].sum()
    features['garantias_pref_b']  = grp['garantias_pref_b'].sum()
    features['tiene_garantia_a']  = (features['garantias_pref_a'] > 0).astype(int)

    # Ratio de cobertura: qué % de la deuda está cubierta con garantía A
    features['ratio_cobertura'] = np.where(
        features['prestamos_total'] > 0,
        features['garantias_pref_a'] / features['prestamos_total'],
        0.0
    )

    # Señales de alerta: 1 si en AL MENOS UNA entidad está marcado
    features['refinanciado']      = grp['refinanciaciones'].apply(lambda x: int((x == 1).any()))
    features['proceso_judicial']  = grp['proceso_judicial'].apply(lambda x: int((x == 1).any()))
    features['recategorizado']    = grp['recategorizacion_obligatoria'].apply(lambda x: int((x == 1).any()))
    features['irrecuperable']     = grp['irrecuperable'].apply(lambda x: int((x == 1).any()))

    # Cuántas entidades distintas reportan a este CUIT
    features['cant_entidades'] = grp['cod_entidad'].nunique()

    # Actividad: tomar la primera no-desconocida; si todas son desconocidas → 'desconocido'
    def primera_actividad(serie):
        conocidas = serie[serie != 'desconocido']
        return conocidas.iloc[0] if len(conocidas) > 0 else 'desconocido'

    features['actividad'] = grp['actividad'].apply(primera_actividad)

    features = features.reset_index()
    print(f"  → {len(features):,} CUITs con features actuales")
    return features


# ── Features temporales (24DSF.txt) ─────────────────────────────────────────

def build_features_temporales(df_24dsf: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de evolución crediticia por CUIT a partir del historial
    de 24 meses. mes_relativo=1 es el más reciente.

    Retorna un DataFrame con una fila por CUIT.
    """
    print("Construyendo features temporales...")

    def calcular_features_cuit(grupo):
        # Ordenar del más antiguo al más reciente (mes 24 → mes 1)
        g = grupo.sort_values('mes_relativo', ascending=False)
        situaciones = g['situacion'].values
        montos      = g['monto'].values
        n           = len(situaciones)

        # Estabilidad positiva
        meses_en_sit1    = int((situaciones == 1).sum())
        meses_sit_mala   = int((situaciones >= 3).sum())
        peor_situacion   = int(situaciones.max())

        # Tendencia: promedio últimos 3 meses vs meses 4-12
        ultimos_3   = situaciones[:3].mean() if n >= 3 else situaciones.mean()
        anteriores  = situaciones[3:12].mean() if n > 3 else situaciones.mean()
        # Negativo = mejorando (bajó la situación), positivo = empeorando
        tendencia   = float(round(ultimos_3 - anteriores, 3))

        # Racha actual en situación 1 (desde el mes más reciente)
        racha = 0
        for s in situaciones:  # ya ordenado de reciente a antiguo
            if s == 1:
                racha += 1
            else:
                break

        # Evolución de monto: mes actual vs hace 12 meses
        monto_actual   = montos[0] if n >= 1 else 0
        monto_hace_12  = montos[11] if n >= 12 else montos[-1]
        if monto_hace_12 > 0:
            variacion_12m = float(round((monto_actual - monto_hace_12) / monto_hace_12, 3))
        else:
            variacion_12m = 0.0

        # Estadísticas de monto
        monto_promedio = float(round(montos.mean(), 1))
        monto_max      = float(montos.max())
        meses_con_deuda = int((montos > 0).sum())

        return pd.Series({
            'meses_en_sit1':       meses_en_sit1,
            'meses_sit_mala':      meses_sit_mala,
            'peor_situacion_24m':  peor_situacion,
            'tendencia_situacion': tendencia,
            'racha_sit1_actual':   racha,
            'variacion_monto_12m': variacion_12m,
            'monto_promedio_24m':  monto_promedio,
            'monto_max_24m':       monto_max,
            'meses_con_deuda':     meses_con_deuda,
        })

    features_temp = (
        df_24dsf
        .groupby('nro_id')
        .apply(calcular_features_cuit)
        .reset_index()
    )

    print(f"  → {len(features_temp):,} CUITs con features temporales")
    return features_temp


# ── Join de ambas fuentes ────────────────────────────────────────────────────

def combinar_features(
    df_actuales: pd.DataFrame,
    df_temporales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une las features actuales con las temporales por CUIT.
    Los CUITs que no aparecen en 24DSF quedan con NaN en las features temporales
    (pueden existir si el 24DSF es de un período anterior).
    """
    print("Combinando features actuales y temporales...")

    df = df_actuales.merge(df_temporales, on='nro_id', how='left')

    # Si no hay 24DSF todavía, rellenar features temporales con valores neutros
    cols_temporales = [
        'meses_en_sit1', 'meses_sit_mala', 'peor_situacion_24m',
        'tendencia_situacion', 'racha_sit1_actual', 'variacion_monto_12m',
        'monto_promedio_24m', 'monto_max_24m', 'meses_con_deuda',
    ]
    for col in cols_temporales:
        if col not in df.columns:
            df[col] = np.nan

    sin_historico = df[cols_temporales[0]].isna().sum()
    if sin_historico > 0:
        print(f"  ⚠ {sin_historico:,} CUITs sin historial en 24DSF (features temporales en NaN)")

    print(f"  → Dataset combinado: {len(df):,} filas x {len(df.columns)} columnas")
    return df
