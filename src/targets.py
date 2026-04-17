"""
targets.py
----------
Genera las variables target a partir de las features ya construidas.

Target: producto_recomendado
    0 → sin_producto      (perfiles de alto riesgo)
    1 → tarjeta_credito   (perfiles medios)
    2 → prestamo_personal (perfiles buenos)

La lógica de asignación es configurable — está centralizada en REGLAS
para que sea fácil de ajustar sin tocar el resto del código.
"""

import pandas as pd
import numpy as np


# ── Reglas de negocio ────────────────────────────────────────────────────────
# Estas constantes definen los umbrales de cada producto.
# Cambiarlas acá afecta todo el pipeline automáticamente.

REGLAS = {
    # Condiciones de exclusión total (sin producto)
    'exclusion': {
        'situacion_min':       4,    # situación 4 o 5 → excluido
        'proceso_judicial':    True, # cualquier proceso judicial → excluido
        'irrecuperable':       True, # marcado como irrecuperable → excluido
        'dias_atraso_max':     90,   # más de 90 días de atraso → excluido
    },

    # Préstamo personal: perfil más sólido
    'prestamo_personal': {
        'situacion_max':       1,    # solo situación 1
        'dias_atraso_max':     0,    # sin atrasos
        'meses_en_sit1_min':   18,   # al menos 18 de los últimos 24 meses en normal
        'racha_sit1_min':      6,    # al menos 6 meses consecutivos en normal
        'meses_sit_mala_max':  0,    # sin historial de situación mala
    },

    # Tarjeta de crédito: perfil intermedio
    'tarjeta_credito': {
        'situacion_max':       2,    # situación 1 o 2
        'dias_atraso_max':     30,   # hasta 30 días de atraso
        'meses_en_sit1_min':   12,   # al menos 12 de los últimos 24 meses en normal
        'meses_sit_mala_max':  3,    # máximo 3 meses con situación mala
    },
}

# Montos máximos sugeridos (en miles de pesos)
MONTOS = {
    'sin_producto':      0,
    'tarjeta_credito':   150,   # límite de tarjeta
    'prestamo_personal': 500,   # monto de préstamo
}


# ── Función de asignación ────────────────────────────────────────────────────

def _asignar_producto(row: pd.Series, tiene_historico: bool) -> str:
    """Aplica las reglas de negocio a una fila y retorna el producto."""

    exc = REGLAS['exclusion']
    pre = REGLAS['prestamo_personal']
    tar = REGLAS['tarjeta_credito']

    # 1. Exclusión total
    if row['situacion'] >= exc['situacion_min']:
        return 'sin_producto'
    if exc['proceso_judicial'] and row['proceso_judicial'] == 1:
        return 'sin_producto'
    if exc['irrecuperable'] and row['irrecuperable'] == 1:
        return 'sin_producto'
    if row['dias_atraso_max'] > exc['dias_atraso_max']:
        return 'sin_producto'

    # Si no hay historial temporal, usamos solo las features actuales
    # con criterios más conservadores
    if not tiene_historico or pd.isna(row.get('meses_en_sit1', np.nan)):
        if row['situacion'] == 1 and row['dias_atraso_max'] == 0 and row['refinanciado'] == 0:
            return 'tarjeta_credito'   # sin historial → no damos préstamo
        elif row['situacion'] <= 2 and row['dias_atraso_max'] <= 30:
            return 'tarjeta_credito'
        else:
            return 'sin_producto'

    # 2. Préstamo personal (criterios más estrictos)
    if (
        row['situacion'] <= pre['situacion_max']
        and row['dias_atraso_max'] <= pre['dias_atraso_max']
        and row['meses_en_sit1'] >= pre['meses_en_sit1_min']
        and row['racha_sit1_actual'] >= pre['racha_sit1_min']
        and row['meses_sit_mala'] <= pre['meses_sit_mala_max']
        and row['refinanciado'] == 0
        and row['proceso_judicial'] == 0
    ):
        return 'prestamo_personal'

    # 3. Tarjeta de crédito
    if (
        row['situacion'] <= tar['situacion_max']
        and row['dias_atraso_max'] <= tar['dias_atraso_max']
        and row['meses_en_sit1'] >= tar['meses_en_sit1_min']
        and row['meses_sit_mala'] <= tar['meses_sit_mala_max']
    ):
        return 'tarjeta_credito'

    # 4. Si no califica para ninguno
    return 'sin_producto'


def generar_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las columnas target al DataFrame de features.

    Columnas agregadas:
        producto_recomendado  → string ('sin_producto', 'tarjeta_credito', 'prestamo_personal')
        producto_cod          → int (0, 1, 2) para usar en el modelo
        monto_maximo          → int en miles de pesos
    """
    print("Generando targets...")

    tiene_historico = 'meses_en_sit1' in df.columns and df['meses_en_sit1'].notna().any()

    df = df.copy()
    df['producto_recomendado'] = df.apply(
        lambda row: _asignar_producto(row, tiene_historico), axis=1
    )

    # Encoding numérico
    encoding = {'sin_producto': 0, 'tarjeta_credito': 1, 'prestamo_personal': 2}
    df['producto_cod'] = df['producto_recomendado'].map(encoding)

    # Monto máximo sugerido
    df['monto_maximo'] = df['producto_recomendado'].map(MONTOS)

    # Reporte de distribución
    dist = df['producto_recomendado'].value_counts()
    total = len(df)
    print("  Distribución de productos:")
    for producto, count in dist.items():
        print(f"    {producto:25s}: {count:6,} ({count/total*100:.1f}%)")

    return df
