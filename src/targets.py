"""
targets.py
----------
Genera las variables target a partir de las features ya construidas.

El modelo predice DOS cosas por cliente:
    - monto_sugerido  → regresión  (en miles de pesos)
    - cuotas_cod      → clasificación discreta (0, 1, 2, 3)

Estas fórmulas se aplican UNA SOLA VEZ para construir el dataset de
entrenamiento. El modelo Keras aprende a aproximarlas y luego las
generaliza sobre perfiles nuevos.

Todos los parámetros están agrupados en secciones claramente separadas
para facilitar su ajuste sin tocar la lógica del código.
"""

import pandas as pd
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS — editá estos valores para ajustar el comportamiento
# ════════════════════════════════════════════════════════════════════════════

# ── Exclusión: perfiles que reciben monto=0 y cuotas=0 ──────────────────────

SITUACION_EXCLUSION     = 4      # situación >= este valor → excluido (4 y 5)
DIAS_ATRASO_EXCLUSION   = 90     # más de N días de atraso → excluido
EXCLUIR_JUDICIAL        = True   # True = proceso judicial activo → excluido
EXCLUIR_IRRECUPERABLE   = True   # True = marcado irrecuperable   → excluido

# ── Monto sugerido ───────────────────────────────────────────────────────────
# Base de cálculo: max(prestamos_total, monto_max_24m)
# Representa el pico histórico de endeudamiento que el cliente ya sostuvo.

# Factor por situación crediticia actual
# Cuánto del pico histórico se habilita según el riesgo
FACTOR_MONTO_POR_SITUACION = {
    1: 0.80,   # situación normal     → 80% del pico histórico
    2: 0.50,   # seguimiento especial → 50%
    3: 0.20,   # con problemas        → 20%
}

# Bonus por estabilidad histórica
# Se suma al factor anterior. Máximo cuando meses_en_sit1 == 24.
BONUS_ESTABILIDAD_MAX   = 0.20   # +20% si estuvo los 24 meses impecable
MESES_HISTORICO_BASE    = 24     # denominador para calcular el bonus

# Penalidad por días de atraso actuales
# Se resta al factor. Cada tupla es (dia_desde, dia_hasta): penalidad
PENALIDAD_ATRASO = {
    (0,   0):  0.00,   # sin atraso
    (1,  30):  0.10,   # atraso leve     → -10%
    (31, 90):  0.25,   # atraso moderado → -25%
    (91, 999): 0.50,   # atraso grave    → -50%
}

# ── Cuotas sugeridas (clasificación discreta) ────────────────────────────────
# El modelo predice una de estas 4 clases.
# La fintech decide después qué hacer con la predicción.

CUOTAS_VALORES = {
    0:  0,    # no aplica préstamo
    1:  3,    # perfil marginal  →  3 cuotas
    2: 12,    # perfil aceptable → 12 cuotas
    3: 24,    # perfil sólido    → 24 cuotas
}

# Umbrales para asignar la clase de cuotas
CUOTAS_SIT_MALA_UMBRAL  = 6    # meses_sit_mala >= N → máximo 3 cuotas
CUOTAS_RACHA_UMBRAL     = 6    # racha_sit1 >= N con sit==1 → 24 cuotas


# ════════════════════════════════════════════════════════════════════════════
#  LÓGICA DE CÁLCULO — no es necesario editar debajo de esta línea
# ════════════════════════════════════════════════════════════════════════════

def _penalidad_por_atraso(dias: int) -> float:
    """Devuelve la penalidad correspondiente a los días de atraso."""
    for (lo, hi), penalidad in PENALIDAD_ATRASO.items():
        if lo <= dias <= hi:
            return penalidad
    return 0.50  # fallback para valores fuera de rango


def _calcular_monto(row: pd.Series, tiene_historico: bool) -> float:
    """
    Calcula el monto sugerido en miles de pesos.

    Fórmula:
        base             = max(prestamos_total, monto_max_24m)
        factor_situacion = FACTOR_MONTO_POR_SITUACION[situacion]
        bonus            = (meses_en_sit1 / 24) * BONUS_ESTABILIDAD_MAX
        penalidad        = según rango de dias_atraso_max
        monto            = base * max(0, factor + bonus - penalidad)
    """
    sit    = int(row['situacion'])
    atraso = int(row['dias_atraso_max'])

    # Exclusión directa
    if sit >= SITUACION_EXCLUSION:
        return 0.0
    if EXCLUIR_JUDICIAL and row['proceso_judicial'] == 1:
        return 0.0
    if EXCLUIR_IRRECUPERABLE and row['irrecuperable'] == 1:
        return 0.0
    if atraso > DIAS_ATRASO_EXCLUSION:
        return 0.0

    # Base: pico histórico de endeudamiento
    prestamos = float(row['prestamos_total'])
    if tiene_historico and not pd.isna(row.get('monto_max_24m', np.nan)):
        base = max(prestamos, float(row['monto_max_24m']))
    else:
        base = prestamos

    if base == 0:
        return 0.0

    # Factor por situación
    factor = FACTOR_MONTO_POR_SITUACION.get(sit, 0.0)

    # Bonus por estabilidad (solo si hay historial)
    bonus = 0.0
    if tiene_historico and not pd.isna(row.get('meses_en_sit1', np.nan)):
        bonus = (float(row['meses_en_sit1']) / MESES_HISTORICO_BASE) * BONUS_ESTABILIDAD_MAX

    # Penalidad por atraso
    penalidad = _penalidad_por_atraso(atraso)

    monto = base * max(0.0, factor + bonus - penalidad)
    return round(monto, 1)


def _calcular_cuotas_cod(row: pd.Series, monto: float, tiene_historico: bool) -> int:
    """
    Asigna la clase de cuotas (0, 1, 2, 3).

    Reglas en orden de prioridad:
        monto == 0                           → clase 0 (no aplica)
        sit >= SITUACION_EXCLUSION           → clase 0
        sit == 3 o meses_sit_mala >= umbral  → clase 1 ( 3 cuotas)
        sit == 1 y racha >= umbral           → clase 3 (24 cuotas)
        resto                                → clase 2 (12 cuotas)
    """
    if monto == 0:
        return 0

    sit      = int(row['situacion'])
    racha    = float(row.get('racha_sit1_actual', 0) or 0)
    sit_mala = float(row.get('meses_sit_mala', 0) or 0)

    if sit >= SITUACION_EXCLUSION:
        return 0

    if sit == 3 or sit_mala >= CUOTAS_SIT_MALA_UMBRAL:
        return 1   # 3 cuotas

    if sit == 1 and racha >= CUOTAS_RACHA_UMBRAL:
        return 3   # 24 cuotas

    return 2       # 12 cuotas (sit 2, o sit 1 con racha corta)


# ── Función principal ────────────────────────────────────────────────────────

def generar_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las columnas target al DataFrame de features.

    Columnas agregadas:
        monto_sugerido  → float, miles de pesos   (target de regresión)
        cuotas_cod      → int 0-3, clase discreta  (target de clasificación)
        cuotas_valor    → int, cuotas reales       (legible, no entra al modelo)
    """
    print("Generando targets...")

    tiene_historico = (
        'meses_en_sit1' in df.columns
        and df['meses_en_sit1'].notna().any()
    )

    df = df.copy()

    df['monto_sugerido'] = df.apply(
        lambda row: _calcular_monto(row, tiene_historico), axis=1
    )
    df['cuotas_cod'] = df.apply(
        lambda row: _calcular_cuotas_cod(row, row['monto_sugerido'], tiene_historico),
        axis=1,
    )
    df['cuotas_valor'] = df['cuotas_cod'].map(CUOTAS_VALORES)

    # Reporte
    total        = len(df)
    con_prestamo = (df['monto_sugerido'] > 0).sum()
    print(f"  Con préstamo sugerido : {con_prestamo:,} ({con_prestamo/total*100:.1f}%)")
    print(f"  Sin préstamo (monto=0): {total-con_prestamo:,} ({(total-con_prestamo)/total*100:.1f}%)")
    print("  Distribución de cuotas:")
    for cod, valor in CUOTAS_VALORES.items():
        n = (df['cuotas_cod'] == cod).sum()
        print(f"    clase {cod} ({valor:2d} cuotas): {n:6,} ({n/total*100:.1f}%)")
    media = df.loc[df['monto_sugerido'] > 0, 'monto_sugerido'].mean()
    print(f"  Monto promedio (con préstamo): {media:.1f} miles de $")

    return df