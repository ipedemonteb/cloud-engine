"""
targets.py
----------
Genera el target para el modelo de scoring crediticio.

Target: score_crediticio (float, 0.0 a 1.0)

    El score se deriva del promedio real de situacion BCRA de los meses 1 a 6
    del 24DSF, normalizado a un rango [0, 1] donde:

        score = 1 - (situacion_promedio - 1) / 4

    Equivalencias:
        situacion promedio 1.0 -> score 1.00  (perfil excelente)
        situacion promedio 2.0 -> score 0.75  (perfil aceptable)
        situacion promedio 3.0 -> score 0.50  (perfil riesgoso)
        situacion promedio 4.0 -> score 0.25  (alto riesgo)
        situacion promedio 5.0 -> score 0.00  (irrecuperable)

Este target es real y proviene del 24DSF del BCRA.
Los meses 1-6 son el futuro que el modelo aprende a predecir.
Los meses 7-24 son el historial que usa como input (features).

El modelo tiene una neurona de salida con activacion sigmoid.
La fintech aplica sus propios umbrales sobre el score.
"""

import pandas as pd
import numpy as np


MESES_TARGET = 6


def calcular_score(sits: list) -> float | None:
    """
    Convierte una lista de situaciones BCRA (1-5) en un score normalizado (0-1).

    Parametros
    ----------
    sits : list de int o None
        Situaciones de los meses 1 a 6 (mas recientes). Puede contener None
        si el CUIT no fue reportado en algun mes.

    Retorna
    -------
    float entre 0.0 y 1.0, o None si no hay suficientes datos validos.
    """
    validas = [s for s in sits if s is not None and 1 <= s <= 5]
    if not validas:
        return None

    promedio = np.mean(validas)
    score = 1.0 - (promedio - 1.0) / 4.0
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def generar_targets(df: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Une el DataFrame de features con los scores reales derivados del 24DSF.

    Parametros
    ----------
    df        : DataFrame de features (una fila por CUIT)
    df_target : DataFrame con columnas ['nro_id', 'score_crediticio']
                generado por cargar_24dsf()

    Retorna
    -------
    DataFrame combinado con 'score_crediticio' como target.
    El join es inner, los CUITs sin score se descartan.
    """
    print('Generando targets...')

    df_out = df.merge(df_target, on='nro_id', how='inner')

    descartados = len(df) - len(df_out)
    if descartados > 0:
        print(f"  WARNING: {descartados:,} CUITs descartados por no tener score en 24DSF")

    total = len(df_out)
    score = df_out['score_crediticio']

    print('  Score crediticio:')
    print(f"    Media   : {score.mean():.3f}")
    print(f"    Mediana : {score.median():.3f}")
    print(f"    Min     : {score.min():.3f}")
    print(f"    Max     : {score.max():.3f}")
    print(f"    Score >= 0.75 (excelente): {(score >= 0.75).sum():,} ({(score >= 0.75).mean()*100:.1f}%)")
    print(f"    Score >= 0.50 (aceptable): {(score >= 0.50).sum():,} ({(score >= 0.50).mean()*100:.1f}%)")
    print(f"    Score <  0.25 (critico)  : {(score <  0.25).sum():,} ({(score <  0.25).mean()*100:.1f}%)")
    print(f"  -> Dataset final: {total:,} filas")

    return df_out
