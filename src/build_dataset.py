"""
build_dataset.py
----------------
Script principal del pipeline de datos. Ejecutar desde la raíz del repo:

    python src/build_dataset.py

Genera data_processed/dataset_final.csv listo para entrenar el modelo.

Estructura de carpetas esperada:
    data/
    ├── 202601DEUDORES/
    │   └── deudores.txt
    └── 202601_24DSF/          ← opcional, puede no estar todavía
        └── 24DSF.txt
    data_processed/
    └── dataset_final.csv      ← output
"""

import pandas as pd
import numpy as np
from pathlib import Path

from load_data import cargar_deudores, cargar_24dsf
from features import build_features_actuales, build_features_temporales, combinar_features
from targets import generar_targets


# ── Configuración de rutas ───────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / 'data'
OUTPUT_DIR    = ROOT / 'data_processed'

# Carpetas de los archivos BCRA (ajustar si cambia el nombre)
DEUDORES_DIR  = DATA_DIR / '202601DEUDORES'
DSF24_DIR     = DATA_DIR / '202601_24DSF'

DEUDORES_PATH = DEUDORES_DIR / 'deudores.txt'
DSF24_PATH    = DSF24_DIR    / '24DSF.txt'

OUTPUT_PATH   = OUTPUT_DIR / 'dataset_final.csv'


# ── Columnas finales del dataset de entrenamiento ────────────────────────────

FEATURES_NUMERICAS = [
    'situacion',
    'prestamos_total',
    'dias_atraso_max',
    'tiene_garantia_a',
    'ratio_cobertura',
    'refinanciado',
    'proceso_judicial',
    'recategorizado',
    'irrecuperable',
    'cant_entidades',
    # Features temporales (NaN si no hay 24DSF todavía)
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

FEATURES_CATEGORICAS = [
    'actividad',   # se encodea después en el notebook de entrenamiento
]

TARGETS = [
    'producto_recomendado',  # string (legible)
    'producto_cod',          # int (para el modelo)
    'monto_maximo',          # int en miles de pesos
]


def main():
    print("=" * 55)
    print("  Pipeline de construcción del dataset BCRA")
    print("=" * 55)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 1. Cargar deudores.txt ───────────────────────────────
    df_deudores = cargar_deudores(DEUDORES_PATH)

    # ── 2. Cargar 24DSF.txt (si está disponible) ─────────────
    df_temporales = None
    if DSF24_PATH.exists():
        df_24dsf = cargar_24dsf(DSF24_PATH)
        df_temporales = build_features_temporales(df_24dsf)
    else:
        print(f"\n⚠  24DSF.txt no encontrado en {DSF24_DIR}")
        print("   Continuando solo con features actuales.")
        print("   Las features temporales quedarán en NaN hasta que esté disponible.\n")

    # ── 3. Construir features actuales ───────────────────────
    df_actuales = build_features_actuales(df_deudores)

    # ── 4. Combinar ambas fuentes ─────────────────────────────
    df_features = combinar_features(df_actuales, df_temporales)

    # ── 5. Generar targets ────────────────────────────────────
    df_final = generar_targets(df_features)

    # ── 6. Seleccionar columnas finales ───────────────────────
    columnas_finales = (
        ['nro_id']
        + FEATURES_NUMERICAS
        + FEATURES_CATEGORICAS
        + TARGETS
    )
    # Incluir solo las que existen (las temporales pueden ser NaN si no hay 24DSF)
    columnas_finales = [c for c in columnas_finales if c in df_final.columns]
    df_export = df_final[columnas_finales]

    # ── 7. Exportar ───────────────────────────────────────────
    df_export.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓ Dataset guardado en: {OUTPUT_PATH}")
    print(f"  Filas:    {len(df_export):,}")
    print(f"  Columnas: {len(df_export.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df_export.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
