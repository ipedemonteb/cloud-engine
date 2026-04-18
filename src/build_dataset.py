"""
build_dataset.py
----------------
Script principal del pipeline de datos. Ejecutar desde la raíz del repo:

    python src/build_dataset.py

Genera data_processed/dataset_final.csv listo para entrenar el modelo.

Estructura de carpetas esperada:
    data/
    ├── 202602DEUDORES/
    │   └── deudores.txt
    └── 24DSF202602/           ← opcional, puede no estar todavía
        └── 24DSF.txt
    data_processed/
    └── dataset_final.csv      ← output
"""

import pandas as pd
from pathlib import Path

from load_data import cargar_deudores, cargar_24dsf
from features import combinar_features
from targets import generar_targets


# ── Configuración de rutas ───────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / 'data'
OUTPUT_DIR    = ROOT / 'data_processed'

# Carpetas de los archivos BCRA (ajustar si cambia el nombre)
DEUDORES_DIR  = DATA_DIR / '202602DEUDORES'
DSF24_DIR     = DATA_DIR / '24DSF202602'

OUTPUT_PATH   = OUTPUT_DIR / 'dataset_final.csv'

# Si True, conserva solo personas humanas (prefijos 20/23/24/27)
PERSONA_HUMANA_ONLY = True


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
    'monto_sugerido',
    'cuotas_cod',
    'cuotas_valor',
]


TEMPORALES_ESPERADAS = [
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


def _resolver_path_preferido(path_principal: Path, path_fallback: Path) -> Path | None:
    """Usa path principal; si no existe, usa fallback; si no hay ninguno, None."""
    if path_principal.exists():
        return path_principal
    if path_fallback.exists():
        return path_fallback
    return None


def _validar_columnas(df: pd.DataFrame, esperadas: list[str], nombre_df: str) -> None:
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"{nombre_df} no tiene columnas esperadas: {', '.join(faltantes)}")


def _validar_rangos_temporales(df: pd.DataFrame) -> None:
    """Valida rangos básicos para detectar errores de parseo/agregación."""
    checks = {
        'meses_en_sit1': (0, 24),
        'meses_sit_mala': (0, 24),
        'racha_sit1_actual': (0, 24),
        'meses_con_deuda': (0, 24),
        'peor_situacion_24m': (1, 5),
    }

    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue

        serie = df[col].dropna()
        if serie.empty:
            continue

        invalidos = ((serie < lo) | (serie > hi)).sum()
        if invalidos:
            raise ValueError(
                f"Rango inválido en {col}: {invalidos:,} filas fuera de [{lo}, {hi}]"
            )


def main():
    print("=" * 55)
    print("  Pipeline de construcción del dataset BCRA")
    print("=" * 55)

    OUTPUT_DIR.mkdir(exist_ok=True)

    deudores_path = _resolver_path_preferido(
        DEUDORES_DIR / 'deudores.txt',
        DEUDORES_DIR / 'deudores_test.txt',
    )
    dsf24_path = _resolver_path_preferido(
        DSF24_DIR / '24DSF.txt',
        DSF24_DIR / '24DSF_test.txt',
    )

    if deudores_path is None:
        raise FileNotFoundError(
            f"No se encontró deudores.txt ni deudores_test.txt en {DEUDORES_DIR}"
        )

    print(f"Archivo deudores: {deudores_path.name}")
    if dsf24_path is not None:
        print(f"Archivo 24DSF:    {dsf24_path.name}")
    else:
        print("Archivo 24DSF:    no disponible (opcional)")
    print(
        "Filtro personas humanas: "
        + ("activo (20/23/24/27)" if PERSONA_HUMANA_ONLY else "desactivado")
    )

    # ── 1. Cargar deudores.txt ───────────────────────────────
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

    # ── 2. Cargar 24DSF.txt (si está disponible) ─────────────
    df_temporales = None
    if dsf24_path is not None:
        df_temporales = cargar_24dsf(
            dsf24_path,
            persona_humana_only=PERSONA_HUMANA_ONLY,
        )
        _validar_columnas(df_temporales, ['nro_id', *TEMPORALES_ESPERADAS], 'df_temporales')
        if df_temporales['nro_id'].duplicated().any():
            raise ValueError('df_temporales tiene CUITs duplicados; se esperaba una fila por nro_id')
        _validar_rangos_temporales(df_temporales)
    else:
        print(f"\n⚠  24DSF.txt no encontrado en {DSF24_DIR}")
        print("   Continuando solo con features actuales.")
        print("   Las features temporales quedarán en NaN hasta que esté disponible.\n")

    # ── 3. Combinar ambas fuentes ─────────────────────────────
    df_features = combinar_features(df_actuales, df_temporales)

    # ── 4. Generar targets ────────────────────────────────────
    df_final = generar_targets(df_features)
    _validar_columnas(df_final, TARGETS, 'df_final')

    # ── 5. Seleccionar columnas finales ───────────────────────
    columnas_finales = (
        ['nro_id']
        + FEATURES_NUMERICAS
        + FEATURES_CATEGORICAS
        + TARGETS
    )

    # Si no hay 24DSF, combinar_features ya crea temporales en NaN,
    # así que todas las columnas esperadas deberían existir.
    _validar_columnas(df_final, columnas_finales, 'df_final (columnas exportables)')
    df_export = df_final[columnas_finales]

    # ── 6. Exportar ───────────────────────────────────────────
    df_export.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓ Dataset guardado en: {OUTPUT_PATH}")
    print(f"  Filas:    {len(df_export):,}")
    print(f"  Columnas: {len(df_export.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df_export.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
