"""
load_data.py
------------
Parsea los archivos de ancho fijo del BCRA y los devuelve como DataFrames limpios.
No hace ninguna transformación de features — eso es responsabilidad de features.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Definición de columnas según el LEAME del BCRA ──────────────────────────

DEUDORES_COLSPECS = [
    (0,   5),   # cod_entidad
    (5,   11),  # fecha_info
    (11,  13),  # tipo_id
    (13,  24),  # nro_id
    (24,  27),  # actividad
    (27,  29),  # situacion
    (29,  41),  # prestamos
    (41,  53),  # sin_uso
    (53,  65),  # garantias_otorgadas
    (65,  77),  # otros_conceptos
    (77,  89),  # garantias_pref_a
    (89,  101), # garantias_pref_b
    (101, 113), # sin_garantias
    (113, 125), # contragarantias_pref_a
    (125, 137), # contragarantias_pref_b
    (137, 149), # sin_contragarantias
    (149, 161), # previsiones
    (161, 162), # deuda_cubierta
    (162, 163), # proceso_judicial
    (163, 164), # refinanciaciones
    (164, 165), # recategorizacion_obligatoria
    (165, 166), # situacion_juridica
    (166, 167), # irrecuperable
    (167, 171), # dias_atraso
]

DEUDORES_NOMBRES = [
    'cod_entidad', 'fecha_info', 'tipo_id', 'nro_id', 'actividad',
    'situacion', 'prestamos', 'sin_uso', 'garantias_otorgadas',
    'otros_conceptos', 'garantias_pref_a', 'garantias_pref_b',
    'sin_garantias', 'contragarantias_pref_a', 'contragarantias_pref_b',
    'sin_contragarantias', 'previsiones', 'deuda_cubierta',
    'proceso_judicial', 'refinanciaciones', 'recategorizacion_obligatoria',
    'situacion_juridica', 'irrecuperable', 'dias_atraso',
]

# Campos numéricos del BCRA que usan coma como decimal (ej: "991,0")
CAMPOS_MONTO = [
    'prestamos', 'garantias_otorgadas', 'otros_conceptos',
    'garantias_pref_a', 'garantias_pref_b', 'sin_garantias',
    'contragarantias_pref_a', 'contragarantias_pref_b',
    'sin_contragarantias', 'previsiones',
]


def _limpiar_monto(serie: pd.Series) -> pd.Series:
    """Convierte '991,0' → 991.0 (miles de pesos). Blancos y nulos → 0."""
    return (
        serie.astype(str)
             .str.strip()
             .str.replace(',', '.', regex=False)
             .str.replace(' ', '', regex=False)
             .pipe(pd.to_numeric, errors='coerce')
             .fillna(0.0)
    )


def cargar_deudores(path: str | Path) -> pd.DataFrame:
    """
    Carga deudores.txt y devuelve un DataFrame limpio.

    Parámetros
    ----------
    path : ruta al archivo deudores.txt

    Retorna
    -------
    DataFrame con una fila por (entidad, CUIT, mes).
    Los montos están en miles de pesos con un decimal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print(f"Cargando {path.name}...")

    df = pd.read_fwf(
        path,
        colspecs=DEUDORES_COLSPECS,
        names=DEUDORES_NOMBRES,
        header=None,
        dtype=str,       # todo como string primero, convertimos después
        encoding='latin-1',
    )

    # Limpiar identificadores
    df['nro_id']      = df['nro_id'].str.strip()
    df['cod_entidad'] = df['cod_entidad'].str.strip()
    df['fecha_info']  = df['fecha_info'].str.strip()
    df['actividad']   = df['actividad'].str.strip().replace('000', 'desconocido')

    # Situación: numérica, 1-5 (o 11 = cubierto por garantías A)
    df['situacion'] = pd.to_numeric(df['situacion'].str.strip(), errors='coerce')

    # Días de atraso
    df['dias_atraso'] = pd.to_numeric(df['dias_atraso'].str.strip(), errors='coerce').fillna(0).astype(int)

    # Flags binarios (0/1/9)
    for col in ['deuda_cubierta', 'proceso_judicial', 'refinanciaciones',
                'recategorizacion_obligatoria', 'situacion_juridica', 'irrecuperable']:
        df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce').fillna(0).astype(int)

    # Montos en miles de pesos
    for col in CAMPOS_MONTO:
        df[col] = _limpiar_monto(df[col])

    # Descartar campo sin uso
    df = df.drop(columns=['sin_uso', 'tipo_id'])

    print(f"  → {len(df):,} registros cargados | {df['nro_id'].nunique():,} CUITs únicos")
    return df


# ── Carga del 24DSF ──────────────────────────────────────────────────────────

# El 24DSF tiene 3 campos identificadores + (situacion, monto, proc_judicial) x 24 meses
# Total: 3 + 72 = 75 campos
# Los primeros 3: cod_entidad(5), tipo_id(2), nro_id(11)
# Luego bloques de 3 campos de longitud 2+12+1=15 chars c/u

def _build_24dsf_colspecs():
    specs = [
        (0,  5),   # cod_entidad
        (5,  7),   # tipo_id
        (7,  18),  # nro_id
    ]
    offset = 18
    for _ in range(24):
        specs.append((offset,      offset + 2))   # situacion
        specs.append((offset + 2,  offset + 14))  # monto
        specs.append((offset + 14, offset + 15))  # proceso_judicial
        offset += 15
    return specs


def _build_24dsf_nombres():
    nombres = ['cod_entidad', 'tipo_id', 'nro_id']
    for i in range(1, 25):
        nombres.append(f'sit_m{i:02d}')
        nombres.append(f'monto_m{i:02d}')
        nombres.append(f'procjud_m{i:02d}')
    return nombres


def cargar_24dsf(path: str | Path) -> pd.DataFrame:
    """
    Carga 24DSF.txt y devuelve un DataFrame en formato largo (long format).
    Cada fila = (CUIT, mes_relativo, situacion, monto).
    mes_relativo=1 es el más reciente, mes_relativo=24 el más antiguo.

    Parámetros
    ----------
    path : ruta al archivo 24DSF.txt
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print(f"Cargando {path.name}...")

    df_wide = pd.read_fwf(
        path,
        colspecs=_build_24dsf_colspecs(),
        names=_build_24dsf_nombres(),
        header=None,
        dtype=str,
        encoding='latin-1',
    )

    df_wide['nro_id'] = df_wide['nro_id'].str.strip()

    # Convertir de formato ancho a largo
    registros = []
    for mes in range(1, 25):
        col_sit  = f'sit_m{mes:02d}'
        col_monto = f'monto_m{mes:02d}'

        tmp = df_wide[['nro_id', col_sit, col_monto]].copy()
        tmp.columns = ['nro_id', 'situacion', 'monto']
        tmp['mes_relativo'] = mes  # 1 = más reciente
        registros.append(tmp)

    df_long = pd.concat(registros, ignore_index=True)

    # Limpiar valores
    df_long['situacion'] = pd.to_numeric(df_long['situacion'].str.strip(), errors='coerce')
    df_long['monto']     = _limpiar_monto(df_long['monto'])

    # Eliminar filas sin datos (meses donde el CUIT no fue reportado)
    df_long = df_long.dropna(subset=['situacion'])
    df_long = df_long[df_long['situacion'] > 0]

    print(f"  → {len(df_long):,} registros en formato largo | {df_long['nro_id'].nunique():,} CUITs únicos")
    return df_long
