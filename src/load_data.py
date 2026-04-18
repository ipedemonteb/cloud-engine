"""
load_data.py
------------
Parsea los archivos de ancho fijo del BCRA usando procesamiento en chunks.

Incluye dos estrategias:
    - En memoria: chunk + acumulador global por CUIT (más rápida)
    - Low-RAM: particiona a disco por hash de CUIT y reduce por bucket
      (más robusta para archivos gigantes cuando la RAM es limitada)

Estrategia:
    - Lee el archivo en bloques de CHUNK_SIZE filas
    - Agrega por CUIT con el mismo criterio de negocio
    - En low-RAM, evita un acumulador global gigante en memoria
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import tempfile
import zlib


# ── Tamaño de chunk ──────────────────────────────────────────────────────────
# 500k filas ≈ ~200MB en memoria por chunk. Ajustar según RAM disponible.
# Más grande = más rápido pero más RAM. Más chico = más lento pero más seguro.
CHUNK_SIZE = 500_000

# Si el archivo supera este tamaño, usar automáticamente estrategia low-RAM.
LOW_RAM_FILE_THRESHOLD_GB = 1.0

# Cantidad de buckets para particionar a disco en modo low-RAM.
# Más buckets = menos RAM por bucket, pero más archivos temporales.
LOW_RAM_BUCKETS = 128

# Prefijos de identificador para personas humanas.
PERSONA_HUMANA_PREFIXES = frozenset({'20', '23', '24', '27'})


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

# Solo las columnas que necesitamos — el resto se ignora al leer
COLUMNAS_UTILES = [
    'cod_entidad', 'nro_id', 'actividad', 'situacion',
    'prestamos', 'garantias_pref_a', 'garantias_pref_b',
    'proceso_judicial', 'refinanciaciones', 'recategorizacion_obligatoria',
    'irrecuperable', 'dias_atraso',
]


def _limpiar_monto(valor: str) -> float:
    """Convierte '991,0' → 991.0. Blancos y nulos → 0."""
    try:
        return float(str(valor).strip().replace(',', '.').replace(' ', ''))
    except (ValueError, TypeError):
        return 0.0


def _limpiar_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y tipifica un chunk crudo."""
    df['nro_id']      = df['nro_id'].str.strip()
    df['cod_entidad'] = df['cod_entidad'].str.strip()
    df['actividad']   = df['actividad'].str.strip().replace('000', 'desconocido')

    df['situacion']   = pd.to_numeric(df['situacion'].str.strip(), errors='coerce')
    df['dias_atraso'] = pd.to_numeric(df['dias_atraso'].str.strip(), errors='coerce').fillna(0).astype(int)

    for col in ['proceso_judicial', 'refinanciaciones',
                'recategorizacion_obligatoria', 'irrecuperable']:
        df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce').fillna(0).astype(int)

    for col in ['prestamos', 'garantias_pref_a', 'garantias_pref_b']:
        df[col] = df[col].apply(_limpiar_monto)

    # Descartar filas con CUIT vacío o situación inválida
    df = df[df['nro_id'].str.len() > 0]
    df = df[df['situacion'].notna()]

    return df


def _acumular_chunk(acum: dict, df: pd.DataFrame) -> None:
    """
    Agrega un chunk en el diccionario acumulador.
    Para cada CUIT guarda los valores necesarios para calcular
    max, sum y any al final — sin guardar filas individuales.
    """
    for row in df.itertuples(index=False):
        cuit = row.nro_id
        a    = acum[cuit]

        # Situación: peor (máximo)
        sit = int(row.situacion) if not pd.isna(row.situacion) else 1
        if sit > a['situacion']:
            a['situacion'] = sit

        # Montos: suma
        a['prestamos_total']   += row.prestamos
        a['garantias_pref_a']  += row.garantias_pref_a
        a['garantias_pref_b']  += row.garantias_pref_b

        # Días de atraso: máximo
        if row.dias_atraso > a['dias_atraso_max']:
            a['dias_atraso_max'] = row.dias_atraso

        # Flags: cualquiera activo
        if row.proceso_judicial == 1:
            a['proceso_judicial'] = 1
        if row.refinanciaciones == 1:
            a['refinanciado'] = 1
        if row.recategorizacion_obligatoria == 1:
            a['recategorizado'] = 1
        if row.irrecuperable == 1:
            a['irrecuperable'] = 1

        # Entidades distintas
        a['entidades'].add(row.cod_entidad)

        # Actividad: primera no-desconocida
        if a['actividad'] == 'desconocido' and row.actividad != 'desconocido':
            a['actividad'] = row.actividad


def _acum_default() -> dict:
    """Valor inicial del acumulador para un CUIT nuevo."""
    return {
        'situacion':      0,
        'prestamos_total': 0.0,
        'garantias_pref_a': 0.0,
        'garantias_pref_b': 0.0,
        'dias_atraso_max': 0,
        'proceso_judicial': 0,
        'refinanciado':    0,
        'recategorizado':  0,
        'irrecuperable':   0,
        'entidades':       set(),
        'actividad':       'desconocido',
    }


def _acumulador_a_dataframe(acum: dict) -> pd.DataFrame:
    """Convierte el diccionario acumulador en un DataFrame de features."""
    filas = []
    for cuit, a in acum.items():
        filas.append({
            'nro_id':           cuit,
            'situacion':        a['situacion'],
            'prestamos_total':  round(a['prestamos_total'], 1),
            'garantias_pref_a': round(a['garantias_pref_a'], 1),
            'garantias_pref_b': round(a['garantias_pref_b'], 1),
            'tiene_garantia_a': int(a['garantias_pref_a'] > 0),
            'ratio_cobertura':  round(
                a['garantias_pref_a'] / a['prestamos_total'], 4
            ) if a['prestamos_total'] > 0 else 0.0,
            'dias_atraso_max':  a['dias_atraso_max'],
            'proceso_judicial': a['proceso_judicial'],
            'refinanciado':     a['refinanciado'],
            'recategorizado':   a['recategorizado'],
            'irrecuperable':    a['irrecuperable'],
            'cant_entidades':   len(a['entidades']),
            'actividad':        a['actividad'],
        })
    return pd.DataFrame(filas)


def _hash_bucket(valor: str, buckets: int) -> int:
    return zlib.crc32(str(valor).encode('utf-8')) % buckets


def _usar_low_ram(path: Path, low_ram: bool | None) -> bool:
    if low_ram is not None:
        return low_ram
    return path.stat().st_size >= int(LOW_RAM_FILE_THRESHOLD_GB * 1e9)


def _filtrar_personas_humanas(
    df: pd.DataFrame,
    persona_humana_only: bool,
) -> tuple[pd.DataFrame, int]:
    if not persona_humana_only or df.empty:
        return df, 0

    mask = df['nro_id'].astype(str).str[:2].isin(PERSONA_HUMANA_PREFIXES)
    descartadas = int((~mask).sum())
    return df[mask], descartadas


def _normalizar_chunk_deudores_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['nro_id'] = df['nro_id'].astype(str).str.strip()
    df['cod_entidad'] = df['cod_entidad'].astype(str).str.strip()
    df['actividad'] = df['actividad'].fillna('desconocido').astype(str).str.strip()
    df.loc[df['actividad'].isin(['', '000', 'nan', 'None']), 'actividad'] = 'desconocido'

    df['situacion'] = pd.to_numeric(df['situacion'], errors='coerce').fillna(1)
    df['dias_atraso'] = pd.to_numeric(df['dias_atraso'], errors='coerce').fillna(0).astype(int)

    for col in ['proceso_judicial', 'refinanciaciones', 'recategorizacion_obligatoria', 'irrecuperable']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col in ['prestamos', 'garantias_pref_a', 'garantias_pref_b']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = df[df['nro_id'].str.len() > 0]
    return df


def _cargar_deudores_en_memoria(
    path: Path,
    persona_humana_only: bool,
) -> pd.DataFrame:
    acum = defaultdict(_acum_default)
    chunks_proc = 0
    filas_proc = 0
    filas_descartadas = 0

    reader = pd.read_fwf(
        path,
        colspecs=DEUDORES_COLSPECS,
        names=DEUDORES_NOMBRES,
        header=None,
        dtype=str,
        encoding='latin-1',
        chunksize=CHUNK_SIZE,
        usecols=COLUMNAS_UTILES,
    )

    for chunk in reader:
        chunk = _limpiar_chunk(chunk)
        chunk, descartadas = _filtrar_personas_humanas(chunk, persona_humana_only)
        filas_descartadas += descartadas
        if chunk.empty:
            chunks_proc += 1
            continue

        _acumular_chunk(acum, chunk)

        chunks_proc += 1
        filas_proc += len(chunk)
        cuits_vistos = len(acum)

        print(
            f"  chunk {chunks_proc:3d} | filas procesadas: {filas_proc:>12,} "
            f"| CUITs únicos: {cuits_vistos:>10,}",
            end='\r',
        )

    print()
    print("  ✓ Procesamiento completo")
    print(f"    Filas totales procesadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    print(f"    CUITs únicos encontrados : {len(acum):,}")

    df = _acumulador_a_dataframe(acum)
    del acum

    print(f"    DataFrame final          : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def _particionar_deudores(
    path: Path,
    tmp_dir: Path,
    buckets: int,
    persona_humana_only: bool,
) -> tuple[list[Path], int, int]:
    bucket_paths = [tmp_dir / f'deudores_bucket_{i:03d}.csv' for i in range(buckets)]

    chunks_proc = 0
    filas_proc = 0
    filas_descartadas = 0

    reader = pd.read_fwf(
        path,
        colspecs=DEUDORES_COLSPECS,
        names=DEUDORES_NOMBRES,
        header=None,
        dtype=str,
        encoding='latin-1',
        chunksize=CHUNK_SIZE,
        usecols=COLUMNAS_UTILES,
    )

    for chunk in reader:
        chunk = _limpiar_chunk(chunk)
        chunk, descartadas = _filtrar_personas_humanas(chunk, persona_humana_only)
        filas_descartadas += descartadas
        if chunk.empty:
            chunks_proc += 1
            continue

        chunk['_bucket'] = chunk['nro_id'].map(lambda x: _hash_bucket(x, buckets))

        for bucket_id, parte in chunk.groupby('_bucket', sort=False):
            bucket_path = bucket_paths[int(bucket_id)]
            write_header = not bucket_path.exists()
            parte.drop(columns=['_bucket']).to_csv(
                bucket_path,
                mode='a',
                index=False,
                header=write_header,
            )

        chunks_proc += 1
        filas_proc += len(chunk)
        print(
            f"  chunk {chunks_proc:3d} | filas particionadas: {filas_proc:>12,}",
            end='\r',
        )

    print()
    print("  ✓ Particionado completo")
    print(f"    Filas totales particionadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    return bucket_paths, filas_proc, filas_descartadas


def _reducir_deudores_buckets(bucket_paths: list[Path]) -> pd.DataFrame:
    frames = []
    buckets_con_datos = [p for p in bucket_paths if p.exists()]

    for i, bucket_path in enumerate(buckets_con_datos, start=1):
        acum = defaultdict(_acum_default)

        reader = pd.read_csv(bucket_path, dtype=str, chunksize=CHUNK_SIZE)
        for chunk in reader:
            chunk = _normalizar_chunk_deudores_bucket(chunk)
            _acumular_chunk(acum, chunk)

        bucket_df = _acumulador_a_dataframe(acum)
        frames.append(bucket_df)

        print(
            f"  bucket {i:3d}/{len(buckets_con_datos):3d} | "
            f"CUITs acumulados en bucket: {len(bucket_df):>10,}",
            end='\r',
        )

    print()
    if not frames:
        return pd.DataFrame(columns=[
            'nro_id', 'situacion', 'prestamos_total', 'garantias_pref_a',
            'garantias_pref_b', 'tiene_garantia_a', 'ratio_cobertura',
            'dias_atraso_max', 'proceso_judicial', 'refinanciado',
            'recategorizado', 'irrecuperable', 'cant_entidades', 'actividad',
        ])

    return pd.concat(frames, ignore_index=True)


def _cargar_deudores_low_ram(
    path: Path,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> pd.DataFrame:
    print(f"  Modo low-RAM activado ({buckets} buckets temporales)")
    with tempfile.TemporaryDirectory(prefix='bcra_deudores_') as tmp:
        tmp_dir = Path(tmp)
        bucket_paths, filas_proc, filas_descartadas = _particionar_deudores(
            path,
            tmp_dir,
            buckets,
            persona_humana_only,
        )
        df = _reducir_deudores_buckets(bucket_paths)

    print("  ✓ Reducción de buckets completa")
    print(f"    Filas totales procesadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    print(f"    CUITs únicos encontrados : {len(df):,}")
    print(f"    DataFrame final          : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ── Carga principal de deudores.txt ─────────────────────────────────────────

def cargar_deudores(
    path: str | Path,
    low_ram: bool | None = None,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> pd.DataFrame:
    """
    Carga deudores.txt procesando en chunks para manejar archivos grandes.
    Devuelve un DataFrame con una fila por CUIT, ya agregado.

    Parámetros
    ----------
    path : ruta al archivo deudores.txt
    persona_humana_only : si True, conserva solo identificadores de
        persona humana (prefijos 20, 23, 24, 27)

    Retorna
    -------
    DataFrame con features actuales agregadas por CUIT.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print(f"Cargando {path.name} en chunks de {CHUNK_SIZE:,} filas...")
    print(f"  Tamaño del archivo: {path.stat().st_size / 1e9:.1f} GB")
    if persona_humana_only:
        print("  Filtro persona humana: activo (prefijos 20/23/24/27)")

    if _usar_low_ram(path, low_ram):
        return _cargar_deudores_low_ram(
            path,
            buckets=buckets,
            persona_humana_only=persona_humana_only,
        )
    return _cargar_deudores_en_memoria(path, persona_humana_only=persona_humana_only)


# ── Carga del 24DSF.txt ──────────────────────────────────────────────────────

def _build_24dsf_colspecs():
    specs = [(0, 5), (5, 7), (7, 18)]  # cod_entidad, tipo_id, nro_id
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


def _parse_monto_24dsf(valor: str) -> float:
    try:
        return float(str(valor).strip().replace(',', '.').replace(' ', ''))
    except (ValueError, TypeError):
        return 0.0


def _procesar_chunk_24dsf(chunk: pd.DataFrame, acum: dict) -> None:
    """
    Procesa un chunk del 24DSF y acumula features temporales por CUIT.
    Calcula directamente los agregados para no guardar filas individuales.
    """
    sit_cols   = [f'sit_m{i:02d}' for i in range(1, 25)]
    monto_cols = [f'monto_m{i:02d}' for i in range(1, 25)]

    for row in chunk.itertuples(index=False):
        cuit = str(row.nro_id).strip()
        if not cuit:
            continue

        # Extraer situaciones y montos de los 24 meses
        sits   = []
        montos = []
        for i, (sc, mc) in enumerate(zip(sit_cols, monto_cols)):
            s = pd.to_numeric(getattr(row, sc, None), errors='coerce')
            m = _parse_monto_24dsf(getattr(row, mc, 0))
            if pd.notna(s) and s > 0:
                sits.append(int(s))
                montos.append(m)
            else:
                sits.append(None)
                montos.append(None)

        # Calcular features temporales
        sits_validos   = [s for s in sits   if s is not None]
        montos_validos = [m for m in montos if m is not None]

        if not sits_validos:
            continue

        sits_arr = np.array(sits_validos)

        meses_en_sit1    = int((sits_arr == 1).sum())
        meses_sit_mala   = int((sits_arr >= 3).sum())
        peor_situacion   = int(sits_arr.max())
        meses_con_deuda  = int(sum(1 for m in montos_validos if m and m > 0))

        # Tendencia: promedio últimos 3m vs meses 4-12
        ultimos_3  = np.mean(sits_arr[:3])  if len(sits_arr) >= 3 else sits_arr.mean()
        anteriores = np.mean(sits_arr[3:12]) if len(sits_arr) > 3  else sits_arr.mean()
        tendencia  = round(float(ultimos_3 - anteriores), 3)

        # Racha actual en situación 1
        racha = 0
        for s in sits_arr:
            if s == 1:
                racha += 1
            else:
                break

        # Evolución de monto
        montos_arr     = np.array([m for m in montos_validos if m is not None])
        monto_actual   = montos_arr[0]  if len(montos_arr) >= 1  else 0
        monto_hace_12  = montos_arr[11] if len(montos_arr) >= 12 else montos_arr[-1]
        variacion_12m  = round(
            float((monto_actual - monto_hace_12) / monto_hace_12), 3
        ) if monto_hace_12 > 0 else 0.0

        monto_promedio = round(float(montos_arr.mean()), 1) if len(montos_arr) > 0 else 0.0
        monto_max      = float(montos_arr.max())            if len(montos_arr) > 0 else 0.0

        # Si el CUIT ya fue visto en un chunk anterior, tomamos el peor caso
        # (puede aparecer en múltiples entidades)
        if cuit in acum:
            a = acum[cuit]
            acum[cuit] = {
                'meses_en_sit1':       min(a['meses_en_sit1'],      meses_en_sit1),
                'meses_sit_mala':      max(a['meses_sit_mala'],     meses_sit_mala),
                'peor_situacion_24m':  max(a['peor_situacion_24m'], peor_situacion),
                'tendencia_situacion': max(a['tendencia_situacion'],tendencia),
                'racha_sit1_actual':   min(a['racha_sit1_actual'],  racha),
                'variacion_monto_12m': a['variacion_monto_12m'],
                'monto_promedio_24m':  round((a['monto_promedio_24m'] + monto_promedio) / 2, 1),
                'monto_max_24m':       max(a['monto_max_24m'],      monto_max),
                'meses_con_deuda':     max(a['meses_con_deuda'],    meses_con_deuda),
            }
        else:
            acum[cuit] = {
                'meses_en_sit1':       meses_en_sit1,
                'meses_sit_mala':      meses_sit_mala,
                'peor_situacion_24m':  peor_situacion,
                'tendencia_situacion': tendencia,
                'racha_sit1_actual':   racha,
                'variacion_monto_12m': variacion_12m,
                'monto_promedio_24m':  monto_promedio,
                'monto_max_24m':       monto_max,
                'meses_con_deuda':     meses_con_deuda,
            }


def _cargar_24dsf_en_memoria(
    path: Path,
    persona_humana_only: bool,
) -> pd.DataFrame:
    acum = {}
    chunks_proc = 0
    filas_proc = 0
    filas_descartadas = 0

    reader = pd.read_fwf(
        path,
        colspecs=_build_24dsf_colspecs(),
        names=_build_24dsf_nombres(),
        header=None,
        dtype=str,
        encoding='latin-1',
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk['nro_id'] = chunk['nro_id'].astype(str).str.strip()
        chunk = chunk[chunk['nro_id'].str.len() > 0]
        chunk, descartadas = _filtrar_personas_humanas(chunk, persona_humana_only)
        filas_descartadas += descartadas
        if chunk.empty:
            chunks_proc += 1
            continue

        _procesar_chunk_24dsf(chunk, acum)

        chunks_proc += 1
        filas_proc += len(chunk)

        print(
            f"  chunk {chunks_proc:3d} | filas procesadas: {filas_proc:>12,} "
            f"| CUITs únicos: {len(acum):>10,}",
            end='\r',
        )

    print()
    print("  ✓ Procesamiento completo")
    print(f"    Filas totales procesadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    print(f"    CUITs únicos encontrados : {len(acum):,}")

    df = pd.DataFrame.from_dict(acum, orient='index')
    df.index.name = 'nro_id'
    df = df.reset_index()

    del acum
    print(f"    DataFrame final          : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def _particionar_24dsf(
    path: Path,
    tmp_dir: Path,
    buckets: int,
    persona_humana_only: bool,
) -> tuple[list[Path], int, int]:
    bucket_paths = [tmp_dir / f'dsf24_bucket_{i:03d}.csv' for i in range(buckets)]

    chunks_proc = 0
    filas_proc = 0
    filas_descartadas = 0

    reader = pd.read_fwf(
        path,
        colspecs=_build_24dsf_colspecs(),
        names=_build_24dsf_nombres(),
        header=None,
        dtype=str,
        encoding='latin-1',
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk['nro_id'] = chunk['nro_id'].astype(str).str.strip()
        chunk = chunk[chunk['nro_id'].str.len() > 0]
        chunk, descartadas = _filtrar_personas_humanas(chunk, persona_humana_only)
        filas_descartadas += descartadas
        if chunk.empty:
            chunks_proc += 1
            continue

        chunk['_bucket'] = chunk['nro_id'].map(lambda x: _hash_bucket(x, buckets))

        for bucket_id, parte in chunk.groupby('_bucket', sort=False):
            bucket_path = bucket_paths[int(bucket_id)]
            write_header = not bucket_path.exists()
            parte.drop(columns=['_bucket']).to_csv(
                bucket_path,
                mode='a',
                index=False,
                header=write_header,
            )

        chunks_proc += 1
        filas_proc += len(chunk)
        print(
            f"  chunk {chunks_proc:3d} | filas particionadas: {filas_proc:>12,}",
            end='\r',
        )

    print()
    print("  ✓ Particionado completo")
    print(f"    Filas totales particionadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    return bucket_paths, filas_proc, filas_descartadas


def _reducir_24dsf_buckets(bucket_paths: list[Path]) -> pd.DataFrame:
    frames = []
    buckets_con_datos = [p for p in bucket_paths if p.exists()]

    for i, bucket_path in enumerate(buckets_con_datos, start=1):
        acum = {}

        reader = pd.read_csv(bucket_path, dtype=str, chunksize=CHUNK_SIZE)
        for chunk in reader:
            _procesar_chunk_24dsf(chunk, acum)

        if acum:
            bucket_df = pd.DataFrame.from_dict(acum, orient='index')
            bucket_df.index.name = 'nro_id'
            bucket_df = bucket_df.reset_index()
            frames.append(bucket_df)

        print(
            f"  bucket {i:3d}/{len(buckets_con_datos):3d} | "
            f"CUITs acumulados en bucket: {len(acum):>10,}",
            end='\r',
        )

    print()
    if not frames:
        return pd.DataFrame(columns=[
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
        ])

    return pd.concat(frames, ignore_index=True)


def _cargar_24dsf_low_ram(
    path: Path,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> pd.DataFrame:
    print(f"  Modo low-RAM activado ({buckets} buckets temporales)")
    with tempfile.TemporaryDirectory(prefix='bcra_24dsf_') as tmp:
        tmp_dir = Path(tmp)
        bucket_paths, filas_proc, filas_descartadas = _particionar_24dsf(
            path,
            tmp_dir,
            buckets,
            persona_humana_only,
        )
        df = _reducir_24dsf_buckets(bucket_paths)

    print("  ✓ Reducción de buckets completa")
    print(f"    Filas totales procesadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    print(f"    CUITs únicos encontrados : {len(df):,}")
    print(f"    DataFrame final          : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def cargar_24dsf(
    path: str | Path,
    low_ram: bool | None = None,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> pd.DataFrame:
    """
    Carga 24DSF.txt procesando en chunks.
    Devuelve un DataFrame con una fila por CUIT con features temporales.

    Parámetros
    ----------
    path : ruta al archivo 24DSF.txt
    persona_humana_only : si True, conserva solo identificadores de
        persona humana (prefijos 20, 23, 24, 27)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print(f"Cargando {path.name} en chunks de {CHUNK_SIZE:,} filas...")
    print(f"  Tamaño del archivo: {path.stat().st_size / 1e9:.1f} GB")
    print(f"  (Este archivo es grande — puede tardar varios minutos)")
    if persona_humana_only:
        print("  Filtro persona humana: activo (prefijos 20/23/24/27)")

    if _usar_low_ram(path, low_ram):
        return _cargar_24dsf_low_ram(
            path,
            buckets=buckets,
            persona_humana_only=persona_humana_only,
        )
    return _cargar_24dsf_en_memoria(path, persona_humana_only=persona_humana_only)
