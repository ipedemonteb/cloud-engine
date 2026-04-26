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


def _parse_situacion_24dsf(valor: str) -> int | None:
    sit = pd.to_numeric(valor, errors='coerce')
    if pd.isna(sit):
        return None
    sit = int(sit)
    return sit if 1 <= sit <= 5 else None


def features_desde_api(response_json: dict) -> dict | None:
    """
    Transforma la respuesta de /Deudas/Historicas/{CUIT} en un diccionario
    de features listo para el modelo.

    Parametros
    ----------
    response_json : dict
        Contenido del campo 'results' de la respuesta de la API del BCRA.

    Retorna
    -------
    dict con todas las features del modelo en el mismo formato que el CSV
    de entrenamiento, o None si no hay datos suficientes (menos de 7 periodos).
    """
    periodos = response_json.get('periodos', [])
    if len(periodos) < 7:
        return None

    meses = []
    for p in periodos:
        entidades = p.get('entidades', [])
        if not entidades:
            continue

        situaciones = [
            _parse_situacion_24dsf(e.get('situacion'))
            for e in entidades
        ]
        situaciones_validas = [s for s in situaciones if s is not None]
        if not situaciones_validas:
            continue

        meses.append({
            'periodo': p.get('periodo'),
            'situacion_max': max(situaciones_validas),
            'monto_total': sum(_parse_monto_24dsf(e.get('monto', 0)) for e in entidades),
            'dias_atraso_max': max(int(e.get('diasAtrasoPago') or 0) for e in entidades),
            'refinanciado': any(e.get('refinanciaciones') for e in entidades),
            'proceso_judicial': any(e.get('procesoJud') for e in entidades),
            'recategorizado': any(e.get('recategorizacionOblig') for e in entidades),
            'irrecuperable': any(e.get('irrecDisposicionTecnica') for e in entidades),
            'cant_entidades': len(entidades),
        })

    if len(meses) < 7:
        return None

    mes_actual = meses[0]
    hist = meses[6:]

    features = {
        'situacion': mes_actual['situacion_max'],
        'prestamos_total': mes_actual['monto_total'],
        'dias_atraso_max': mes_actual['dias_atraso_max'],
        'tiene_garantia_a': 0,
        'ratio_cobertura': 0.0,
        'refinanciado': int(mes_actual['refinanciado']),
        'proceso_judicial': int(mes_actual['proceso_judicial']),
        'recategorizado': int(mes_actual['recategorizado']),
        'irrecuperable': int(mes_actual['irrecuperable']),
        'cant_entidades': mes_actual['cant_entidades'],
    }

    sits = [m['situacion_max'] for m in hist]
    montos = [m['monto_total'] for m in hist]

    meses_en_sit1 = sum(1 for s in sits if s == 1)
    meses_sit_mala = sum(1 for s in sits if s >= 3)
    peor_situacion_24m = max(sits) if sits else None
    if len(sits) >= 4:
        bloque = sits[3:12]
        tendencia = round((sum(sits[:3]) / 3) - (sum(bloque) / len(bloque)), 3)
    else:
        tendencia = 0.0

    racha = 0
    for s in sits:
        if s == 1:
            racha += 1
        else:
            break

    monto_actual_hist = montos[0] if len(montos) >= 1 else 0
    monto_hace_12 = montos[11] if len(montos) >= 12 else (montos[-1] if montos else 0)
    variacion_monto_12m = (
        round((monto_actual_hist - monto_hace_12) / monto_hace_12, 3)
        if monto_hace_12 > 0
        else 0.0
    )

    monto_promedio_24m = round(sum(montos) / len(montos), 1) if montos else 0.0
    monto_max_24m = max(montos) if montos else 0.0
    meses_con_deuda = sum(1 for m in montos if m > 0)

    features.update({
        'meses_en_sit1': meses_en_sit1,
        'meses_sit_mala': meses_sit_mala,
        'peor_situacion_24m': peor_situacion_24m,
        'tendencia_situacion': tendencia,
        'racha_sit1_actual': racha,
        'variacion_monto_12m': variacion_monto_12m,
        'monto_promedio_24m': monto_promedio_24m,
        'monto_max_24m': monto_max_24m,
        'meses_con_deuda': meses_con_deuda,
        'actividad': 'desconocido',
    })

    return features


def _procesar_chunk_24dsf(chunk: pd.DataFrame, acum: dict, target_sits: dict) -> None:
    """
    Procesa un chunk del 24DSF y acumula features temporales por CUIT.
    Separa meses 1-6 (target) y meses 7-24 (features) para evitar leakage.
    """
    for row in chunk.itertuples(index=False):
        cuit = str(row.nro_id).strip()
        if not cuit:
            continue

        if cuit not in target_sits:
            target_sits[cuit] = [None] * 6

        if cuit not in acum:
            acum[cuit] = {
                'sits_7_24': [None] * 18,
                'montos_7_24': [0.0] * 18,
            }

        for i in range(1, 25):
            sit = _parse_situacion_24dsf(getattr(row, f'sit_m{i:02d}', None))

            if i <= 6:
                if sit is None:
                    continue

                prev = target_sits[cuit][i - 1]
                target_sits[cuit][i - 1] = sit if prev is None else max(prev, sit)
                continue

            if sit is None:
                continue

            idx = i - 7
            prev = acum[cuit]['sits_7_24'][idx]
            acum[cuit]['sits_7_24'][idx] = sit if prev is None else max(prev, sit)
            acum[cuit]['montos_7_24'][idx] += _parse_monto_24dsf(getattr(row, f'monto_m{i:02d}', 0))


def _features_temporales_desde_acum(acum: dict) -> pd.DataFrame:
    columnas = [
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

    filas = []
    for cuit, valores in acum.items():
        sits = []
        montos = []
        for sit, monto in zip(valores['sits_7_24'], valores['montos_7_24']):
            if sit is None:
                continue
            sits.append(sit)
            montos.append(monto)

        if not sits:
            continue

        meses_en_sit1 = sum(1 for s in sits if s == 1)
        meses_sit_mala = sum(1 for s in sits if s >= 3)
        peor_situacion_24m = max(sits)

        if len(sits) >= 4:
            bloque = sits[3:12]
            tendencia = round((sum(sits[:3]) / 3) - (sum(bloque) / len(bloque)), 3)
        else:
            tendencia = 0.0

        racha = 0
        for s in sits:
            if s == 1:
                racha += 1
            else:
                break

        monto_actual_hist = montos[0] if len(montos) >= 1 else 0
        monto_hace_12 = montos[11] if len(montos) >= 12 else (montos[-1] if montos else 0)
        variacion_monto_12m = (
            round((monto_actual_hist - monto_hace_12) / monto_hace_12, 3)
            if monto_hace_12 > 0
            else 0.0
        )

        monto_promedio_24m = round(sum(montos) / len(montos), 1) if montos else 0.0
        monto_max_24m = max(montos) if montos else 0.0
        meses_con_deuda = sum(1 for m in montos if m > 0)

        filas.append({
            'nro_id': cuit,
            'meses_en_sit1': meses_en_sit1,
            'meses_sit_mala': meses_sit_mala,
            'peor_situacion_24m': peor_situacion_24m,
            'tendencia_situacion': tendencia,
            'racha_sit1_actual': racha,
            'variacion_monto_12m': variacion_monto_12m,
            'monto_promedio_24m': monto_promedio_24m,
            'monto_max_24m': monto_max_24m,
            'meses_con_deuda': meses_con_deuda,
        })

    return pd.DataFrame(filas, columns=columnas)


def _targets_desde_sits(target_sits: dict) -> pd.DataFrame:
    try:
        from preprocessing.targets import calcular_score
    except ImportError:
        from targets import calcular_score

    filas = []
    for cuit, sits in target_sits.items():
        if len([s for s in sits if s is not None]) < 3:
            continue

        score = calcular_score(sits)
        if score is None:
            continue

        filas.append({
            'nro_id': cuit,
            'score_crediticio': score,
        })

    return pd.DataFrame(filas, columns=['nro_id', 'score_crediticio'])


def _cargar_24dsf_en_memoria(
    path: Path,
    persona_humana_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acum = {}
    target_sits = {}
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

        _procesar_chunk_24dsf(chunk, acum, target_sits)

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

    df = _features_temporales_desde_acum(acum)
    df_target = _targets_desde_sits(target_sits)

    del acum
    del target_sits
    print(f"    DataFrame final          : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"    CUITs con target valido  : {len(df_target):,}")
    return df, df_target


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


def _reducir_24dsf_buckets(bucket_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    target_frames = []
    buckets_con_datos = [p for p in bucket_paths if p.exists()]

    for i, bucket_path in enumerate(buckets_con_datos, start=1):
        acum = {}
        target_sits = {}

        reader = pd.read_csv(bucket_path, dtype=str, chunksize=CHUNK_SIZE)
        for chunk in reader:
            _procesar_chunk_24dsf(chunk, acum, target_sits)

        if acum:
            bucket_df = _features_temporales_desde_acum(acum)
            frames.append(bucket_df)

        if target_sits:
            bucket_target = _targets_desde_sits(target_sits)
            target_frames.append(bucket_target)

        print(
            f"  bucket {i:3d}/{len(buckets_con_datos):3d} | "
            f"CUITs acumulados en bucket: {len(acum):>10,}",
            end='\r',
        )

    print()
    if not frames:
        df_features = _features_temporales_desde_acum({})
    else:
        df_features = pd.concat(frames, ignore_index=True)

    if target_frames:
        df_target = pd.concat(target_frames, ignore_index=True)
    else:
        df_target = _targets_desde_sits({})

    return df_features, df_target


def _cargar_24dsf_low_ram(
    path: Path,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  Modo low-RAM activado ({buckets} buckets temporales)")
    with tempfile.TemporaryDirectory(prefix='bcra_24dsf_') as tmp:
        tmp_dir = Path(tmp)
        bucket_paths, filas_proc, filas_descartadas = _particionar_24dsf(
            path,
            tmp_dir,
            buckets,
            persona_humana_only,
        )
        df_features, df_target = _reducir_24dsf_buckets(bucket_paths)

    print("  ✓ Reducción de buckets completa")
    print(f"    Filas totales procesadas : {filas_proc:,}")
    if persona_humana_only:
        print(f"    Filas descartadas (no persona humana): {filas_descartadas:,}")
    print(f"    CUITs únicos encontrados : {len(df_features):,}")
    print(f"    DataFrame final          : {df_features.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"    CUITs con target valido  : {len(df_target):,}")
    return df_features, df_target


def cargar_24dsf(
    path: str | Path,
    low_ram: bool | None = None,
    buckets: int = LOW_RAM_BUCKETS,
    persona_humana_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga 24DSF.txt procesando en chunks.
    Devuelve una tupla con:
        - DataFrame de features temporales por CUIT (meses 7..24)
        - DataFrame target por CUIT con ['nro_id', 'score_crediticio']

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
