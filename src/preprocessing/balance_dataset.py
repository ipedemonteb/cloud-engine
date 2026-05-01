"""
balance_dataset.py
------------------
Genera un dataset de entrenamiento reducido, ya sea balanceado por bins de
score_crediticio o mediante un muestreo aleatorio uniforme.

Objetivo
--------
Partiendo de data_processed/dataset_final.csv (muy grande), construye un CSV
mas chico para entrenamiento. Permite dos estrategias:
1) balance: downsampling para que cada bin de score_crediticio tenga la 
   misma cantidad de filas, evitando que una clase domine.
2) random: muestreo aleatorio uniforme sobre todo el dataset.

Bins usados para 'balance' (por defecto):
    - b0: [0.00, 0.25)
    - b1: [0.25, 0.50)
    - b2: [0.50, 0.75)
    - b3: [0.75, 1.00]

El script hace 2 pasadas en chunks:
1) Cuenta cuantas filas hay (y por bin si corresponde).
2) Selecciona filas con muestreo exacto (sin reemplazo) mediante la distribucion 
   hipergeometrica y escribe directo a disco para no cargar todo en RAM.

Uso ejemplo (desde src/):
    uv run python -m preprocessing.balance_dataset --strategy random --max-total-rows 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data_processed" / "dataset_final.csv"
DEFAULT_OUTPUT = ROOT / "data_processed" / "dataset_train_balanced.csv"

TARGET_COLUMN = "score_crediticio"
BIN_EDGES = [-0.000001, 0.25, 0.50, 0.75, 1.000001]
BIN_LABELS = ["b0_0_025", "b1_025_050", "b2_050_075", "b3_075_100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce dataset_final.csv por balanceo de bins o muestreo aleatorio"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV de entrada")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV de salida")
    parser.add_argument(
        "--target-col",
        default=TARGET_COLUMN,
        help="Nombre de la columna target",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Filas por chunk para lectura",
    )
    parser.add_argument(
        "--strategy",
        choices=["balance", "random"],
        default="balance",
        help="Estrategia: balance (por bins) o random (uniforme)",
    )
    parser.add_argument(
        "--rows-per-bin",
        type=int,
        default=None,
        help="Filas objetivo por bin (si no se indica, usa el minimo entre bins). Para 'random' equivale a esto x 4.",
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=None,
        help="Tope total de filas del dataset de salida",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--include-bin-col",
        action="store_true",
        help="Si se indica, agrega columna score_bin al CSV de salida",
    )
    return parser.parse_args()


def _as_bin_codes(score: pd.Series) -> pd.Series:
    binned = pd.cut(
        score,
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )

    if binned.isna().any():
        invalidas = int(binned.isna().sum())
        raise ValueError(
            f"Se encontraron {invalidas:,} filas con target fuera de rango [0.0, 1.0]"
        )

    return binned


def _contar_filas_y_bins(path: Path, target_col: str, chunksize: int, strategy: str) -> tuple[int, dict[str, int]]:
    counts = {label: 0 for label in BIN_LABELS}
    filas = 0

    reader = pd.read_csv(path, usecols=[target_col], chunksize=chunksize)
    for chunk in reader:
        if chunk[target_col].isna().any():
            raise ValueError("El dataset contiene NaN en el target")

        if strategy == "balance":
            bins = _as_bin_codes(chunk[target_col].astype(float))
            vc = bins.value_counts()
            for label in BIN_LABELS:
                counts[label] += int(vc.get(label, 0))

        filas += len(chunk)

    print("Conteo (dataset completo):")
    if strategy == "balance":
        for label in BIN_LABELS:
            cnt = counts[label]
            pct = (cnt / filas * 100.0) if filas else 0.0
            print(f"  {label:>10}: {cnt:>12,} ({pct:5.2f}%)")
    print(f"  {'TOTAL':>10}: {filas:>12,}")

    return filas, counts


def _resolver_objetivo_por_bin(
    counts: dict[str, int],
    rows_per_bin: int | None,
    max_total_rows: int | None,
) -> dict[str, int]:
    if rows_per_bin is not None and max_total_rows is not None:
        raise ValueError("Usar solo uno: --rows-per-bin o --max-total-rows")

    non_empty_counts = {k: v for k, v in counts.items() if v > 0}
    if not non_empty_counts:
        raise ValueError("No se encontraron datos validos para balancear (todos los bins vacios).")

    if rows_per_bin is not None:
        if rows_per_bin <= 0:
            raise ValueError("--rows-per-bin debe ser > 0")
        objetivo = rows_per_bin
    elif max_total_rows is not None:
        if max_total_rows <= 0:
            raise ValueError("--max-total-rows debe ser > 0")
        objetivo = max_total_rows // len(non_empty_counts)
        if objetivo <= 0:
            raise ValueError("--max-total-rows es demasiado chico para los bins disponibles")
    else:
        objetivo = min(non_empty_counts.values())

    objetivo_por_bin: dict[str, int] = {}
    for label in BIN_LABELS:
        if counts[label] > 0:
            objetivo_por_bin[label] = min(objetivo, counts[label])
        else:
            objetivo_por_bin[label] = 0

    return objetivo_por_bin


def _samplear_y_exportar(
    input_path: Path,
    output_path: Path,
    target_col: str,
    chunksize: int,
    strategy: str,
    objetivo_por_bin: dict[str, int] | None,
    objetivo_total: int | None,
    filas_total: int,
    counts_pool: dict[str, int],
    seed: int,
    include_bin_col: bool,
) -> dict[str, int] | int:
    rng = np.random.default_rng(seed)

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = True
    chunks = 0
    reader = pd.read_csv(input_path, chunksize=chunksize)

    if strategy == "balance":
        assert objetivo_por_bin is not None
        remaining_needed = dict(objetivo_por_bin)
        remaining_pool = dict(counts_pool)
        selected_counts = {label: 0 for label in BIN_LABELS}

        for chunk in reader:
            chunks += 1
            bins = _as_bin_codes(chunk[target_col].astype(float))
            selected_positions: list[np.ndarray] = []

            for label in BIN_LABELS:
                pos = np.flatnonzero((bins == label).to_numpy())
                n = int(pos.size)
                if n == 0:
                    continue

                need = remaining_needed[label]
                pool = remaining_pool[label]

                if need < 0 or pool < need:
                    raise RuntimeError(f"Estado invalido para {label}: need={need}, pool={pool}")

                if need == 0:
                    remaining_pool[label] -= n
                    continue

                bad = pool - need
                k = int(rng.hypergeometric(ngood=need, nbad=bad, nsample=n))

                if k > 0:
                    if k == n:
                        picked = pos
                    else:
                        picked = pos[rng.choice(n, size=k, replace=False)]
                    selected_positions.append(picked)
                    selected_counts[label] += k

                remaining_needed[label] -= k
                remaining_pool[label] -= n

            if selected_positions:
                all_pos = np.sort(np.concatenate(selected_positions))
                out_chunk = chunk.iloc[all_pos].copy()
                if include_bin_col:
                    out_chunk["score_bin"] = bins.iloc[all_pos].astype(str).to_numpy()

                out_chunk.to_csv(
                    output_path, mode="w" if write_header else "a", header=write_header, index=False
                )
                write_header = False

            if chunks % 10 == 0:
                print(f"  chunk {chunks:4d} | filas seleccionadas: {sum(selected_counts.values()):>12,}", end="\r")

        print()
        
        pendientes = {k: v for k, v in remaining_needed.items() if v != 0}
        if pendientes:
            raise RuntimeError(f"No se pudo completar el muestreo exacto: {pendientes}")
            
        return selected_counts

    else:  # strategy == "random"
        assert objetivo_total is not None
        remaining_needed_tot = objetivo_total
        remaining_pool_tot = filas_total
        selected_tot = 0

        for chunk in reader:
            chunks += 1
            n = len(chunk)
            
            if remaining_needed_tot == 0:
                # Ya cumplimos el objetivo, iteramos vacio (o break)
                # break podria estar bien si solo estamos muestreando, no hace falta leer mas
                break
                
            need = remaining_needed_tot
            pool = remaining_pool_tot
            
            bad = pool - need
            k = int(rng.hypergeometric(ngood=need, nbad=bad, nsample=n))
            
            if k > 0:
                if k == n:
                    picked = np.arange(n)
                else:
                    picked = rng.choice(n, size=k, replace=False)
                
                picked = np.sort(picked)
                out_chunk = chunk.iloc[picked].copy()
                
                if include_bin_col:
                    bins = _as_bin_codes(out_chunk[target_col].astype(float))
                    out_chunk["score_bin"] = bins.astype(str).to_numpy()

                out_chunk.to_csv(
                    output_path, mode="w" if write_header else "a", header=write_header, index=False
                )
                write_header = False
                selected_tot += k

            remaining_needed_tot -= k
            remaining_pool_tot -= n

            if chunks % 10 == 0:
                print(f"  chunk {chunks:4d} | filas seleccionadas: {selected_tot:>12,}", end="\r")

        print()
        
        if remaining_needed_tot != 0:
             raise RuntimeError(f"No se pudo completar el muestreo exacto. Faltaron: {remaining_needed_tot}")
             
        return selected_tot


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de entrada: {input_path}")

    print("=" * 70)
    print(f"  Reduccion de dataset_final (Estrategia: {args.strategy.upper()})")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target: {args.target_col}")
    print(f"Chunk:  {args.chunksize:,} filas")

    filas_total, counts = _contar_filas_y_bins(input_path, args.target_col, args.chunksize, args.strategy)

    objetivo_por_bin = None
    objetivo_total = None

    if args.strategy == "balance":
        objetivo_por_bin = _resolver_objetivo_por_bin(
            counts,
            rows_per_bin=args.rows_per_bin,
            max_total_rows=args.max_total_rows,
        )
        print("\nObjetivo por bin (train balanceado):")
        for label in BIN_LABELS:
            print(f"  {label:>10}: {objetivo_por_bin[label]:>12,}")
        print(f"  {'TOTAL':>10}: {sum(objetivo_por_bin.values()):>12,}")
    else:
        if args.max_total_rows is not None:
            objetivo_total = args.max_total_rows
        elif args.rows_per_bin is not None:
            objetivo_total = args.rows_per_bin * len(BIN_LABELS)
        else:
            raise ValueError("Para estrategia 'random' debe especificar --max-total-rows o --rows-per-bin")
            
        objetivo_total = min(objetivo_total, filas_total)
        print(f"\nObjetivo total (aleatorio): {objetivo_total:>12,}")

    print("\nMuestreando y exportando...")
    resultado = _samplear_y_exportar(
        input_path=input_path,
        output_path=output_path,
        target_col=args.target_col,
        chunksize=args.chunksize,
        strategy=args.strategy,
        objetivo_por_bin=objetivo_por_bin,
        objetivo_total=objetivo_total,
        filas_total=filas_total,
        counts_pool=counts,
        seed=args.seed,
        include_bin_col=args.include_bin_col,
    )

    if args.strategy == "balance":
        assert isinstance(resultado, dict)
        total = sum(resultado.values())
        print("\nResultado final:")
        for label in BIN_LABELS:
            cnt = resultado[label]
            pct = cnt / total * 100.0 if total else 0.0
            print(f"  {label:>10}: {cnt:>12,} ({pct:5.2f}%)")
        print(f"  {'TOTAL':>10}: {total:>12,}")
    else:
        print(f"\nResultado final: {resultado:>12,} filas extraidas aleatoriamente.")
        
    print(f"\n✓ Dataset guardado en: {output_path}")
    print("  Para entrenar:")
    print(f"  uv run python src/model/train_model.py --dataset {output_path}")


if __name__ == "__main__":
    main()
