from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.load_data import features_desde_api


FEATURE_COLUMNS = [
    "situacion",
    "prestamos_total",
    "dias_atraso_max",
    "tiene_garantia_a",
    "ratio_cobertura",
    "refinanciado",
    "proceso_judicial",
    "recategorizado",
    "irrecuperable",
    "cant_entidades",
    "meses_en_sit1",
    "meses_sit_mala",
    "peor_situacion_24m",
    "tendencia_situacion",
    "racha_sit1_actual",
    "variacion_monto_12m",
    "monto_promedio_24m",
    "monto_max_24m",
    "meses_con_deuda",
    "actividad",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "data" / "query" / "query.json"
    default_output = repo_root / "data" / "query" / "query_procesed.csv"

    parser = argparse.ArgumentParser(
        description="Convierte JSON de la API BCRA a CSV para el modelo"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(default_input),
        help="Ruta a JSON o '-' para stdin",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(default_output),
        help="Ruta de salida CSV o '-' para stdout",
    )
    parser.add_argument(
        "--nro-id",
        default=None,
        help="Sobrescribe el nro_id (CUIT)",
    )
    parser.add_argument(
        "--actividad",
        default=None,
        help="Sobrescribe actividad (codigo)",
    )
    parser.add_argument(
        "--no-sort-periodos",
        action="store_true",
        help="No reordenar periodos por fecha desc",
    )
    return parser.parse_args()


def _read_json(input_path: str) -> dict:
    if input_path == "-":
        raw = sys.stdin.read().strip()
        if not raw:
            raise ValueError("No se recibio JSON por stdin")
        return json.loads(raw)

    path = Path(input_path).expanduser().resolve()
    if not path.exists() and path.suffix.lower() == ".json":
        alt_path = path.with_suffix(".JSON")
        if alt_path.exists():
            path = alt_path

    if not path.exists():
        raise FileNotFoundError(f"No se encontro JSON de entrada: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_results(payload: dict) -> dict:
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
        if isinstance(results, dict):
            return results
        raise ValueError("El campo 'results' debe ser un objeto JSON")

    if isinstance(payload, dict) and "periodos" in payload:
        return payload

    raise ValueError("No se encontro 'results' ni 'periodos' en el JSON")


def _sort_periodos_desc(results: dict) -> dict:
    periodos = results.get("periodos", [])
    if not isinstance(periodos, list):
        raise ValueError("El campo 'periodos' debe ser una lista")

    ordenados = sorted(
        periodos,
        key=lambda p: str(p.get("periodo", "")),
        reverse=True,
    )
    out = dict(results)
    out["periodos"] = ordenados
    return out


def main() -> None:
    args = parse_args()
    payload = _read_json(args.input)
    results = _extract_results(payload)

    if not args.no_sort_periodos:
        results = _sort_periodos_desc(results)

    nro_id = args.nro_id or results.get("identificacion")
    if nro_id is None:
        raise ValueError("No se encontro 'identificacion' para nro_id")

    features = features_desde_api(results)
    if features is None:
        raise ValueError("No hay suficientes periodos para generar features (min 7)")

    if args.actividad is not None:
        features["actividad"] = str(args.actividad)

    features["nro_id"] = str(nro_id)

    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Faltan features requeridas: {', '.join(missing)}")

    columns = ["nro_id", *FEATURE_COLUMNS]
    df = pd.DataFrame([features], columns=columns)

    if args.output == "-":
        df.to_csv(sys.stdout, index=False)
        return

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV listo para prediccion: {output_path}")


if __name__ == "__main__":
    main()
