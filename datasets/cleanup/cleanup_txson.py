#!/usr/bin/env python3
"""
TxSON station cleaner.

Drop this file in the `cleanup` folder and run it with no arguments:

    python cleanup_txson.py

It will:
  1. find TxSON_data_2026-02-24 (or txson_data_26) next to / under cleanup
  2. pair SITE.dat with SITE_met.dat
  3. write Station-style hourly CSVs into cleanup/CSV/
  4. process stations in parallel across all CPU cores (Ryzen 9 5950X)

GPU is not used. This job is CSV parse + join + interpolate; a 3090
cannot speed that up. The 32-thread CPU is the right tool.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEPTHS = (5, 10, 20, 50)
SOIL_COLS = [f"SWC_{d}" for d in DEPTHS]
TEMP_COLS = [f"T_{d}" for d in DEPTHS]
METEO_COLS = ["Tair", "RH", "Wind speed", "Wind direction", "Srad"]
CORE_ORDER = SOIL_COLS + TEMP_COLS + ["Flag"] + METEO_COLS + ["Ppt"]

DATA_DIR_NAMES = (
    "TxSON_data_2026-02-24",
    "txson_data_26",
    "TxSON_data_26",
    "txson_data_2026-02-24",
)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def discover_data_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"data directory not found: {p}")
        return p

    here = script_dir()
    roots = [
        here,
        here.parent,
        here.parent / "datasets",
        here / "datasets",
        Path.cwd(),
        Path.cwd().parent,
    ]
    tried: list[Path] = []
    for root in roots:
        for name in DATA_DIR_NAMES:
            cand = (root / name).resolve()
            tried.append(cand)
            if cand.is_dir() and any(cand.glob("*.dat")):
                return cand
        # any TxSON_data_* folder that actually contains .dat files
        if root.is_dir():
            for cand in sorted(root.glob("TxSON_data*")) + sorted(root.glob("txson_data*")):
                if cand.is_dir() and any(cand.glob("*.dat")):
                    return cand.resolve()

    raise SystemExit(
        "Could not find the TxSON .dat folder.\n"
        "Looked for TxSON_data_2026-02-24 / txson_data_26 under:\n  "
        + "\n  ".join(str(p) for p in dict.fromkeys(tried))
        + "\nPass it explicitly:  python cleanup_txson.py --data path\\to\\TxSON_data_2026-02-24"
    )


def discover_outdir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    here = script_dir()
    csv_dir = here / "CSV"
    return csv_dir if csv_dir.is_dir() or here.name.lower() == "cleanup" else here / "CSV"


def pair_station_files(data_dir: Path) -> list[dict]:
    """Group SITE.dat + SITE_met.dat. Skip orphan _met files as primary inputs."""
    dats = sorted(data_dir.glob("*.dat"))
    by_stem = {p.stem: p for p in dats}
    jobs = []
    seen = set()
    for path in dats:
        stem = path.stem
        if stem.endswith("_met"):
            site = stem[: -len("_met")]
            if site in by_stem:
                continue
            site_id = site
            soil, met = None, path
        else:
            site_id = stem
            soil, met = path, by_stem.get(f"{stem}_met")
        if site_id in seen:
            continue
        seen.add(site_id)
        jobs.append({"site_id": site_id, "soil": soil, "met": met})
    return jobs


def _looks_like_header(line: str) -> bool:
    s = line.lstrip(",")
    return (
        s.startswith("Date,")
        or s.startswith("SWC_5,")
        or s.startswith("Ppt,")
        or s.startswith("Tair,")
    )


def find_header_line(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh):
            if _looks_like_header(html.unescape(raw)):
                return i
    raise ValueError(f"No CSV header found in {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "date": "Date",
        "time": "Date",
        "timestamp": "Date",
        "unnamed: 0": "Date",
        "ppt": "Ppt",
        "precip": "Ppt",
        "precipitation": "Ppt",
        "rain": "Ppt",
        "rainfall": "Ppt",
        "tair": "Tair",
        "t_air": "Tair",
        "air temp": "Tair",
        "air temperature": "Tair",
        "air_temperature": "Tair",
        "rh": "RH",
        "relative humidity": "RH",
        "relative_humidity": "RH",
        "humidity": "RH",
        "wind speed": "Wind speed",
        "wind_speed": "Wind speed",
        "ws": "Wind speed",
        "wspd": "Wind speed",
        "wind direction": "Wind direction",
        "wind_direction": "Wind direction",
        "wd": "Wind direction",
        "wdir": "Wind direction",
        "srad": "Srad",
        "solar": "Srad",
        "solar radiation": "Srad",
        "solar_radiation": "Srad",
        "rg": "Srad",
        "flag": "Flag",
    }
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        low = key.lower().replace("-", " ")
        rename[c] = mapping.get(low, mapping.get(low.replace("_", " "), key))
    return df.rename(columns=rename)


def read_dat(path: Path) -> pd.DataFrame:
    header_idx = find_header_line(path)
    df = pd.read_csv(path, skiprows=header_idx, na_values=["NaN", "nan", "NA", ""])
    df = _normalize_columns(df)
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    df = df.set_index("Date")
    df.index.name = None
    for col in df.columns:
        if col == "Flag":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full), int(len(full.difference(df.index)))


def fill_gaps(df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    out = df.copy()
    interp = [c for c in SOIL_COLS + TEMP_COLS + ["Tair", "RH", "Srad"] if c in out.columns]
    if interp:
        out[interp] = out[interp].interpolate(method="time", limit=limit, limit_direction="both")
    for col in ("Wind speed", "Wind direction"):
        if col in out.columns:
            out[col] = out[col].ffill(limit=limit).bfill(limit=limit)
    if "Ppt" in out.columns:
        out["Ppt"] = out["Ppt"].fillna(0.0)
    if "Flag" in out.columns:
        out["Flag"] = out["Flag"].fillna(0).astype("int64")
    return out


def arrange_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in CORE_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df.loc[:, ordered + extras]


def merge_soil_met(soil: pd.DataFrame | None, met: pd.DataFrame | None) -> pd.DataFrame:
    if soil is None and met is None:
        raise ValueError("no soil or met frame")
    if soil is None:
        return met
    if met is None:
        return soil
    # soil wins on overlapping measurement columns except meteo-only fields
    met_only = [c for c in met.columns if c not in soil.columns or c in METEO_COLS]
    # if both have Ppt, keep soil Ppt (in-situ gauge on the soil file)
    met_only = [c for c in met_only if not (c == "Ppt" and "Ppt" in soil.columns)]
    if not met_only:
        return soil
    return soil.join(met[met_only], how="outer")


def process_station(job: dict, outdir: str, fill: bool, fill_limit: int) -> dict:
    site = job["site_id"]
    soil_path = Path(job["soil"]) if job["soil"] else None
    met_path = Path(job["met"]) if job["met"] else None
    try:
        soil = read_dat(soil_path) if soil_path else None
        met = read_dat(met_path) if met_path else None
        df = merge_soil_met(soil, met)
        df, n_missing = to_hourly(df)
        if fill:
            df = fill_gaps(df, limit=fill_limit)
        df = arrange_columns(df)
        dest = Path(outdir) / f"{site}_filled_Data.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=True, index_label="")
        return {
            "site_id": site,
            "ok": True,
            "soil": soil_path.name if soil_path else "",
            "met": met_path.name if met_path else "",
            "n_out": int(len(df)),
            "hours_inserted": n_missing,
            "start": str(df.index.min()) if len(df) else "",
            "end": str(df.index.max()) if len(df) else "",
            "output": dest.name,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "site_id": site,
            "ok": False,
            "soil": soil_path.name if soil_path else "",
            "met": met_path.name if met_path else "",
            "n_out": 0,
            "hours_inserted": 0,
            "start": "",
            "end": "",
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def default_workers() -> int:
    n = os.cpu_count() or 8
    # 5950X = 16c/32t. Leave 2 threads for the parent + disk.
    return max(1, min(n - 2, 24))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TxSON .dat stations into filled hourly CSVs")
    parser.add_argument("inputs", nargs="*", help="optional files/dirs; default = auto-discover")
    parser.add_argument("--data", type=Path, default=None, help="TxSON_data_* folder")
    parser.add_argument("-o", "--outdir", type=Path, default=None, help="output folder (default cleanup/CSV)")
    parser.add_argument("--fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-limit", type=int, default=6)
    parser.add_argument("--workers", type=int, default=None, help="process pool size (default: CPU-2)")
    parser.add_argument("--combined", action="store_true", help="also write all_sites_filled_Data.csv")
    args = parser.parse_args()

    outdir = discover_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        files: list[Path] = []
        for item in args.inputs:
            path = Path(item)
            if path.is_dir():
                files.extend(sorted(path.glob("*.dat")))
            elif path.is_file():
                files.append(path)
            else:
                raise SystemExit(f"not found: {path}")
        data_dir = files[0].parent if files else discover_data_dir(args.data)
        jobs = pair_station_files(data_dir)
        wanted = {p.resolve() for p in files}
        jobs = [
            j
            for j in jobs
            if (j["soil"] and Path(j["soil"]).resolve() in wanted)
            or (j["met"] and Path(j["met"]).resolve() in wanted)
        ]
        if not jobs:
            # treat explicit files as soil-only
            jobs = [
                {"site_id": p.stem.replace("_met", ""), "soil": p if not p.stem.endswith("_met") else None,
                 "met": p if p.stem.endswith("_met") else None}
                for p in files
            ]
    else:
        data_dir = discover_data_dir(args.data)
        jobs = pair_station_files(data_dir)

    if not jobs:
        raise SystemExit("no station .dat files found")

    workers = args.workers or default_workers()
    workers = max(1, min(workers, len(jobs)))

    print(f"data   : {jobs[0]['soil'].parent if jobs[0]['soil'] else jobs[0]['met'].parent}")
    print(f"output : {outdir}")
    print(f"sites  : {len(jobs)}")
    print(f"workers: {workers}  (cpu_count={os.cpu_count()})")
    print()

    # serialize Paths for Windows spawn
    payload = [
        {
            "site_id": j["site_id"],
            "soil": str(j["soil"]) if j["soil"] else None,
            "met": str(j["met"]) if j["met"] else None,
        }
        for j in jobs
    ]

    inventory = []
    if workers == 1:
        for job in payload:
            rec = process_station(job, str(outdir), args.fill, args.fill_limit)
            inventory.append(rec)
            flag = "ok" if rec["ok"] else "FAIL"
            print(f"[{flag:4}] {rec['site_id']:<8} {rec['output']}  {rec['error']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(process_station, job, str(outdir), args.fill, args.fill_limit): job["site_id"]
                for job in payload
            }
            for fut in as_completed(futs):
                rec = fut.result()
                inventory.append(rec)
                flag = "ok" if rec["ok"] else "FAIL"
                extra = rec["error"] or f"{rec['n_out']} rows"
                print(f"[{flag:4}] {rec['site_id']:<8} {rec.get('output','')}  {extra}")

    inv = pd.DataFrame(inventory).sort_values("site_id")
    inv_path = outdir / "inventory.csv"
    inv.to_csv(inv_path, index=False)
    n_ok = int(inv["ok"].sum()) if "ok" in inv.columns else 0
    print(f"\n{n_ok}/{len(inv)} stations written to {outdir}")
    print(f"inventory: {inv_path}")

    if args.combined:
        frames = []
        for rec in inventory:
            if not rec["ok"]:
                continue
            path = outdir / rec["output"]
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.insert(0, "site_id", rec["site_id"])
            frames.append(df.reset_index().rename(columns={"index": "Date"}))
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            dest = outdir / "all_sites_filled_Data.csv"
            combined.to_csv(dest, index=False)
            print(f"combined: {dest} ({len(combined)} rows)")


if __name__ == "__main__":
    # Required on Windows so the process pool does not re-run main()
    import multiprocessing as mp

    mp.freeze_support()
    main()