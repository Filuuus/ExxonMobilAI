import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from openmeteo_historical_miner import get_map_grid_coordinates, fetch_historical_station_cached

INFO_TXT = "datasets/final_clean/CSV/CB_FD_Station_info.txt"
CATALOG_CSV = "datasets/final_clean/CSV/txson_expanded_station_catalog.csv"
GLOBAL_START = pd.Timestamp("2014-09-01")
GLOBAL_END = pd.Timestamp("2024-09-01")


def parse_station_coordinates(info_txt=INFO_TXT):
    """Parses Station_ID -> (lat, lon) from the harvested CB/FD info text file."""
    coords = {}
    with open(info_txt, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith(("Soil", "24 Hours", "7 days", "1 Month", "1 Year", "Latitude", "Longitude")):
            current_id = line
            lat = lon = None
            j = i + 1
            while j < len(lines) and lines[j].strip() != "":
                l = lines[j].strip()
                if l.startswith("Latitude"):
                    lat = float(l.split("\t")[-1])
                elif l.startswith("Longitude"):
                    lon = float(l.split("\t")[-1])
                j += 1
            if lat is not None and lon is not None:
                coords[current_id] = (lat, lon)
            i = j
        else:
            i += 1
    return coords


def build_cbfd_targets():
    """Matches harvested coordinates against the station catalog's valid date ranges."""
    coords = parse_station_coordinates()
    catalog = pd.read_csv(CATALOG_CSV)
    catalog = catalog[catalog["Start_Date"] != "N/A"]

    records = []
    missing = []
    for _, row in catalog.iterrows():
        sid = row["Station_ID"]
        if sid not in coords:
            missing.append(sid)
            continue
        lat, lon = coords[sid]
        start = max(pd.Timestamp(row["Start_Date"]), GLOBAL_START)
        end = min(pd.Timestamp(row["End_Date"]), GLOBAL_END)
        if start >= end:
            continue
        records.append({
            "ID": sid,
            "Lat": lat,
            "Lon": lon,
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d"),
        })

    if missing:
        print(f"[!] No harvested coordinates for {len(missing)} catalog stations, skipping: {missing}")

    return pd.DataFrame(records)


def fetch_cbfd_dataset(df_targets):
    results = []
    failed_rows = []
    print(f"[+] Querying archive-api.open-meteo.com for {len(df_targets)} CB/FD station coordinates...")
    print("    -> Using Sequential fetching to respect API rate limits.")

    for idx, row in df_targets.iterrows():
        df_st, was_cached = fetch_historical_station_cached(row, start_date=row["Start"], end_date=row["End"])
        if df_st is not None:
            results.append(df_st)
            print(f"    -> [{idx+1}/{len(df_targets)}] {row['ID']:6s} ({row['Start']} to {row['End']}): {len(df_st)} daily records acquired.{' [cached]' if was_cached else ''}")
        else:
            print(f"    -> [{idx+1}/{len(df_targets)}] {row['ID']:6s}: FAILED, will retry after cooldown.")
            failed_rows.append(row)
        if not was_cached:
            time.sleep(6.0)

    if failed_rows:
        print(f"\n[+] Cooling down 60s before retrying {len(failed_rows)} failed station(s): {[r['ID'] for r in failed_rows]}")
        time.sleep(60)
        still_failed = []
        for row in failed_rows:
            df_st, was_cached = fetch_historical_station_cached(row, start_date=row["Start"], end_date=row["End"])
            if df_st is not None:
                results.append(df_st)
                print(f"    -> Retry succeeded: {row['ID']:6s} ({len(df_st)} daily records acquired).")
            else:
                still_failed.append(row['ID'])
            if not was_cached:
                time.sleep(6.0)
        if still_failed:
            print(f"[!] Permanently failed after retry pass: {still_failed}")

    if not results:
        raise RuntimeError("No CB/FD station data was successfully fetched.")

    return pd.concat(results, ignore_index=True)


def build_expanded_openmeteo_dataset(out_path="datasets/OpenMeteo_Synthetic_Grid_Dataset.csv"):
    print("=" * 80)
    print("  OPENMETEO EXPANDED MINER (TXS01-06 + CB/FD STATIONS)")
    print("=" * 80)

    # 1. TXS01-06 anchors + spatial interpolation grid + Tamaulipas grid (existing approach)
    df_grid = get_map_grid_coordinates()
    # Relabel the 6 real TxSON anchors with their plain station IDs (GRID_TX_A_TXS01 -> TXS01)
    # so they line up with the real in-situ station names used for blind validation.
    df_grid["ID"] = df_grid["ID"].str.replace(r"^GRID_TX_A_(TXS\d+)$", r"\1", regex=True)

    print(f"[+] TXS01-06 Anchor + Spatial Grid + Tamaulipas Grid: {len(df_grid)} coordinates.")

    results = []
    for idx, row in df_grid.iterrows():
        df_st, was_cached = fetch_historical_station_cached(row, start_date="2014-09-01", end_date="2024-09-01")
        if df_st is not None:
            results.append(df_st)
        print(f"    -> Progress: {idx+1}/{len(df_grid)} grid coordinates acquired ({row['ID']}){' [cached]' if was_cached else ''}.")
        if not was_cached:
            time.sleep(6.0)

    # 2. CB/FD real station coordinates (relevant per-station timeframe)
    df_targets = build_cbfd_targets()
    print(f"\n[+] CB/FD Real Station Coordinates Matched: {len(df_targets)}")
    print(df_targets[["ID", "Lat", "Lon", "Start", "End"]].to_string(index=False))

    df_cbfd = fetch_cbfd_dataset(df_targets)
    results.append(df_cbfd)

    # 3. Unify and save
    df_unified = pd.concat(results, ignore_index=True)
    df_unified = df_unified.sort_values(["Station_ID", "Date"]).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"[+] Download Complete! Total Daily Observations: {len(df_unified):,}")
    print(f"    - Spanning {df_unified['Station_ID'].nunique()} Spatial Coordinates/Stations")

    df_unified.to_csv(out_path, index=False)
    print(f"[+] Saved Unified OpenMeteo Training Dataset to: '{out_path}'")
    print("=" * 80)
    return df_unified


if __name__ == "__main__":
    build_expanded_openmeteo_dataset()
