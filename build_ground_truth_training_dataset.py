import os
import time
import json
import requests
from pathlib import Path
import pandas as pd
import numpy as np

# Cache and Directory Paths
CACHE_DIR = Path(".cache/openmeteo_miner")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INFO_TXT = "datasets/final_clean/CSV/CB_FD_Station_info.txt"
CATALOG_CSV = "datasets/final_clean/CSV/txson_expanded_station_catalog.csv"
CBFD_CSV_DIR = Path("datasets/final_clean/CSV")
TXS_DIR = Path("datasets/TxSON-Station-Files")

OUTPUT_CSV = "datasets/OpenMeteo_Synthetic_Grid_Dataset.csv"
GLOBAL_START = pd.Timestamp("2014-09-01")
GLOBAL_END = pd.Timestamp("2024-09-01")

# 6 Original TxSON Anchors Coordinates
TXS_COORDS = {
    'TXS01': (30.42047, -98.80332),
    'TXS02': (30.34771, -98.77726),
    'TXS03': (30.30154, -98.70618),
    'TXS04': (30.27982, -98.84109),
    'TXS05': (30.22915, -98.77663),
    'TXS06': (30.17846, -98.69485),
}

def parse_cbfd_coordinates(info_txt=INFO_TXT):
    """Parses Station_ID -> (lat, lon) from the harvested CB/FD info text file."""
    coords = {}
    if not os.path.exists(info_txt):
        print(f"[!] Warning: {info_txt} not found.")
        return coords

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


def fetch_openmeteo_archive(st_id, lat, lon, start_date="2014-09-01", end_date="2024-09-01", retries=8):
    """
    Fetches OpenMeteo archive weather + soil moisture proxy with local disk cache shield.
    """
    cache_file = CACHE_DIR / f"{st_id}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        df_cached = pd.read_csv(cache_file)
        df_cached['Date'] = pd.to_datetime(df_cached['Date'])
        return df_cached, True

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily=precipitation_sum,temperature_2m_max,temperature_2m_min&"
        f"hourly=soil_moisture_3_to_9cm&"
        f"timezone=auto"
    )

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()

            # Parse Daily features
            dates = pd.to_datetime(data['daily']['time'])
            precip = np.array(data['daily']['precipitation_sum'], dtype=np.float32)
            tmax = np.array(data['daily']['temperature_2m_max'], dtype=np.float32)
            tmin = np.array(data['daily']['temperature_2m_min'], dtype=np.float32)

            # Aggregate Hourly Soil Moisture to Daily Mean
            df_hourly = pd.DataFrame({
                'time': pd.to_datetime(data['hourly']['time']),
                'sm': np.array(data['hourly']['soil_moisture_3_to_9cm'], dtype=np.float32)
            })
            df_hourly['date'] = df_hourly['time'].dt.date
            daily_sm = df_hourly.groupby('date')['sm'].mean().values

            # Interpolate short missing gaps
            precip = np.nan_to_num(precip, nan=0.0)
            tmax = pd.Series(tmax).interpolate(limit=7).bfill().ffill().values
            tmin = pd.Series(tmin).interpolate(limit=7).bfill().ffill().values
            daily_sm = pd.Series(daily_sm).interpolate(limit=7).bfill().ffill().values

            df_clean = pd.DataFrame({
                'Date': dates,
                'Precipitation': precip,
                'T_Max': tmax,
                'T_Min': tmin,
                'OpenMeteo_SM': daily_sm
            })

            # Save immediately to cache
            df_clean.to_csv(cache_file, index=False)
            return df_clean, False

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = min(180, 15 * (2 ** attempt))
                print(f"    [-] Rate limited (429) on {st_id}, waiting {wait_time}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                print(f"    [-] HTTP error on {st_id}: {e}")
                time.sleep(5)
        except Exception as e:
            print(f"    [-] Error on {st_id}: {e}, retrying...")
            time.sleep(5)

    print(f"    [!] Failed to fetch {st_id} after {retries} retries.")
    return None, False


def load_insitu_ground_truth(st_id):
    """
    Loads and aggregates hourly in-situ sensor data to daily mean SWC_5.
    """
    # 1. Check TXS01-06
    if st_id.startswith("TXS"):
        st_num = st_id.replace("TXS0", "").replace("TXS", "")
        txs_file = TXS_DIR / f"Station{st_num}_filled_Data.csv"
        if txs_file.exists():
            df = pd.read_csv(txs_file)
            time_col = df.columns[0]
            df['Date'] = pd.to_datetime(df[time_col], errors='coerce').dt.normalize()
            if 'SWC_5' in df.columns:
                df_daily = df.dropna(subset=['Date', 'SWC_5']).groupby('Date')['SWC_5'].mean().reset_index()
                return df_daily

    # 2. Check CB/FD clean files
    cbfd_file = CBFD_CSV_DIR / f"{st_id}_filled_Data.csv"
    if cbfd_file.exists():
        df = pd.read_csv(cbfd_file)
        time_col = df.columns[0]
        df['Date'] = pd.to_datetime(df[time_col], errors='coerce').dt.normalize()
        if 'SWC_5' in df.columns:
            df_daily = df.dropna(subset=['Date', 'SWC_5']).groupby('Date')['SWC_5'].mean().reset_index()
            return df_daily

    return None


def build_unified_ground_truth_dataset():
    print("=" * 80)
    print("  TxSON GROUND-TRUTH & OPENMETEO WEATHER INTEGRATOR")
    print("  Mining & Merging 29 Ground Stations Only (Safe, Cached, Fast)")
    print("=" * 80)

    cbfd_coords = parse_cbfd_coordinates()
    catalog = pd.read_csv(CATALOG_CSV) if os.path.exists(CATALOG_CSV) else None

    # Build target list: 6 TXS + 23 CB/FD
    all_targets = []

    # 1. Add 6 TXS Anchors
    for sid, (lat, lon) in TXS_COORDS.items():
        all_targets.append({
            'Station_ID': sid,
            'Lat': lat,
            'Lon': lon,
            'Start': "2014-09-01",
            'End': "2024-09-01",
            'Type': 'TXS_Anchor'
        })

    # 2. Add CB/FD Stations
    missing_coords = []
    if catalog is not None:
        for _, row in catalog.iterrows():
            sid = row['Station_ID']
            if sid in cbfd_coords:
                lat, lon = cbfd_coords[sid]
                start = max(pd.Timestamp(row['Start_Date']), GLOBAL_START) if row['Start_Date'] != 'N/A' else GLOBAL_START
                end = min(pd.Timestamp(row['End_Date']), GLOBAL_END) if row['End_Date'] != 'N/A' else GLOBAL_END
                if start < end:
                    all_targets.append({
                        'Station_ID': sid,
                        'Lat': lat,
                        'Lon': lon,
                        'Start': start.strftime('%Y-%m-%d'),
                        'End': end.strftime('%Y-%m-%d'),
                        'Type': 'CBFD_Station'
                    })
            else:
                missing_coords.append(sid)

    print(f"[+] Total Station Targets: {len(all_targets)} Ground Stations")
    print(f"    - TXS Anchors: {sum(1 for t in all_targets if t['Type'] == 'TXS_Anchor')}")
    print(f"    - CB/FD Stations: {sum(1 for t in all_targets if t['Type'] == 'CBFD_Station')}")
    if missing_coords:
        print(f"    - Skipped (no coordinates in info file): {missing_coords}")

    merged_station_dfs = []

    for idx, target in enumerate(all_targets):
        sid = target['Station_ID']
        lat, lon = target['Lat'], target['Lon']
        start, end = target['Start'], target['End']

        print(f"\n[+] [{idx+1}/{len(all_targets)}] Processing {sid:6s} ({lat:.4f}, {lon:.4f}) | {start} to {end}...")

        # 1. Fetch / Load OpenMeteo Weather Features
        df_weather, was_cached = fetch_openmeteo_archive(sid, lat, lon, start, end)
        if df_weather is None or len(df_weather) == 0:
            print(f"    [!] Failed to acquire weather features for {sid}, skipping.")
            continue

        cache_status = "[CACHED]" if was_cached else "[DOWNLOADED]"
        print(f"    -> Weather Data: {len(df_weather)} daily records {cache_status}")

        # 2. Load In-Situ Ground Truth Probe Data
        df_gt = load_insitu_ground_truth(sid)
        if df_gt is None or len(df_gt) == 0:
            print(f"    [!] No valid in-situ SWC_5 ground truth found for {sid}, skipping.")
            continue

        print(f"    -> Ground Truth Data: {len(df_gt)} daily in-situ SWC_5 records (Mean SWC = {df_gt['SWC_5'].mean():.4f})")

        # 3. Inner Merge on Date
        df_merged = pd.merge(df_weather, df_gt, on='Date', how='inner')
        if len(df_merged) < 30:
            print(f"    [!] Insufficient overlapping dates ({len(df_merged)} rows) for {sid}, skipping.")
            continue

        # Format Columns for Pipeline Compatibility
        # 'SMAP_Moisture' is the primary Ground Truth target column expected by the LSTM pipeline
        df_merged['Station_ID'] = sid
        df_merged['Latitude'] = lat
        df_merged['Longitude'] = lon
        df_merged['SMAP_Moisture'] = df_merged['SWC_5'].clip(0.01, 0.60)

        cols_ordered = ['Station_ID', 'Date', 'Latitude', 'Longitude', 'Precipitation', 'T_Max', 'T_Min', 'OpenMeteo_SM', 'SMAP_Moisture']
        df_merged = df_merged[cols_ordered].sort_values('Date')
        merged_station_dfs.append(df_merged)
        print(f"    -> [SUCCESS] Merged {len(df_merged):,} matched ground-truth days for {sid}.")

        # Politeness delay for live requests
        if not was_cached:
            time.sleep(3.5)

    if not merged_station_dfs:
        raise RuntimeError("No station records were successfully merged.")

    df_final = pd.concat(merged_station_dfs, ignore_index=True)
    df_final = df_final.sort_values(['Station_ID', 'Date']).reset_index(drop=True)

    # Save outputs
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 80)
    print(f"[+] Dataset Construction Complete!")
    print(f"    - Total Observations: {len(df_final):,} daily matched records")
    print(f"    - Number of Active Validated Stations: {df_final['Station_ID'].nunique()}")
    print(f"    - Stations: {', '.join(sorted(df_final['Station_ID'].unique()))}")
    print(f"    - Date Span: {df_final['Date'].min().strftime('%Y-%m-%d')} to {df_final['Date'].max().strftime('%Y-%m-%d')}")
    print(f"    - Saved to: '{OUTPUT_CSV}'")
    print("=" * 80)

    return df_final


if __name__ == "__main__":
    build_unified_ground_truth_dataset()
