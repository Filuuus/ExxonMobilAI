import os
import time
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from pathlib import Path

# Extract coordinates logic from spatial map
def get_map_grid_coordinates():
    txson_stations = {
        'TXS01': (30.42047, -98.80332),
        'TXS02': (30.34771, -98.77726),
        'TXS03': (30.30154, -98.70618),
        'TXS04': (30.27982, -98.84109),
        'TXS05': (30.22915, -98.77663),
        'TXS06': (30.17846, -98.69485)
    }
    tamaulipas_center = (25.4479, -98.0122)
    resolution = 0.045
    
    records = []
    
    # Texas Anchors & Grid
    for station_id, (lat, lon) in txson_stations.items():
        records.append({'ID': f"GRID_TX_A_{station_id}", 'Lat': lat, 'Lon': lon})
        idx = 1
        for lat_off in [-resolution, 0, resolution]:
            for lon_off in [-resolution, 0, resolution]:
                if lat_off == 0 and lon_off == 0: continue
                records.append({'ID': f"GRID_TX_N_{station_id}_{idx}", 'Lat': lat + lat_off, 'Lon': lon + lon_off})
                idx += 1
                
    # Tamaulipas Anchor & Grid
    records.append({'ID': "GRID_TAM_A_01", 'Lat': tamaulipas_center[0], 'Lon': tamaulipas_center[1]})
    idx = 1
    for lat_off in [-resolution*2, -resolution, 0, resolution, resolution*2]:
        for lon_off in [-resolution*2, -resolution, 0, resolution, resolution*2]:
            if lat_off == 0 and lon_off == 0: continue
            records.append({'ID': f"GRID_TAM_N_{idx}", 'Lat': tamaulipas_center[0] + lat_off, 'Lon': tamaulipas_center[1] + lon_off})
            idx += 1
            
    return pd.DataFrame(records)

def fetch_historical_station(row, start_date="2014-09-01", end_date="2024-09-01"):
    st_id = row['ID']
    lat = row['Lat']
    lon = row['Lon']
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily=precipitation_sum,temperature_2m_max,temperature_2m_min&"
        f"hourly=soil_moisture_3_to_9cm&"
        f"timezone=auto"
    )
    
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
    
    # Replace NaNs
    precip = np.nan_to_num(precip, nan=0.0)
    tmax = pd.Series(tmax).interpolate(limit=7).bfill().ffill().values
    tmin = pd.Series(tmin).interpolate(limit=7).bfill().ffill().values
    daily_sm = pd.Series(daily_sm).interpolate(limit=7).bfill().ffill().values
    
    df_clean = pd.DataFrame({
        'Station_ID': st_id,
        'Date': dates,
        'Precipitation': precip,
        'T_Max': tmax,
        'T_Min': tmin,
        'SMAP_Moisture': daily_sm
    })
    
    return df_clean

def fetch_historical_station_with_retry(row, start_date="2014-09-01", end_date="2024-09-01", retries=8):
    for attempt in range(retries):
        try:
            return fetch_historical_station(row, start_date, end_date)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = min(180, 20 * (2 ** attempt))
                print(f"    [-] Rate limited (429) on {row['ID']}, waiting {wait_time}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
            else:
                print(f"[-] HTTP Error on {row['ID']}: {e}")
                return None
        except Exception as e:
            print(f"[-] Error on {row['ID']}: {e}")
            return None
    print(f"    [!] Exhausted all {retries} retries for {row['ID']}, giving up.")
    return None

STATION_FETCH_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "openmeteo_miner"

def fetch_historical_station_cached(row, start_date="2014-09-01", end_date="2024-09-01", retries=8):
    """Disk-cached wrapper: skips the network entirely if this exact station/date-range
    was already fetched successfully in a prior (possibly interrupted) run."""
    STATION_FETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = STATION_FETCH_CACHE_DIR / f"{row['ID']}_{start_date}_{end_date}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=['Date']), True

    df_st = fetch_historical_station_with_retry(row, start_date, end_date, retries=retries)
    if df_st is not None:
        df_st.to_csv(cache_file, index=False)
    return df_st, False

def build_openmeteo_dataset():
    print("=" * 80)
    print("  OPENMETEO HISTORICAL MINER (SYNTHETIC GRID TRAINING SET)")
    print("=" * 80)
    
    df_coords = get_map_grid_coordinates()
    print(f"[+] Extracted {len(df_coords)} interpolation coordinates from Spatial Map Grid.")
    
    results = []
    
    print("[+] Querying archive-api.open-meteo.com (10-Year History: 2014-2024)...")
    print("    -> Using Sequential fetching to respect API rate limits (Wait ~5-10 minutes).")
    
    for idx, row in df_coords.iterrows():
        df_st, was_cached = fetch_historical_station_cached(row)
        if df_st is not None:
            results.append(df_st)

        print(f"    -> Progress: {idx+1}/{len(df_coords)} grids acquired ({row['ID']}){' [cached]' if was_cached else ''}.")
        if not was_cached:
            time.sleep(6.0) # Wait between requests to stay well clear of burst rate limits
            
    df_unified = pd.concat(results, ignore_index=True)
    df_unified = df_unified.sort_values(['Station_ID', 'Date']).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print(f"[+] Download Complete! Total Daily Observations: {len(df_unified):,}")
    print(f"    - Spanning {df_unified['Station_ID'].nunique()} Spatial Coordinates")
    
    out_path = 'datasets/OpenMeteo_Synthetic_Grid_Dataset.csv'
    df_unified.to_csv(out_path, index=False)
    print(f"[+] Saved Synthetic Training Dataset to: '{out_path}'")
    
if __name__ == "__main__":
    build_openmeteo_dataset()
