import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from openmeteo_historical_miner import get_map_grid_coordinates

CACHE_DIR = Path(".cache/openmeteo_miner")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_archive(lat, lon, start_date="2014-09-01", end_date="2024-09-01", max_retries=8):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat:.4f}&longitude={lon:.4f}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily=precipitation_sum,temperature_2m_max,temperature_2m_min&"
        f"hourly=soil_moisture_3_to_9cm&"
        f"timezone=auto"
    )
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code == 429:
                wait_time = min(120, 10 * (2 ** attempt))
                print(f"    [-] Rate limited (429), waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            
            df_daily = pd.DataFrame(data['daily'])
            df_daily.rename(columns={
                'time': 'Date',
                'precipitation_sum': 'Precipitation',
                'temperature_2m_max': 'T_Max',
                'temperature_2m_min': 'T_Min'
            }, inplace=True)
            df_daily['Date'] = pd.to_datetime(df_daily['Date'])
            
            df_hourly = pd.DataFrame(data['hourly'])
            df_hourly['Date'] = pd.to_datetime(df_hourly['time']).dt.floor('D')
            df_sm = df_hourly.groupby('Date')['soil_moisture_3_to_9cm'].mean().reset_index()
            df_sm.rename(columns={'soil_moisture_3_to_9cm': 'OpenMeteo_SM'}, inplace=True)
            
            df_merged = pd.merge(df_daily, df_sm, on='Date', how='left')
            return df_merged
        except requests.exceptions.RequestException as e:
            print(f"    [-] Network error: {e}, retrying...")
            time.sleep(10)
    return None

def main():
    print("=" * 80)
    print("  MINING & CACHING ALL 79 SPATIAL GRID COORDINATES")
    print("=" * 80)
    
    grid_df = get_map_grid_coordinates()
    print(f"[+] Total Grid Coordinates to Verify: {len(grid_df)}")
    
    all_grid_records = []
    
    for idx, row in grid_df.iterrows():
        st_id = row['ID']
        lat = row['Lat']
        lon = row['Lon']
        
        cache_file = CACHE_DIR / f"{st_id}_2014-09-01_2024-09-01.csv"
        
        # Check for matching cache file with prefix
        existing = list(CACHE_DIR.glob(f"{st_id}_*.csv"))
        
        if existing and existing[0].stat().st_size > 1000:
            df_weather = pd.read_csv(existing[0])
            df_weather['Date'] = pd.to_datetime(df_weather['Date'])
            print(f"[{idx+1}/{len(grid_df)}] {st_id:20s} ({lat:.4f}, {lon:.4f}) -> [CACHED] {len(df_weather)} days")
        else:
            print(f"[{idx+1}/{len(grid_df)}] {st_id:20s} ({lat:.4f}, {lon:.4f}) -> [DOWNLOADING]...")
            df_weather = fetch_archive(lat, lon, start_date="2014-09-01", end_date="2024-09-01")
            if df_weather is not None and len(df_weather) > 0:
                df_weather.to_csv(cache_file, index=False)
                print(f"    -> Successfully cached {len(df_weather)} records to {cache_file.name}")
            else:
                print(f"    [!] Failed to download {st_id}")
                continue
            time.sleep(3.0)
            
        df_weather['Station_ID'] = st_id
        df_weather['Latitude'] = lat
        df_weather['Longitude'] = lon
        all_grid_records.append(df_weather)
        
    if all_grid_records:
        full_grid_df = pd.concat(all_grid_records, ignore_index=True)
        out_path = "datasets/OpenMeteo_All_Grid_Coordinates_Dataset.csv"
        full_grid_df.to_csv(out_path, index=False)
        print("\n" + "=" * 80)
        print(f"[+] Complete Grid Dataset Saved: '{out_path}'")
        print(f"    - Total Observations: {len(full_grid_df):,}")
        print(f"    - Total Coordinates: {full_grid_df['Station_ID'].nunique()}")
        print("=" * 80)

if __name__ == "__main__":
    main()
