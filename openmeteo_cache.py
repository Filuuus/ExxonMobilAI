import os
import json
import hashlib
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "openmeteo"

def get_cache_key(lat_chunk, lon_chunk, past_days=14, forecast_days=0):
    """Generates a stable cache key based on coordinates and parameters."""
    lat_repr = ",".join(f"{float(x):.4f}" for x in lat_chunk)
    lon_repr = ",".join(f"{float(x):.4f}" for x in lon_chunk)
    query_str = f"lat={lat_repr}&lon={lon_repr}&past_days={past_days}&forecast_days={forecast_days}"
    md5_hash = hashlib.md5(query_str.encode('utf-8')).hexdigest()
    return md5_hash, query_str

def parse_openmeteo_responses(responses):
    """Parses OpenMeteo JSON responses into standardized numpy feature sequences."""
    if not isinstance(responses, list):
        responses = [responses]
        
    chunk_results = []
    for data in responses:
        precip = np.array(data['daily']['precipitation_sum'], dtype=np.float32)
        tmax = np.array(data['daily']['temperature_2m_max'], dtype=np.float32)
        tmin = np.array(data['daily']['temperature_2m_min'], dtype=np.float32)
        
        hourly_time = data['hourly']['time']
        hourly_sm = data['hourly']['soil_moisture_3_to_9cm']
        df_hourly = pd.DataFrame({'time': pd.to_datetime(hourly_time), 'sm': hourly_sm})
        df_hourly['date'] = df_hourly['time'].dt.date.astype(str)
        
        daily_sm = df_hourly.groupby('date')['sm'].mean().values
        
        precip = np.nan_to_num(precip, nan=0.0)
        tmax = np.nan_to_num(tmax, nan=25.0)
        tmin = np.nan_to_num(tmin, nan=15.0)
        daily_sm = np.nan_to_num(daily_sm, nan=0.2)
        
        # Sequence shape: (14, 4) -> [Precipitation, T_Max, T_Min, SMAP_Moisture]
        chunk_results.append(np.column_stack((precip[-14:], tmax[-14:], tmin[-14:], daily_sm[-14:])))
        
    return chunk_results

def fetch_single_chunk_cached(lat_chunk, lon_chunk, cache_dir=DEFAULT_CACHE_DIR, max_age_hours=24):
    """
    Fetches OpenMeteo microclimate with persistent disk caching to protect against API rate limits.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_key, query_str = get_cache_key(lat_chunk, lon_chunk)
    cache_file = cache_path / f"{cache_key}.json"
    
    # Check cache freshness
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as fh:
                cached_data = json.load(fh)
            cache_mtime = cache_file.stat().st_mtime
            if (time.time() - cache_mtime) < (max_age_hours * 3600):
                # Valid cache hit
                return parse_openmeteo_responses(cached_data), True
        except Exception as e:
            # If reading corrupted cache fails, fallback to network
            pass
            
    # Network fetch
    lat_str = ",".join(map(str, lat_chunk))
    lon_str = ",".join(map(str, lon_chunk))
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat_str}&longitude={lon_str}&"
        f"daily=precipitation_sum,temperature_2m_max,temperature_2m_min&"
        f"hourly=soil_moisture_3_to_9cm&"
        f"past_days=14&forecast_days=0&timezone=auto"
    )
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Write atomically to cache
    temp_file = cache_path / f"{cache_key}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    temp_file.replace(cache_file)
    
    return parse_openmeteo_responses(data), False
