import os
import sys
import concurrent.futures
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configure multi-threading for AMD Ryzen 9 5950X (16C/32T)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from forecaster_engine_v2 import SoilMoistureForecaster
from openmeteo_cache import fetch_single_chunk_cached

def fetch_openmeteo_cached_parallel(lats, lons, chunk_size=20):
    """
    Parallelized OpenMeteo fetching with local disk cache shield.
    Prevents API rate limits and accelerates iteration.
    """
    lat_chunks = [lats[i:i+chunk_size] for i in range(0, len(lats), chunk_size)]
    lon_chunks = [lons[i:i+chunk_size] for i in range(0, len(lons), chunk_size)]
    
    all_results = [None] * len(lat_chunks)
    cache_hits = 0
    cache_misses = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(lat_chunks), 16)) as executor:
        future_to_idx = {
            executor.submit(fetch_single_chunk_cached, lat_c, lon_c): idx
            for idx, (lat_c, lon_c) in enumerate(zip(lat_chunks, lon_chunks))
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk_data, was_cached = future.result()
            all_results[idx] = chunk_data
            if was_cached:
                cache_hits += 1
            else:
                cache_misses += 1
                
    print(f"    -> OpenMeteo Acquisition Stats: {cache_hits} chunk cache hits, {cache_misses} network fetches.")
    
    # Flatten in order
    flat_results = []
    for chunk in all_results:
        flat_results.extend(chunk)
    return flat_results

def generate_spatial_timeline(output_file='capstone_spatial_forecast.html'):
    print("=" * 75)
    print("  CAPSTONE PROPOSAL: 1-YEAR FUTURE BLIND SPATIAL FORECAST (v2)")
    print("  Accelerated Engine + OpenMeteo Disk Cache Shield")
    print("=" * 75)
    
    # 1. Load FULL Expanded TxSON Dataset (90,000+ rows across 27 stations)
    print("\n[+] Ingesting Expanded TxSON Unified Dataset (27 stations, 2014 to Present)...")
    expanded_dataset_path = 'datasets/TxSON_Expanded_Unified_Dataset.csv'
    orig_dataset_path = 'datasets/TxSON_SMAP_Weather_Merged.csv'
    
    if os.path.exists(expanded_dataset_path):
        dataset_path = expanded_dataset_path
        print(f"    -> Using Expanded Network Dataset: '{dataset_path}'")
    else:
        dataset_path = orig_dataset_path
        print(f"    -> Fallback to Original Dataset: '{dataset_path}'")
        
    df_raw = pd.read_csv(dataset_path)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_clean = df_raw.dropna(subset=['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']).sort_values('Date')
    
    print(f"    -> Training Dataset Span: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')} ({len(df_clean):,} rows across {df_clean['Station_ID'].nunique()} stations)")
    
    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    data_vals = df_clean[features].values
    
    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(data_vals)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(data_vals[:, 3].reshape(-1, 1))
    
    look_back = 14
    
    # Load pre-trained expanded model or train on the dataset
    saved_model_path = "expanded_lstm_model.keras"
    if os.path.exists(saved_model_path):
        print(f"\n[+] Loading High-Capacity Pre-Trained LSTM Engine from '{saved_model_path}'...")
        model = tf.keras.models.load_model(saved_model_path)
        print("    -> Pre-trained Expanded LSTM Model successfully loaded.")
    else:
        print("\n[+] Training Shared LSTM Engine on 100% of Available TxSON Ground Data...")
        X, y = [], []
        for i in range(look_back, len(data_scaled)):
            X.append(data_scaled[i-look_back:i])
            y.append(data_scaled[i, 3])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(look_back, 4)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)
        print("    -> LSTM Model Training Complete.")
    
    engine = SoilMoistureForecaster(model, scaler_X, scaler_y, look_back=look_back)
    
    # 2. Build 79 Spatial Coordinates (TxSON Anchors + Fredericksburg & Tamaulipas Grids)
    txson_stations = {
        'TXS01': (30.42047, -98.80332),
        'TXS02': (30.34771, -98.77726),
        'TXS03': (30.30154, -98.70618),
        'TXS04': (30.27982, -98.84109),
        'TXS05': (30.22915, -98.77663),
        'TXS06': (30.17846, -98.69485)
    }
    tamaulipas_center = (25.4479, -98.0122) # General Francisco Gonzalez Villarreal
    resolution = 0.045
    
    all_lats, all_lons, all_labels = [], [], []
    
    # Add Texas Anchors & Grid
    for station_id, (lat, lon) in txson_stations.items():
        all_lats.append(lat)
        all_lons.append(lon)
        all_labels.append(f"Texas Anchor ({station_id})")
        
        for lat_off in [-resolution, 0, resolution]:
            for lon_off in [-resolution, 0, resolution]:
                if lat_off == 0 and lon_off == 0:
                    continue
                all_lats.append(lat + lat_off)
                all_lons.append(lon + lon_off)
                all_labels.append("Texas (Grid Interpolation)")
                
    # Add Tamaulipas Anchor & Grid
    all_lats.append(tamaulipas_center[0])
    all_lons.append(tamaulipas_center[1])
    all_labels.append("Tamaulipas Anchor (TM01)")
    
    for lat_off in [-resolution*2, -resolution, 0, resolution, resolution*2]:
        for lon_off in [-resolution*2, -resolution, 0, resolution, resolution*2]:
            if lat_off == 0 and lon_off == 0:
                continue
            all_lats.append(tamaulipas_center[0] + lat_off)
            all_lons.append(tamaulipas_center[1] + lon_off)
            all_labels.append("Tamaulipas (Grid Interpolation)")
            
    # 3. Parallelized Live / Cached OpenMeteo Microclimate Acquisition
    print(f"\n[+] Acquiring OpenMeteo Microclimate Data for {len(all_lats)} Coordinates (Disk Cache Protected)...")
    recent_batch = fetch_openmeteo_cached_parallel(all_lats, all_lons)
    seed_batch = np.array(recent_batch, dtype=np.float32) # Shape: (79, 14, 4)
    
    # 4. Synthesize Future Meteorological Forcing for Next 364 Days (Seasonal Climatology)
    print("\n[+] Synthesizing 1-Year Climatological Forcing Profile (Precip + Thermal Cycle)...")
    horizon = 364
    future_forcing_batch = np.zeros((len(all_lats), horizon, 3), dtype=np.float32)
    
    days_in_year = np.arange(horizon)
    tmax_cycle = 25.0 + 10.0 * np.sin(2 * np.pi * (days_in_year - 100) / 365.0)
    tmin_cycle = 12.0 + 9.0 * np.sin(2 * np.pi * (days_in_year - 100) / 365.0)
    
    for i in range(len(all_lats)):
        future_forcing_batch[i, :, 0] = 0.0 # Drydown default (Precip)
        future_forcing_batch[i, :, 1] = tmax_cycle
        future_forcing_batch[i, :, 2] = tmin_cycle
        
    # 5. Vectorized 1-Year Blind Future Forecast
    print(f"\n[+] Executing Vectorized 364-Day Blind Forecast Across All {len(all_lats)} Locations...")
    future_predictions = engine.predict_future_batch(
        seed_batch, 
        forecast_horizon=horizon, 
        future_forcing_batch=future_forcing_batch
    )
    
    # 6. Format Timeline Records (52 Weekly Scrubber Frames)
    records = []
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day in range(0, horizon, 7):
        date_str = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')
        for i, (lat, lon, label) in enumerate(zip(all_lats, all_lons, all_labels)):
            records.append({
                'Date': date_str,
                'Location': label,
                'Latitude': lat,
                'Longitude': lon,
                'Predicted_SMAP': max(0.05, min(0.50, float(future_predictions[i][day])))
            })
            
    df_forecast = pd.DataFrame(records)
    
    min_val = float(df_forecast['Predicted_SMAP'].min())
    max_val = float(df_forecast['Predicted_SMAP'].max())
    
    print(f"\n[+] Dynamic Palette Range: [{min_val:.4f}, {max_val:.4f}] m³/m³")

    # 7. Render High-Contrast City-Scale Heatmap
    print("\n[+] Rendering Interactive Spatial Timeline Heatmap...")
    fig = px.density_map(
        df_forecast, 
        lat="Latitude", 
        lon="Longitude", 
        z="Predicted_SMAP",
        hover_name="Location", 
        hover_data={"Date": False, "Predicted_SMAP": ':.4f', "Latitude": False, "Longitude": False},
        animation_frame="Date",
        animation_group="Location",
        color_continuous_scale=px.colors.diverging.RdYlBu,
        range_color=[min_val, max_val],
        radius=95,
        zoom=5.5, 
        center={"lat": 27.85, "lon": -98.4},
        title="Capstone Proposal: 1-Year Future Soil Moisture Trend (Weekly Scrubber)",
        labels={'Predicted_SMAP': 'Volumetric Soil Moisture (m³/m³)'}
    )

    fig.update_layout(
        map_style="carto-positron",
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(
            title="SMAP Moisture",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside",
            tickformat=".2f"
        )
    )
    
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    fig.write_html(output_file)
    print(f"\n[+] Success! Final 1-Year Spatial Forecast Heatmap Saved To:\n    {os.path.abspath(output_file)}")
    print("=" * 75)

if __name__ == "__main__":
    generate_spatial_timeline()
