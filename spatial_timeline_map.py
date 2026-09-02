import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from forecaster_engine import SoilMoistureForecaster

def fetch_single_chunk(lat_chunk, lon_chunk):
    """Fetches OpenMeteo microclimate for a specific chunk of coordinates."""
    lat_str = ",".join(map(str, lat_chunk))
    lon_str = ",".join(map(str, lon_chunk))
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&hourly=soil_moisture_3_to_9cm&past_days=14&forecast_days=0&timezone=auto"
    
    r = requests.get(url)
    responses = r.json()
    if not isinstance(responses, list):
        responses = [responses]
        
    chunk_results = []
    for data in responses:
        precip = np.array(data['daily']['precipitation_sum'])
        tmax = np.array(data['daily']['temperature_2m_max'])
        tmin = np.array(data['daily']['temperature_2m_min'])
        
        hourly_time = data['hourly']['time']
        hourly_sm = data['hourly']['soil_moisture_3_to_9cm']
        df_hourly = pd.DataFrame({'time': pd.to_datetime(hourly_time), 'sm': hourly_sm})
        df_hourly['date'] = df_hourly['time'].dt.date.astype(str)
        
        daily_sm = df_hourly.groupby('date')['sm'].mean().values
        
        precip = np.nan_to_num(precip, nan=0.0)
        tmax = np.nan_to_num(tmax, nan=25.0)
        tmin = np.nan_to_num(tmin, nan=15.0)
        daily_sm = np.nan_to_num(daily_sm, nan=0.2)
        
        chunk_results.append(np.column_stack((precip[-14:], tmax[-14:], tmin[-14:], daily_sm[-14:])))
        
    return chunk_results

def fetch_openmeteo_parallel(lats, lons, chunk_size=20):
    """Parallelized OpenMeteo fetching across multithreaded workers."""
    lat_chunks = [lats[i:i+chunk_size] for i in range(0, len(lats), chunk_size)]
    lon_chunks = [lons[i:i+chunk_size] for i in range(0, len(lons), chunk_size)]
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lat_chunks)) as executor:
        futures = [executor.submit(fetch_single_chunk, lats_c, lons_c) for lats_c, lons_c in zip(lat_chunks, lon_chunks)]
        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())
            
    return all_results

def generate_spatial_timeline(output_file='capstone_spatial_forecast.html'):
    print("=" * 75)
    print("  CAPSTONE PROPOSAL: 1-YEAR FUTURE BLIND SPATIAL FORECAST (PARALLELIZED)")
    print("=" * 75)
    
    # 1. Load FULL TxSON Dataset (2015 to Present) - No Data Cutoff
    print("\n[+] Ingesting Full Historical TxSON Dataset (2015 to Present)...")
    df_raw = pd.read_csv('datasets/TxSON_SMAP_Weather_Merged.csv')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_clean = df_raw.dropna(subset=['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']).sort_values('Date')
    
    print(f"    -> Historical Training Range: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')} ({len(df_clean)} rows)")
    
    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    data_vals = df_clean[features].values
    
    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(data_vals)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(data_vals[:, 3].reshape(-1, 1))
    
    look_back = 14
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i, 3])
        
    X = np.array(X)
    y = np.array(y)
    
    print("\n[+] Training LSTM Engine on 100% of Available TxSON Ground Data...")
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(look_back, 4)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    engine = SoilMoistureForecaster(model, scaler_X, scaler_y, look_back=look_back)
    
    # 2. Build 79 Spatial Coordinates (TxSON Anchors + Fredericksburg & Tamaulipas Continuous Grids)
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
            
    # 3. Parallelized Live OpenMeteo Microclimate Data Acquisition
    print(f"\n[+] Multithreaded Live OpenMeteo Data Acquisition for {len(all_lats)} Coordinates...")
    recent_batch = fetch_openmeteo_parallel(all_lats, all_lons)
    seed_batch = np.array(recent_batch) # Shape: (79, 14, 4)
    
    # 4. Synthesize Future Meteorological Forcing for Next 364 Days (Seasonal Climatology)
    print("\n[+] Synthesizing 1-Year Climatological Forcing Profile (Precip + Thermal Cycle)...")
    horizon = 364
    future_forcing_batch = np.zeros((len(all_lats), horizon, 3))
    
    days_in_year = np.arange(horizon)
    # Seasonal temperature cycle based on annual solar variation
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
                'Predicted_SMAP': max(0.05, min(0.50, future_predictions[i][day]))
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
