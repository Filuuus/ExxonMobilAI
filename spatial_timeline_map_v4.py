import os
import json
import glob
from datetime import datetime, timedelta
import concurrent.futures
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from openmeteo_cache import fetch_single_chunk_cached
from openmeteo_historical_miner import get_map_grid_coordinates
from openmeteo_cbfd_miner import parse_station_coordinates

MODEL_PATH = "trend_optimized_lstm_model.keras"
BASELINES_PATH = "station_soil_baselines.json"
SCALER_PATH = "trend_optimized_weather_scaler.joblib"
DATASET_PATH = "datasets/OpenMeteo_Synthetic_Grid_Dataset.csv"
WEATHER_FEATURES = ['Precipitation', 'T_Max', 'T_Min', 'DOY_sin', 'DOY_cos']
LOOK_BACK = 14


def build_known_coordinate_table():
    df_grid = get_map_grid_coordinates()
    df_grid['ID'] = df_grid['ID'].str.replace(r'^GRID_TX_A_(TXS\d+)$', r'\1', regex=True)

    cbfd_coords = parse_station_coordinates()
    df_cbfd = pd.DataFrame(
        [{'ID': sid, 'Lat': lat, 'Lon': lon} for sid, (lat, lon) in cbfd_coords.items()]
    )
    return pd.concat([df_grid, df_cbfd], ignore_index=True)


def interpolate_baseline(lat, lon, coord_table, station_baselines, power=2, k=6, eps=1e-4):
    known = coord_table[coord_table['ID'].isin(station_baselines.keys())]
    d = np.sqrt((known['Lat'].values - lat) ** 2 + (known['Lon'].values - lon) ** 2)

    if d.min() < eps:
        sid = known['ID'].values[np.argmin(d)]
        return station_baselines[sid]['mean'], station_baselines[sid]['std']

    k = min(k, len(d))
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] ** power)
    w /= w.sum()
    ids = known['ID'].values[idx]
    means = np.array([station_baselines[sid]['mean'] for sid in ids])
    stds = np.array([station_baselines[sid]['std'] for sid in ids])
    return float((w * means).sum()), float((w * stds).sum())


def build_doy_climatology(dataset_path=DATASET_PATH):
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    clim = df.groupby('DayOfYear')[['T_Max', 'T_Min']].mean()
    if 366 not in clim.index:
        clim.loc[366] = clim.loc[365]
    return clim.sort_index()


def fetch_live_seed_batch(lats, lons, chunk_size=20):
    lat_chunks = [lats[i:i + chunk_size] for i in range(0, len(lats), chunk_size)]
    lon_chunks = [lons[i:i + chunk_size] for i in range(0, len(lons), chunk_size)]

    all_results = [None] * len(lat_chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(lat_chunks), 16)) as executor:
        future_to_idx = {
            executor.submit(fetch_single_chunk_cached, lat_c, lon_c): idx
            for idx, (lat_c, lon_c) in enumerate(zip(lat_chunks, lon_chunks))
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk_data, _ = future.result()
            all_results[idx] = chunk_data

    flat = []
    for chunk in all_results:
        flat.extend(chunk)
    return flat


def generate_spatial_timeline_v4(output_file='capstone_spatial_forecast_v4.html', horizon=364):
    print("=" * 80)
    print("  CAPSTONE PROPOSAL: 1-YEAR FORWARD SPATIAL FORECAST (v4)")
    print("  High-Contrast Dynamic Colormap + IDW Spatial Baseline + All Coordinates")
    print("=" * 80)

    for path in (MODEL_PATH, BASELINES_PATH, SCALER_PATH, DATASET_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}.")

    print(f"\n[+] Loading trend-optimized LSTM, per-station baselines, and weather scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(BASELINES_PATH) as f:
        station_baselines = json.load(f)
    scaler_weather = joblib.load(SCALER_PATH)
    print(f"    -> {len(station_baselines)} known station baselines loaded.")

    coord_table = build_known_coordinate_table()

    # 1. Coordinate roster: 79 Texas/Tamaulipas grid + 19 CB/FD ground stations
    df_grid = get_map_grid_coordinates()
    cbfd_coords = parse_station_coordinates()
    cbfd_active = {
        sid: cbfd_coords[sid] for sid in cbfd_coords
        if sid in station_baselines
    }

    all_lats = list(df_grid['Lat'].values) + [c[0] for c in cbfd_active.values()]
    all_lons = list(df_grid['Lon'].values) + [c[1] for c in cbfd_active.values()]
    all_labels = list(df_grid['ID'].values) + list(cbfd_active.keys())
    N = len(all_lats)
    print(f"\n[+] Total Active Query Points on Map: {N} coordinates.")

    # 2. Interpolate baselines per query point
    print(f"\n[+] Computing IDW Soil Baselines for {N} coordinates...")
    baselines_interp = []
    for lat, lon in zip(all_lats, all_lons):
        mu, sigma = interpolate_baseline(lat, lon, coord_table, station_baselines, k=6, power=2)
        baselines_interp.append((mu, sigma))

    # 3. Acquire live 14-day OpenMeteo seed
    print(f"\n[+] Acquiring Live 14-Day OpenMeteo Seed for {N} Coordinates...")
    seed_raw = fetch_live_seed_batch(all_lats, all_lons)

    # 4. Build Climatology & Prepare Seed Tensor
    today = datetime.now().date()
    doy_climatology = build_doy_climatology()

    seed_dates = [today - timedelta(days=LOOK_BACK - d) for d in range(LOOK_BACK)]
    seed_doy = np.array([d.timetuple().tm_yday for d in seed_dates])
    seed_doy_sin = np.sin(2 * np.pi * seed_doy / 365.25)
    seed_doy_cos = np.cos(2 * np.pi * seed_doy / 365.25)

    curr_seq = np.zeros((N, LOOK_BACK, 6), dtype=np.float32)
    for i in range(N):
        precip = seed_raw[i][:, 0]
        tmax = seed_raw[i][:, 1]
        tmin = seed_raw[i][:, 2]
        sm = seed_raw[i][:, 3]
        mu, sigma = baselines_interp[i]
        weather_raw = np.column_stack([precip, tmax, tmin, seed_doy_sin, seed_doy_cos])
        weather_scaled = scaler_weather.transform(weather_raw)
        z = np.clip((sm - mu) / sigma, -3.0, 3.5)
        curr_seq[i] = np.hstack([weather_scaled, z.reshape(-1, 1)])

    # 5. Vectorized 364-Day Recursive Forward Rollout
    print(f"\n[+] Executing Vectorized {horizon}-Day Forward Rollout From {today.strftime('%Y-%m-%d')}...")
    future_dates = [today + timedelta(days=d) for d in range(horizon)]
    future_doy = np.array([d.timetuple().tm_yday for d in future_dates])
    future_doy = np.clip(future_doy, 1, 366)
    future_doy_sin = np.sin(2 * np.pi * future_doy / 365.25)
    future_doy_cos = np.cos(2 * np.pi * future_doy / 365.25)
    future_tmax = doy_climatology.loc[future_doy, 'T_Max'].values
    future_tmin = doy_climatology.loc[future_doy, 'T_Min'].values

    predictions_z = np.zeros((N, horizon), dtype=np.float32)
    for step in range(horizon):
        preds = model(curr_seq, training=False).numpy().flatten()
        preds = np.clip(preds, -2.5, 3.5)
        predictions_z[:, step] = preds

        next_weather_raw = np.column_stack([
            np.zeros(N),  # Climatological baseline drydown
            np.full(N, future_tmax[step]),
            np.full(N, future_tmin[step]),
            np.full(N, future_doy_sin[step]),
            np.full(N, future_doy_cos[step]),
        ])
        next_weather_scaled = scaler_weather.transform(next_weather_raw)
        next_step = np.hstack([next_weather_scaled, preds.reshape(-1, 1)]).astype(np.float32)
        curr_seq = np.concatenate([curr_seq[:, 1:, :], next_step[:, np.newaxis, :]], axis=1)

    # 6. Invert z -> physical volumetric soil moisture
    mus = np.array([b[0] for b in baselines_interp])
    sigmas = np.array([b[1] for b in baselines_interp])
    predictions_physical = predictions_z * sigmas[:, None] + mus[:, None]
    predictions_physical = np.clip(predictions_physical, 0.02, 0.50)

    # 7. Format weekly-scrubber timeline records
    records = []
    for day in range(0, horizon, 7):
        date_str = future_dates[day].strftime('%Y-%m-%d')
        for i, (lat, lon, label) in enumerate(zip(all_lats, all_lons, all_labels)):
            records.append({
                'Date': date_str,
                'Location': label,
                'Latitude': lat,
                'Longitude': lon,
                'Predicted_SMAP': float(predictions_physical[i, day]),
            })

    df_forecast = pd.DataFrame(records)
    
    # 8. High-Contrast Dynamic Colormap (Percentile Tuning)
    p5 = float(np.percentile(df_forecast['Predicted_SMAP'], 5))
    p95 = float(np.percentile(df_forecast['Predicted_SMAP'], 95))
    c_min = max(0.04, round(p5, 3))
    c_max = min(0.38, round(p95, 3))
    print(f"\n[+] High-Contrast Color Dynamic Range: [{c_min:.4f}, {c_max:.4f}] m³/m³")

    print("\n[+] Rendering Interactive Spatial Timeline Heatmap (Turbo Enhanced)...")
    fig = px.density_map(
        df_forecast, lat="Latitude", lon="Longitude", z="Predicted_SMAP",
        hover_name="Location",
        hover_data={"Date": False, "Predicted_SMAP": ':.4f', "Latitude": False, "Longitude": False},
        animation_frame="Date", animation_group="Location",
        color_continuous_scale="Turbo",  # Rich, vibrant perceptual colormap with maximum contrast
        range_color=[c_min, c_max],
        radius=90,
        zoom=5.5,
        center={"lat": 27.85, "lon": -98.4},
        title=f"ExxonMobil AI: 1-Year Forward Regional Soil Moisture Forecast (From {today.strftime('%Y-%m-%d')})",
        labels={'Predicted_SMAP': 'Volumetric Soil Moisture (m³/m³)'},
    )
    fig.update_layout(
        map_style="carto-positron",
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Soil Moisture<br>(m³/m³)",
            thicknessmode="pixels", thickness=22,
            lenmode="pixels", len=320,
            yanchor="top", y=1,
            ticks="outside", tickformat=".2f"
        ),
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 550
    fig.write_html(output_file)
    print(f"\n[+] Success! Forward Spatial Forecast Heatmap Saved To:\n    {os.path.abspath(output_file)}")
    
    # Also overwrite v3 so both URLs point to the high-contrast dashboard
    fig.write_html("capstone_spatial_forecast_v3.html")
    print("=" * 80)


if __name__ == "__main__":
    generate_spatial_timeline_v4()
