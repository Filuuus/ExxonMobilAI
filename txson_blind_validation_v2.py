import os
import sys
import concurrent.futures
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Leverage AMD Ryzen 9 5950X (16 cores / 32 threads)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from forecaster_engine_v2 import SoilMoistureForecaster

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def evaluate_single_station(station_id, df_train, df_test, engine, features, look_back):
    """Evaluates a blind 1-year forecast for a specific station."""
    st_train = df_train[df_train['Station_ID'] == station_id].sort_values('Date')
    st_test = df_test[df_test['Station_ID'] == station_id].sort_values('Date')
    
    if len(st_train) < look_back or len(st_test) == 0:
        return None
        
    seed_sequence = st_train[features].values[-look_back:]
    meteorological_forcing = st_test[['Precipitation', 'T_Max', 'T_Min']].values
    actual_smap = st_test['SMAP_Moisture'].values
    dates_test = st_test['Date'].values
    
    horizon = len(st_test)
    blind_forecast = engine.predict_future(
        seed_sequence, 
        forecast_horizon=horizon, 
        future_meteorological_forcing=meteorological_forcing
    )
    
    rmse = np.sqrt(mean_squared_error(actual_smap, blind_forecast))
    mae = mean_absolute_error(actual_smap, blind_forecast)
    r2 = r2_score(actual_smap, blind_forecast)
    
    return {
        'Station_ID': station_id,
        'Dates': dates_test,
        'Actual': actual_smap,
        'Forecast': blind_forecast,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def run_parallel_blind_validation():
    print("=" * 75)
    print("  PARALLELIZED BLIND LSTM FORECAST VALIDATION FOR ALL TXSON STATIONS (v2)")
    print("  Optimized for AMD Ryzen 9 5950X (16C/32T) & NVIDIA GeForce RTX 3090")
    print("=" * 75)
    
    # 1. Load Dataset & Filter to Ground-Truth Cutoff (2021-09-01)
    dataset_path = 'datasets/TxSON_SMAP_Weather_Merged.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing required dataset: {dataset_path}")
        
    df_raw = pd.read_csv(dataset_path)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    gt_cutoff = pd.Timestamp('2021-09-01')
    df_filtered = df_raw[df_raw['Date'] <= gt_cutoff].dropna(
        subset=['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    ).sort_values('Date')
    
    # Strict Zero-Leakage Training Cutoff: 2020-09-01 (1 year prior)
    train_cutoff = pd.Timestamp('2020-09-01')
    
    df_train = df_filtered[df_filtered['Date'] <= train_cutoff]
    df_test = df_filtered[(df_filtered['Date'] > train_cutoff) & (df_filtered['Date'] <= gt_cutoff)]
    
    stations = sorted(df_filtered['Station_ID'].unique())
    print(f"\n[+] Zero-Leakage Dataset Partitioning:")
    print(f"    - Training Span:       {df_train['Date'].min().strftime('%Y-%m-%d')} to {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"    - Blind Test Horizon:  {(train_cutoff + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {gt_cutoff.strftime('%Y-%m-%d')}")
    print(f"    - Target Stations ({len(stations)}): {', '.join(stations)}")
    
    # 2. Scale Features & Build Sequences
    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    train_vals = df_train[features].values
    
    scaler_X = MinMaxScaler()
    train_scaled = scaler_X.fit_transform(train_vals)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_vals[:, 3].reshape(-1, 1))
    
    look_back = 14
    X_train, y_train = [], []
    for i in range(look_back, len(train_scaled)):
        X_train.append(train_scaled[i-look_back:i])
        y_train.append(train_scaled[i, 3])
        
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # 3. Train Base Shared LSTM Model
    print(f"\n[+] Training Shared LSTM Architecture (Look-back={look_back} days)...")
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(look_back, 4)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    print("    -> Base LSTM Model Training Complete.")
    
    engine = SoilMoistureForecaster(model, scaler_X, scaler_y, look_back=look_back)
    
    # 4. Multi-threaded Evaluation Across All TxSON Stations
    print(f"\n[+] Dispatching Parallel Workers Across {len(stations)} TxSON Stations...")
    station_results = {}
    metrics_summary = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(stations), 16)) as executor:
        future_to_station = {
            executor.submit(evaluate_single_station, st, df_train, df_test, engine, features, look_back): st 
            for st in stations
        }
        
        for future in concurrent.futures.as_completed(future_to_station):
            res = future.result()
            if res is not None:
                st_id = res['Station_ID']
                station_results[st_id] = res
                metrics_summary.append({
                    'Station_ID': st_id,
                    'RMSE': round(res['RMSE'], 4),
                    'MAE': round(res['MAE'], 4),
                    'R2': round(res['R2'], 4)
                })
                print(f"    -> Finished {st_id}: RMSE={res['RMSE']:.4f}, MAE={res['MAE']:.4f}, R²={res['R2']:.4f}")
                
    # Save Summary CSV
    df_metrics = pd.DataFrame(metrics_summary).sort_values('Station_ID')
    csv_filename = 'txson_all_stations_blind_validation_metrics.csv'
    df_metrics.to_csv(csv_filename, index=False)
    
    print("\n" + "=" * 55)
    print("  FINAL ACCURACY SUMMARY ACROSS ALL STATIONS")
    print("=" * 55)
    print(df_metrics.to_string(index=False))
    print("=" * 55)
    print(f"\n[+] Saved validation summary to '{csv_filename}'")
    
    # 5. Generate Multi-Subplot Comparison Dashboard
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    sorted_st_keys = sorted(station_results.keys())
    for idx, st_id in enumerate(sorted_st_keys):
        res = station_results[st_id]
        ax = axes[idx]
        
        ax.plot(res['Dates'], res['Actual'], label='Ground Truth (TxSON)', color='#1f77b4', linewidth=1.8)
        ax.plot(res['Dates'], res['Forecast'], label='Blind LSTM Forecast', color='#d62728', linestyle='--', linewidth=1.8)
        
        ax.set_title(f"Station {st_id} (RMSE: {res['RMSE']:.4f} | R²: {res['R2']:.4f})", fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        ax.set_ylim(0.0, 0.45)
        
        if idx >= 4:
            ax.set_xlabel("Date", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("Soil Moisture (m³/m³)", fontsize=10)
            
    fig.suptitle(
        "TxSON Capstone Demonstration: 1-Year Blind LSTM Forecast vs Ground Truth (All 6 Stations)\n"
        "Evaluation Period: 2020-09-01 to 2021-09-01 (Zero-Leakage)", 
        fontsize=14, 
        fontweight='bold', 
        y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = 'txson_all_stations_blind_validation.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Multi-Station Validation Dashboard Saved: {os.path.abspath(plot_filename)}")
    print("=" * 75)

if __name__ == "__main__":
    run_parallel_blind_validation()
