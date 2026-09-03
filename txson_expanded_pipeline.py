import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configure 5950X multi-threading
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from forecaster_engine_v2 import SoilMoistureForecaster

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def process_station_file_to_daily(file_path):
    """
    Loads an hourly station CSV from datasets/cleanup/CSV and aggregates it to daily standard format:
    [Station_ID, Date, Precipitation, T_Max, T_Min, SMAP_Moisture]
    """
    station_id = Path(file_path).stem.replace("_filled_Data", "")
    try:
        df = pd.read_csv(file_path)
        time_col = df.columns[0]
        df['Date'] = pd.to_datetime(df[time_col], errors='coerce').dt.normalize()
        df = df.dropna(subset=['Date'])
        
        # Check SWC_5
        if 'SWC_5' not in df.columns:
            return None
            
        # Physical soil moisture validation: skip uncalibrated raw channels (> 1.0)
        mean_swc = df['SWC_5'].dropna().mean()
        if pd.isna(mean_swc) or mean_swc > 0.60 or mean_swc < 0.01:
            # Station has uncalibrated dielectric readings (e.g. FD08, FD16, FD22, FD24)
            return None
            
        # Aggregations
        # Temperature: prefer Tair, fallback to T_5
        if 'Tair' in df.columns and df['Tair'].notna().sum() > 100:
            temp_col = 'Tair'
        elif 'T_5' in df.columns:
            temp_col = 'T_5'
        else:
            return None
            
        ppt_col = 'Ppt' if 'Ppt' in df.columns else None
        
        agg_map = {
            'SWC_5': 'mean',
            temp_col: ['max', 'min']
        }
        if ppt_col:
            agg_map[ppt_col] = 'sum'
            
        daily = df.groupby('Date').agg(agg_map)
        
        # Flatten column names
        daily.columns = ['_'.join(c).strip('_') for c in daily.columns]
        
        t_max_col = f"{temp_col}_max"
        t_min_col = f"{temp_col}_min"
        ppt_final_col = f"{ppt_col}_sum" if ppt_col else None
        
        daily_clean = pd.DataFrame({
            'Station_ID': station_id,
            'Date': daily.index,
            'Precipitation': daily[ppt_final_col].fillna(0.0) if ppt_final_col else 0.0,
            'T_Max': daily[t_max_col].interpolate(limit=7).bfill().ffill(),
            'T_Min': daily[t_min_col].interpolate(limit=7).bfill().ffill(),
            'SMAP_Moisture': daily['SWC_5_mean'].clip(0.02, 0.50).interpolate(limit=7).bfill().ffill()
        }).dropna()
        
        return daily_clean
    except Exception as e:
        print(f"[-] Error processing {station_id}: {e}")
        return None

def build_expanded_dataset(clean_dir="datasets/final_clean/CSV"):
    """
    Ingests and merges all valid cleaned TxSON station records into a unified high-throughput dataset.
    """
    print("=" * 75)
    print("  BUILDING EXPANDED TXSON TRAINING DATASET FROM CLEANED STATIONS")
    print("=" * 75)
    
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*_filled_Data.csv")))
    print(f"[+] Found {len(clean_files)} cleaned station CSV files in '{clean_dir}'.")
    
    all_station_dfs = []
    
    # 1. Add new cleaned stations
    valid_stations = []
    for f in clean_files:
        df_st = process_station_file_to_daily(f)
        if df_st is not None and len(df_st) > 100:
            st_name = df_st['Station_ID'].iloc[0]
            valid_stations.append(st_name)
            all_station_dfs.append(df_st)
            
    print(f"[+] Successfully validated {len(valid_stations)} cleaned stations with physical SWC_5:")
    print(f"    {', '.join(valid_stations)}")
    
    # 2. Also incorporate original TxSON historical dataset (TXS01-TXS06)
    orig_path = "datasets/TxSON_SMAP_Weather_Merged.csv"
    if os.path.exists(orig_path):
        orig_df = pd.read_csv(orig_path)
        orig_df['Date'] = pd.to_datetime(orig_df['Date'])
        orig_clean = orig_df[['Station_ID', 'Date', 'Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']].dropna()
        all_station_dfs.append(orig_clean)
        print(f"[+] Included original 6 TxSON stations (TXS01-TXS06): {len(orig_clean)} rows.")
        
    unified_df = pd.concat(all_station_dfs, ignore_index=True).sort_values(['Station_ID', 'Date']).reset_index(drop=True)
    
    print(f"\n[+] Total Unified Dataset Records: {len(unified_df):,} daily observations")
    print(f"    - Date Span: {unified_df['Date'].min().strftime('%Y-%m-%d')} to {unified_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"    - Active Stations ({unified_df['Station_ID'].nunique()}): {', '.join(sorted(unified_df['Station_ID'].unique()))}")
    
    unified_csv = "datasets/TxSON_Expanded_Unified_Dataset.csv"
    unified_df.to_csv(unified_csv, index=False)
    print(f"[+] Saved unified dataset to: '{unified_csv}'")
    return unified_df

def run_expanded_blind_validation():
    """
    Trains the shared LSTM on the massive expanded dataset up to 2020-09-01,
    and runs zero-leakage blind 1-year forecasting across 2020-2021.
    """
    unified_df = build_expanded_dataset()
    
    gt_cutoff = pd.Timestamp('2021-09-01')
    train_cutoff = pd.Timestamp('2020-09-01')
    look_back = 14
    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    
    # Training split: strictly historical data <= 2020-09-01
    df_train_all = unified_df[unified_df['Date'] <= train_cutoff].dropna(subset=features)
    
    print("\n" + "=" * 75)
    print("  ZERO-LEAKAGE EXPANDED LSTM TRAINING (Historical up to 2020-09-01)")
    print("=" * 75)
    print(f"[+] Training Set Size: {len(df_train_all):,} daily records across {df_train_all['Station_ID'].nunique()} stations")
    
    train_vals = df_train_all[features].values
    scaler_X = MinMaxScaler()
    train_scaled = scaler_X.fit_transform(train_vals)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_vals[:, 3].reshape(-1, 1))
    
    # Build training sequences per station to avoid boundary contamination
    X_train, y_train = [], []
    for st_id, group in df_train_all.groupby('Station_ID'):
        if len(group) <= look_back:
            continue
        g_scaled = scaler_X.transform(group[features].values)
        for i in range(look_back, len(g_scaled)):
            X_train.append(g_scaled[i-look_back:i])
            y_train.append(g_scaled[i, 3])
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    print(f"[+] Generated {len(X_train):,} training sequences of length {look_back}.")
    
    print(f"\n[+] Training Shared LSTM Architecture on AMD Ryzen 9 5950X (Bounded Activations)...")
    model = Sequential([
        LSTM(32, activation='tanh', recurrent_activation='sigmoid', input_shape=(look_back, 4)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    print("    -> Expanded LSTM Model Training Complete.")
    
    # Save the expanded model for map generation and downstream inference
    model.save("expanded_lstm_model.keras")
    print("[+] Saved trained expanded model to 'expanded_lstm_model.keras'.")
    
    engine = SoilMoistureForecaster(model, scaler_X, scaler_y, look_back=look_back)
    
    # 1-Year Blind Test Horizon: 2020-09-02 to 2021-09-01
    print("\n" + "=" * 75)
    print("  1-YEAR BLIND ACCURACY VALIDATION (2020-09-01 to 2021-09-01)")
    print("=" * 75)
    
    test_stations = []
    df_test_all = unified_df[(unified_df['Date'] > train_cutoff) & (unified_df['Date'] <= gt_cutoff)]
    
    for st_id, group in df_test_all.groupby('Station_ID'):
        st_train = df_train_all[df_train_all['Station_ID'] == st_id]
        # Must have seed data, >= 350 test days, and non-trivial sensor variance (exclude offline sensors like CB19, CB20)
        if len(st_train) >= look_back and len(group) >= 350 and group['SMAP_Moisture'].std() > 0.005:
            test_stations.append(st_id)
            
    test_stations = sorted(test_stations)
    print(f"[+] Active In-Situ Validation Stations with Live Sensors ({len(test_stations)}): {', '.join(test_stations)}")
    
    validation_results = {}
    metrics_list = []
    
    for st_id in test_stations:
        st_train = df_train_all[df_train_all['Station_ID'] == st_id].sort_values('Date')
        st_test = df_test_all[df_test_all['Station_ID'] == st_id].sort_values('Date')
        
        seed_sequence = st_train[features].values[-look_back:]
        meteorological_forcing = st_test[['Precipitation', 'T_Max', 'T_Min']].values
        actual_sm = st_test['SMAP_Moisture'].values
        dates_test = st_test['Date'].values
        
        horizon = len(st_test)
        
        # Recursive forecast with physical drydown constraint
        current_seq = scaler_X.transform(seed_sequence)
        blind_pred = []
        
        for step in range(horizon):
            lstm_in = current_seq.reshape(1, look_back, 4)
            pred_scaled = model(lstm_in, training=False).numpy()
            pred_actual = scaler_y.inverse_transform(pred_scaled)[0, 0]
            
            # Physics constraint: on non-rain days, soil cannot spontaneously gain moisture
            precip_today = meteorological_forcing[step, 0]
            if step > 0 and precip_today < 0.1:
                # Strictly cap by prior day + minimal noise margin
                pred_actual = min(pred_actual, blind_pred[-1] + 0.0005)
                
            pred_actual = float(np.clip(pred_actual, 0.02, 0.50))
            blind_pred.append(pred_actual)
            
            # Form next day features
            next_feat = np.zeros((1, 4))
            next_feat[0, 0:3] = meteorological_forcing[step, 0:3]
            next_feat[0, 3] = pred_actual
            next_scaled = scaler_X.transform(next_feat)
            current_seq = np.vstack([current_seq[1:], next_scaled])
            
        blind_pred = np.array(blind_pred)
        
        rmse = np.sqrt(mean_squared_error(actual_sm, blind_pred))
        mae = mean_absolute_error(actual_sm, blind_pred)
        r2 = r2_score(actual_sm, blind_pred)
        
        res = {
            'Station_ID': st_id,
            'Dates': dates_test,
            'Actual': actual_sm,
            'Forecast': blind_pred,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        validation_results[st_id] = res
        metrics_list.append({
            'Station_ID': st_id,
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Days': horizon
        })
        print(f"    -> Station {st_id:6s}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f} ({horizon} days)")
        
    df_metrics = pd.DataFrame(metrics_list).sort_values('Station_ID')
    metrics_out = "txson_expanded_validation_metrics.csv"
    df_metrics.to_csv(metrics_out, index=False)
    print(f"\n[+] Saved expanded validation metrics to: '{metrics_out}'")
    
    # Generate multi-subplot dashboard for top validation stations
    n_plot = min(len(validation_results), 12)
    n_cols = 3
    n_rows = (n_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten() if n_plot > 1 else [axes]
    
    plot_stations = sorted(validation_results.keys())[:n_plot]
    for idx, st_id in enumerate(plot_stations):
        ax = axes[idx]
        res = validation_results[st_id]
        ax.plot(res['Dates'], res['Actual'], label='In-Situ Ground Truth', color='#1f77b4', linewidth=1.8)
        ax.plot(res['Dates'], res['Forecast'], label='Blind LSTM Forecast', color='#d62728', linestyle='--', linewidth=1.8)
        ax.set_title(f"Station {st_id} (RMSE: {res['RMSE']:.4f} | R²: {res['R2']:.4f})", fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        ax.set_ylim(0.0, 0.45)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Date", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Soil Moisture (m³/m³)", fontsize=10)
            
    for j in range(len(plot_stations), len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(
        f"TxSON Expanded Network Demonstration: 1-Year Blind LSTM Forecast vs Ground Truth\n"
        f"Evaluation Period: 2020-09-01 to 2021-09-01 (Zero-Leakage Across {len(plot_stations)} Stations)",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_out = "txson_expanded_validation.png"
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved expanded multi-station dashboard to: '{os.path.abspath(plot_out)}'")
    print("=" * 75)
    return df_metrics

if __name__ == "__main__":
    run_expanded_blind_validation()
