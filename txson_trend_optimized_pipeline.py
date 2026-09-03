import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configure 5950X multi-threading
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def run_baseline_normalized_pipeline():
    print("=" * 80)
    print("  TxSON PER-STATION TREND-OPTIMIZED RECURSIVE LSTM PIPELINE")
    print("  Resolving Underfitting with Seasonality & Huber Loss")
    print("=" * 80)
    
    # 1. Ingest massive synthetic OpenMeteo grid dataset
    dataset_path = 'datasets/OpenMeteo_Synthetic_Grid_Dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing required dataset: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add Day of Year Seasonality
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DOY_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DOY_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    
    train_cutoff = pd.Timestamp('2020-09-01')
    gt_cutoff = pd.Timestamp('2021-09-01')
    look_back = 14
    
    df_train = df[df['Date'] <= train_cutoff].copy()
    df_test = df[(df['Date'] > train_cutoff) & (df['Date'] <= gt_cutoff)].copy()
    
    print(f"[+] Historical Training Span: {df_train['Date'].min().strftime('%Y-%m-%d')} to {df_train['Date'].max().strftime('%Y-%m-%d')} ({len(df_train):,} records)")
    print(f"[+] 1-Year Blind Evaluation Span: 2020-09-02 to 2021-09-01 ({len(df_test):,} records)")
    
    # 2. Compute per-station soil baseline statistics STRICTLY from historical training data
    print("\n[+] Computing In-Situ Soil Hydraulic Baselines (Mean, Std) per Station...")
    station_baselines = {}
    for st_id, group in df_train.groupby('Station_ID'):
        mu = float(group['SMAP_Moisture'].mean())
        sigma = float(group['SMAP_Moisture'].std())
        sigma = max(sigma, 0.015) # Numerical stability floor
        station_baselines[st_id] = {'mean': round(mu, 5), 'std': round(sigma, 5)}
        
    with open('station_soil_baselines.json', 'w') as f:
        json.dump(station_baselines, f, indent=2)
    print(f"    -> Saved baselines for {len(station_baselines)} stations to 'station_soil_baselines.json'")
    
    # 3. Standardize meteorological forcing and soil moisture
    # Weather features use global min-max scaling
    weather_features = ['Precipitation', 'T_Max', 'T_Min', 'DOY_sin', 'DOY_cos']
    scaler_weather = MinMaxScaler()
    scaler_weather.fit(df_train[weather_features].values)

    joblib.dump(scaler_weather, 'trend_optimized_weather_scaler.joblib')
    print("    -> Saved fitted weather scaler to 'trend_optimized_weather_scaler.joblib'.")
    
    # Transform training data to dynamic anomalies (z-scores)
    df_train['SWC_z'] = 0.0
    for st_id in station_baselines:
        mask = df_train['Station_ID'] == st_id
        mu = station_baselines[st_id]['mean']
        sigma = station_baselines[st_id]['std']
        df_train.loc[mask, 'SWC_z'] = (df_train.loc[mask, 'SMAP_Moisture'] - mu) / sigma
        
    # Clip extreme outlier anomalies in training
    df_train['SWC_z'] = df_train['SWC_z'].clip(-3.0, 3.5)
    
    # 4. Construct training sequences
    X_train, y_train = [], []
    for st_id, group in df_train.groupby('Station_ID'):
        if len(group) <= look_back:
            continue
        g_w = scaler_weather.transform(group[weather_features].values)
        g_z = group['SWC_z'].values.reshape(-1, 1)
        g_combined = np.hstack([g_w, g_z])
        
        for i in range(look_back, len(g_combined)):
            X_train.append(g_combined[i-look_back:i])
            y_train.append(g_combined[i, 5])
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    print(f"[+] Constructed {len(X_train):,} training sequences of length {look_back} (Standardized Dynamics).")
    
    # 5. Train Shared Anomaly LSTM Architecture
    print(f"\n[+] Training Shared Hydrological Dynamic LSTM on AMD Ryzen 9 5950X...")
    model = Sequential([
        LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=(look_back, 6)),
        LSTM(32, activation='tanh', recurrent_activation='sigmoid'),
        Dense(32, activation='relu'),
        Dense(1) # Linear output for continuous standardized anomaly z
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(delta=1.0))
    model.fit(X_train, y_train, epochs=12, batch_size=64, verbose=1)
    print("    -> Anomaly LSTM Model Training Complete.")
    
    model.save("trend_optimized_lstm_model.keras")
    print("[+] Saved model to 'trend_optimized_lstm_model.keras'.")
    
    # 6. Execute 1-Year Blind Validation Across Active Stations
    print("\n" + "=" * 80)
    print("  1-YEAR ZERO-LEAKAGE BLIND RECURSIVE EVALUATION (2020-09-01 to 2021-09-01)")
    print("=" * 80)
    
    active_test_stations = []
    for st_id, group in df_test.groupby('Station_ID'):
        st_train = df_train[df_train['Station_ID'] == st_id]
        # Must have seed data, >= 350 test days, and non-trivial sensor variance (exclude offline sensors)
        if len(st_train) >= look_back and len(group) >= 350 and group['SMAP_Moisture'].std() > 0.005:
            active_test_stations.append(st_id)
            
    active_test_stations = sorted(active_test_stations)
    print(f"[+] Active In-Situ Validation Stations ({len(active_test_stations)}): {', '.join(active_test_stations)}")
    
    validation_results = {}
    metrics_list = []
    
    @tf.function(reduce_retracing=True)
    def fast_predict(seq):
        return model(seq, training=False)

    for st_id in active_test_stations:
        st_train = df_train[df_train['Station_ID'] == st_id].sort_values('Date')
        st_test = df_test[df_test['Station_ID'] == st_id].sort_values('Date')
        
        mu = station_baselines[st_id]['mean']
        sigma = station_baselines[st_id]['std']
        
        # Prepare 14-day seed sequence strictly prior to cutoff
        seed_w = scaler_weather.transform(st_train.iloc[-look_back:][weather_features].values)
        seed_z = ((st_train.iloc[-look_back:]['SMAP_Moisture'].values - mu) / sigma).reshape(-1, 1)
        curr_seq = np.hstack([seed_w, seed_z])
        
        # Test period forcing and ground truth
        test_w = scaler_weather.transform(st_test[weather_features].values)
        precip_raw = st_test['Precipitation'].values
        actual_swc = st_test['SMAP_Moisture'].values
        dates_test = st_test['Date'].values
        horizon = len(st_test)
        
        # 365-day recursive rollout in standardized space
        preds_z = []
        for step in range(horizon):
            lstm_in = tf.convert_to_tensor(curr_seq.reshape(1, look_back, 6), dtype=tf.float32)
            pred_z = float(fast_predict(lstm_in).numpy()[0, 0])
            pred_z = float(np.clip(pred_z, -2.5, 3.5))
            preds_z.append(pred_z)
            
            # Update recursive sequence
            next_step = np.zeros((1, 6))
            next_step[0, 0:5] = test_w[step, 0:5]
            next_step[0, 5] = pred_z
            curr_seq = np.vstack([curr_seq[1:], next_step])
            
        preds_z = np.array(preds_z)
        
        # INVERT BACK TO TRUE PHYSICAL VOLUMETRIC SOIL MOISTURE (m3/m3)
        preds_physical = (preds_z * sigma) + mu
        preds_physical = np.clip(preds_physical, 0.02, 0.50)
        
        rmse = np.sqrt(mean_squared_error(actual_swc, preds_physical))
        mae = mean_absolute_error(actual_swc, preds_physical)
        r2 = r2_score(actual_swc, preds_physical)
        corr = np.corrcoef(actual_swc, preds_physical)[0, 1]
        
        validation_results[st_id] = {
            'Dates': dates_test,
            'Actual': actual_swc,
            'Forecast': preds_physical,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Corr': corr,
            'Baseline': mu
        }
        metrics_list.append({
            'Station_ID': st_id,
            'Baseline_Mean': mu,
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Corr': round(corr, 4),
            'Days': horizon
        })
        print(f"    -> Station {st_id:6s} (Base={mu:.3f}): RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:+.4f}, Corr = {corr:.4f}")
        
    df_metrics = pd.DataFrame(metrics_list).sort_values('Station_ID')
    metrics_csv = "txson_trend_optimized_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[+] Saved trend-optimized metrics to: '{metrics_csv}'")
    
    # 7. Generate High-Resolution Multi-Station Comparison Dashboard
    n_plot = min(len(validation_results), 12)
    n_cols = 3
    n_rows = (n_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=True)
    axes = axes.flatten() if n_plot > 1 else [axes]
    
    plot_stations = sorted(validation_results.keys())[:n_plot]
    for idx, st_id in enumerate(plot_stations):
        ax = axes[idx]
        res = validation_results[st_id]
        ax.plot(res['Dates'], res['Actual'], label='In-Situ Ground Truth', color='#1f77b4', linewidth=1.8)
        ax.plot(res['Dates'], res['Forecast'], label='Trend-Optimized Forecast', color='#2ca02c', linestyle='--', linewidth=1.8)
        ax.axhline(res['Baseline'], color='#7f7f7f', linestyle=':', label=f'Soil Baseline ({res["Baseline"]:.2f})', alpha=0.7)
        ax.set_title(f"Station {st_id} (R²: {res['R2']:+.2f} | Corr: {res['Corr']:.2f} | RMSE: {res['RMSE']:.4f})", fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax.set_ylim(0.0, 0.45)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Date", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Soil Moisture (m³/m³)", fontsize=10)
            
    for j in range(len(plot_stations), len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(
        f"TxSON Per-Station Trend-Optimized 1-Year Blind Forecast vs In-Situ Ground Truth\n"
        f"Evaluation Period: 2020-09-01 to 2021-09-01 (Continuous 365-Day Zero-Leakage Rollout)",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_out = "txson_trend_optimized_validation.png"
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved multi-station dashboard to: '{os.path.abspath(plot_out)}'")
    print("=" * 80)
    return df_metrics

if __name__ == "__main__":
    run_baseline_normalized_pipeline()
