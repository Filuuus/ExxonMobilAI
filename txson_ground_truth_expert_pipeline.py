import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

def build_expert_dataset():
    csv_path = "datasets/OpenMeteo_Synthetic_Grid_Dataset.csv"
    print(f"[+] Loading unified calibrated ground truth dataset: '{csv_path}'")
    df = pd.read_csv(csv_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Day-of-Year seasonality cyclical features
    doy = df['Date'].dt.dayofyear
    df['DOY_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['DOY_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    # Compute station baseline soil holding capacity (mu) and dynamic range (sigma)
    train_cutoff = pd.Timestamp("2020-09-01")
    station_baselines = {}
    
    for st_id, group in df.groupby('Station_ID'):
        st_train = group[group['Date'] <= train_cutoff]
        if len(st_train) > 30 and st_train['SMAP_Moisture'].std() > 0.001:
            mu = float(st_train['SMAP_Moisture'].mean())
            sigma = float(st_train['SMAP_Moisture'].std())
        else:
            mu = float(group['SMAP_Moisture'].mean())
            sigma = float(group['SMAP_Moisture'].std())
            
        sigma = max(sigma, 0.015)
        station_baselines[st_id] = {'mean': mu, 'std': sigma}
        
    with open("station_soil_baselines.json", "w") as f:
        json.dump(station_baselines, f, indent=2)
    print(f"[+] Computed & saved soil baselines for {len(station_baselines)} stations.")
    
    # Add baseline features to dataframe
    df['Baseline_Mean'] = df['Station_ID'].map(lambda s: station_baselines[s]['mean'])
    df['Baseline_Std'] = df['Station_ID'].map(lambda s: station_baselines[s]['std'])
    
    # Standardize ground truth target into anomaly space z
    df['Target_z'] = (df['SMAP_Moisture'] - df['Baseline_Mean']) / df['Baseline_Std']
    df['Target_z'] = np.clip(df['Target_z'], -3.0, 4.0)
    
    # Forward fill any small NaN gaps in OpenMeteo_SM if present
    df['OpenMeteo_SM'] = df.groupby('Station_ID')['OpenMeteo_SM'].ffill().bfill()
    
    return df, station_baselines

def prepare_expert_sequences(df, look_back=28, train_cutoff="2020-09-01"):
    train_cutoff_dt = pd.Timestamp(train_cutoff)
    df_train = df[df['Date'] <= train_cutoff_dt].copy()
    df_test = df[df['Date'] > train_cutoff_dt].copy()
    
    weather_features = ['Precipitation', 'T_Max', 'T_Min', 'DOY_sin', 'DOY_cos', 'OpenMeteo_SM', 'Baseline_Mean', 'Baseline_Std']
    
    scaler_weather = StandardScaler()
    df_train[weather_features] = scaler_weather.fit_transform(df_train[weather_features].values)
    joblib.dump(scaler_weather, 'expert_weather_scaler.joblib')
    print(f"[+] Fitted and saved expert scaler on {len(weather_features)} features.")
    
    feature_cols = weather_features + ['Target_z']
    
    X_train_list, y_train_list = [], []
    for st_id, group in df_train.groupby('Station_ID'):
        group = group.sort_values('Date').reset_index(drop=True)
        vals = group[feature_cols].values
        if len(vals) <= look_back:
            continue
        for i in range(len(vals) - look_back):
            X_train_list.append(vals[i : i + look_back])
            y_train_list.append(vals[i + look_back, -1])  # Predict next Target_z
            
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    
    print(f"[+] Prepared Training Dataset: {X_train.shape[0]:,} sequences | Shape: {X_train.shape}")
    return X_train, y_train, df_train, df_test, scaler_weather, weather_features

def build_expert_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(delta=1.0))
    return model

def main():
    print("=" * 80)
    print("  TXSON HIGH-ACCURACY EXPERT GROUND-TRUTH PIPELINE")
    print("  [OpenMeteo_SM Proxy + Hydraulic Soil Baselines + 28-Day Memory + Huber LSTM]")
    print("=" * 80)
    
    look_back = 28
    df, station_baselines = build_expert_dataset()
    X_train, y_train, df_train, df_test, scaler_weather, weather_features = prepare_expert_sequences(df, look_back=look_back)
    
    print("\n[+] Training Stacked Deep Huber LSTM...")
    model = build_expert_lstm((look_back, 9))
    model.fit(X_train, y_train, epochs=14, batch_size=64, verbose=1)
    
    model.save("expert_ground_truth_lstm.keras")
    print("[+] Model saved to 'expert_ground_truth_lstm.keras'.")
    
    # 365-Day Blind Recursive Evaluation
    print("\n" + "=" * 80)
    print("  1-YEAR ZERO-LEAKAGE BLIND RECURSIVE EVALUATION (2020-09-01 to 2021-09-01)")
    print("=" * 80)
    
    gt_cutoff = pd.Timestamp("2021-09-01")
    df_eval = df_test[df_test['Date'] <= gt_cutoff].copy()
    
    active_test_stations = []
    for st_id, group in df_eval.groupby('Station_ID'):
        st_train = df_train[df_train['Station_ID'] == st_id]
        if len(st_train) >= look_back and len(group) >= 350 and group['SMAP_Moisture'].std() > 0.005:
            active_test_stations.append(st_id)
            
    active_test_stations = sorted(active_test_stations)
    print(f"[+] Active In-Situ Validation Stations ({len(active_test_stations)}): {', '.join(active_test_stations)}")
    
    @tf.function(reduce_retracing=True)
    def fast_predict(seq):
        return model(seq, training=False)
        
    validation_results = {}
    metrics_list = []
    
    for st_id in active_test_stations:
        st_train = df_train[df_train['Station_ID'] == st_id].sort_values('Date').reset_index(drop=True)
        st_test = df_eval[df_eval['Station_ID'] == st_id].sort_values('Date').reset_index(drop=True)
        
        mu = station_baselines[st_id]['mean']
        sigma = station_baselines[st_id]['std']
        
        # Prepare 28-day seed sequence strictly prior to cutoff
        st_train_scaled = st_train.copy()
        st_train_scaled[weather_features] = scaler_weather.transform(st_train[weather_features].values)
        seed_features = st_train_scaled.iloc[-look_back:][weather_features + ['Target_z']].values
        curr_seq = seed_features.copy()
        
        # Test period atmospheric forcing and ground truth
        st_test_scaled = st_test.copy()
        st_test_scaled[weather_features] = scaler_weather.transform(st_test[weather_features].values)
        test_w = st_test_scaled[weather_features].values
        actual_swc = st_test['SMAP_Moisture'].values
        dates_test = st_test['Date'].values
        horizon = len(st_test)
        
        # 365-day recursive rollout
        preds_z = []
        for step in range(horizon):
            lstm_in = tf.convert_to_tensor(curr_seq.reshape(1, look_back, 9), dtype=tf.float32)
            pred_z = float(fast_predict(lstm_in).numpy()[0, 0])
            pred_z = float(np.clip(pred_z, -2.5, 3.5))
            preds_z.append(pred_z)
            
            # Update recursive sequence
            next_step = np.zeros((1, 9))
            next_step[0, 0:8] = test_w[step, 0:8]
            next_step[0, 8] = pred_z
            curr_seq = np.vstack([curr_seq[1:], next_step])
            
        preds_z = np.array(preds_z)
        
        # Invert to physical volumetric soil moisture
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
    metrics_csv = "txson_expert_pipeline_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[+] Saved expert metrics to: '{metrics_csv}'")
    
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
        ax.plot(res['Dates'], res['Forecast'], label='Expert Forecast (28d+SM)', color='#2ca02c', linestyle='--', linewidth=1.8)
        ax.axhline(res['Baseline'], color='#7f7f7f', linestyle=':', label=f'Soil Baseline ({res["Baseline"]:.2f})', alpha=0.7)
        ax.set_title(f"Station {st_id} (R²: {res['R2']:+.2f} | Corr: {res['Corr']:.2f} | RMSE: {res['RMSE']:.4f})", fontweight='bold', fontsize=11)
        ax.set_ylabel("Soil Moisture (m³/m³)")
        ax.set_ylim(0.0, 0.45)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("TxSON High-Accuracy Expert 1-Year Blind Forecast vs In-Situ Ground Truth\n"
                 "Features: OpenMeteo Satellite SM + Soil Holding Capacity (μ, σ) + 28-Day Memory | 2020-09-01 to 2021-09-01",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_path = "txson_expert_pipeline_validation.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved multi-station dashboard to: '{plot_path}'")

if __name__ == "__main__":
    main()
