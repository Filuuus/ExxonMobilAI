import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def main():
    print("=" * 80)
    print(" PER-STATION INDEPENDENT DIAGNOSTIC EVALUATION (ZERO-LEAKAGE BASELINE)")
    print("=" * 80)

    df_smap = pd.read_csv("datasets/TxSON_SMAP_Weather_Merged.csv")
    df_smap['Date'] = pd.to_datetime(df_smap['Date'])

    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    look_back = 14
    test_start_date = '2020-01-01'

    results = []

    stations = ['TXS01', 'TXS02', 'TXS03', 'TXS04', 'TXS05', 'TXS06']

    for i, station in enumerate(stations, start=1):
        station_file = f"datasets/TxSON-Station-Files/Station{i}_filled_Data.csv"
        if not os.path.exists(station_file):
            print(f"Warning: File {station_file} not found. Skipping {station}.")
            continue

        # Load station ground truth
        df_target = pd.read_csv(station_file)
        df_target['Date'] = pd.to_datetime(df_target['Unnamed: 0']).dt.normalize()
        target_daily = df_target.groupby('Date')[['SWC_5']].mean().reset_index()

        # Load station features
        features_daily = df_smap[df_smap['Station_ID'] == station].copy()

        # Merge features & target
        merged = pd.merge(features_daily, target_daily, on='Date', how='inner').sort_values('Date').reset_index(drop=True)

        if len(merged) < look_back + 50:
            print(f"Warning: Insufficient data for {station}. Skipping.")
            continue

        # Chronological Split (< 2020-01-01 train, >= 2020-01-01 test)
        train_df = merged[merged['Date'] < test_start_date].copy()
        test_df = merged[merged['Date'] >= test_start_date].copy()

        if len(train_df) <= look_back or len(test_df) == 0:
            print(f"Warning: Train/Test split invalid for {station}. Skipping.")
            continue

        # Fit independent scalers STRICTLY on training data
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(train_df[features].values)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(train_df[['SWC_5']].values)

        # Scale datasets
        scaled_train_feat = feature_scaler.transform(train_df[features].values)
        scaled_train_targ = target_scaler.transform(train_df[['SWC_5']].values)

        scaled_test_feat = feature_scaler.transform(test_df[features].values)
        scaled_test_targ = target_scaler.transform(test_df[['SWC_5']].values)

        # Create training sequences
        X_train_lstm, X_train_ml, y_train = [], [], []
        for j in range(look_back, len(scaled_train_feat)):
            X_train_lstm.append(scaled_train_feat[j-look_back:j, :])
            X_train_ml.append(scaled_train_feat[j, :])
            y_train.append(scaled_train_targ[j, 0])

        X_train_lstm, X_train_ml, y_train = np.array(X_train_lstm), np.array(X_train_ml), np.array(y_train)

        # Create testing sequences
        X_test_lstm, X_test_ml, y_test_scaled = [], [], []
        for j in range(look_back, len(scaled_test_feat)):
            X_test_lstm.append(scaled_test_feat[j-look_back:j, :])
            X_test_ml.append(scaled_test_feat[j, :])
            y_test_scaled.append(scaled_test_targ[j, 0])

        X_test_lstm, X_test_ml = np.array(X_test_lstm), np.array(X_test_ml)
        y_test_actual = target_scaler.inverse_transform(np.array(y_test_scaled).reshape(-1, 1)).flatten()

        # 1. Train & Predict LSTM
        lstm_model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train_lstm, y_train, epochs=40, batch_size=32, verbose=0)
        
        lstm_preds_scaled = lstm_model.predict(X_test_lstm, verbose=0)
        lstm_preds = target_scaler.inverse_transform(lstm_preds_scaled).flatten()

        # 2. Train & Predict XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_ml, y_train)
        
        xgb_preds_scaled = xgb_model.predict(X_test_ml)
        xgb_preds = target_scaler.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).flatten()

        # 3. Train & Predict SVM
        svm_model = SVR(C=1.0, epsilon=0.01)
        svm_model.fit(X_train_ml, y_train)
        
        svm_preds_scaled = svm_model.predict(X_test_ml)
        svm_preds = target_scaler.inverse_transform(svm_preds_scaled.reshape(-1, 1)).flatten()

        # Calculate metrics
        rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_preds))
        r2_lstm = r2_score(y_test_actual, lstm_preds)

        rmse_xgb = np.sqrt(mean_squared_error(y_test_actual, xgb_preds))
        r2_xgb = r2_score(y_test_actual, xgb_preds)

        rmse_svm = np.sqrt(mean_squared_error(y_test_actual, svm_preds))
        r2_svm = r2_score(y_test_actual, svm_preds)

        results.append({
            'Station': station,
            'LSTM_R2': r2_lstm,
            'LSTM_RMSE': rmse_lstm,
            'XGB_R2': r2_xgb,
            'XGB_RMSE': rmse_xgb,
            'SVM_R2': r2_svm,
            'SVM_RMSE': rmse_svm
        })

    # Display Clean Formatted Table
    df_res = pd.DataFrame(results)
    
    # Formatted printout
    print("\n" + "=" * 92)
    print(f"{'Station':<10} | {'LSTM R²':<9} {'LSTM RMSE':<10} | {'XGB R²':<9} {'XGB RMSE':<10} | {'SVM R²':<9} {'SVM RMSE':<10}")
    print("-" * 92)
    for row in results:
        print(f"{row['Station']:<10} | {row['LSTM_R2']:^9.4f} {row['LSTM_RMSE']:^10.4f} | {row['XGB_R2']:^9.4f} {row['XGB_RMSE']:^10.4f} | {row['SVM_R2']:^9.4f} {row['SVM_RMSE']:^10.4f}")
    print("-" * 92)
    
    # Summary Averages
    avg_lstm_r2, avg_lstm_rmse = df_res['LSTM_R2'].mean(), df_res['LSTM_RMSE'].mean()
    avg_xgb_r2, avg_xgb_rmse = df_res['XGB_R2'].mean(), df_res['XGB_RMSE'].mean()
    avg_svm_r2, avg_svm_rmse = df_res['SVM_R2'].mean(), df_res['SVM_RMSE'].mean()
    
    print(f"{'AVERAGE':<10} | {avg_lstm_r2:^9.4f} {avg_lstm_rmse:^10.4f} | {avg_xgb_r2:^9.4f} {avg_xgb_rmse:^10.4f} | {avg_svm_r2:^9.4f} {avg_svm_rmse:^10.4f}")
    print("=" * 92 + "\n")

if __name__ == "__main__":
    main()
