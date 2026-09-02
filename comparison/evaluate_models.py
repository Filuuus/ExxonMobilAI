import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def extract_xgboost_svm_predictions():
    """Extract predictions from XGBoost and SVM using their forecast experiment logic."""
    print("Extracting XGBoost and SVM predictions...")
    sys.path.append('Xgboost y svm')
    import soil_moisture_models
    
    # Load panel dataset
    panel = pd.read_csv('Xgboost y svm/results/daily_panel.csv')
    panel['Date'] = pd.to_datetime(panel['Date'])
    
    # We use horizon 1 to align with LSTM's daily prediction
    soil_moisture_models.HORIZONS = [1]
    
    # Temporarily suppress print statements from the imported module if desired, 
    # but run_forecast doesn't print much.
    fc_res, preds_dict = soil_moisture_models.run_forecast(panel)
    
    preds_h1 = preds_dict[1]
    
    # Filter for TXS01 to align with LSTM evaluation
    mask = preds_h1['station'] == 'TXS01'
    dates_xgb = preds_h1['dates'][mask]
    y_true_xgb = preds_h1['y'][mask]
    y_pred_xgb = preds_h1['XGBoost'][mask]
    y_pred_svm = preds_h1['SVR'][mask]
    
    df_xgb = pd.DataFrame({
        'Date': dates_xgb,
        'y_true': y_true_xgb,
        'XGBoost': y_pred_xgb,
        'SVM': y_pred_svm
    }).set_index('Date').sort_index()
    
    # Remove from sys.path
    sys.path.pop()
    
    return df_xgb

def extract_lstm_predictions(test_start_date, test_end_date):
    """Retrain and extract LSTM predictions using Domain Adaptation (Targeting SWC_5)."""
    print("Extracting LSTM predictions...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    df_smap = pd.read_csv("datasets/TxSON_SMAP_Weather_Merged.csv")
    df_smap['Date'] = pd.to_datetime(df_smap['Date'])
    txs01_smap = df_smap[df_smap['Station_ID'] == 'TXS01'].sort_values('Date')

    df_txson = pd.read_csv("datasets/TxSON-Station-Files/Station1_filled_Data.csv")
    df_txson['Date'] = pd.to_datetime(df_txson['Unnamed: 0']).dt.normalize()
    txs01_ground_truth = df_txson.groupby('Date')[['SWC_5']].mean().reset_index()

    merged_data = pd.merge(txs01_smap, txs01_ground_truth, on='Date', how='inner').set_index('Date')
    
    # 1. Isolate Features (Inputs) and Target (Ground Truth)
    features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
    dataset_features = merged_data[features].values
    dataset_target = merged_data[['SWC_5']].values # Domain Adaptation Target

    # 2. Scale Independently (Strictly fit on training data before test_start_date)
    train_mask_raw = merged_data.index < pd.to_datetime(test_start_date)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(dataset_features[train_mask_raw])
    scaled_features = feature_scaler.transform(dataset_features)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(dataset_target[train_mask_raw])
    scaled_target = target_scaler.transform(dataset_target) 

    # 3. Build Sequences
    look_back = 14
    X_lstm, y_target, dates_lstm = [], [], []
    for i in range(look_back, len(scaled_features)):
        X_lstm.append(scaled_features[i-look_back:i, :])
        y_target.append(scaled_target[i, 0]) # Appending the SWC_5 target
        dates_lstm.append(merged_data.index[i])

    X_lstm = np.array(X_lstm)
    y_target = np.array(y_target)
    dates_lstm = pd.to_datetime(dates_lstm)

    # 4. Train/Test Split
    train_mask = dates_lstm < pd.to_datetime(test_start_date)
    test_mask = (dates_lstm >= pd.to_datetime(test_start_date)) & (dates_lstm <= pd.to_datetime(test_end_date))

    X_train = X_lstm[train_mask]
    y_train = y_target[train_mask]
    
    X_test = X_lstm[test_mask]
    test_dates = dates_lstm[test_mask]

    # 5. Train LSTM
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

    # 6. Predict and Inverse Transform
    scaled_preds = model.predict(X_test, verbose=0)
    y_pred_lstm = target_scaler.inverse_transform(scaled_preds).flatten()

    df_lstm = pd.DataFrame({
        'Date': test_dates,
        'LSTM': y_pred_lstm
    }).set_index('Date').sort_index()

    return df_lstm


def main():
    # 1. Extract Predictions
    df_xgb_svm = extract_xgboost_svm_predictions()
    test_start = df_xgb_svm.index.min()
    test_end = df_xgb_svm.index.max()
    print(f"Standardized Testing Timeframe: {test_start.date()} to {test_end.date()}")
    
    df_lstm = extract_lstm_predictions(test_start, test_end)
    
    # Merge on exactly the same dates
    df_all = pd.concat([df_xgb_svm, df_lstm], axis=1).dropna()
    print(f"Total overlapping days: {len(df_all)}")
    
    # 2. Calculate Metrics
    models = ['LSTM', 'XGBoost', 'SVM']
    colors = {'LSTM': 'Crimson', 'XGBoost': 'ForestGreen', 'SVM': 'SteelBlue'}
    metrics = {'RMSE': {}, 'MAE': {}, 'R2': {}}
    
    print("\nCalculated Metrics:")
    for m in models:
        y_p = df_all[m]
        y_t = df_all['y_true']
        
        metrics['RMSE'][m] = np.sqrt(mean_squared_error(y_t, y_p))
        metrics['MAE'][m] = mean_absolute_error(y_t, y_p)
        metrics['R2'][m] = r2_score(y_t, y_p)
        
        print(f"{m}:")
        print(f"  RMSE: {metrics['RMSE'][m]:.4f}")
        print(f"  MAE:  {metrics['MAE'][m]:.4f}")
        print(f"  R2:   {metrics['R2'][m]:.4f}")
        
    # 3. Generate Visualizations
    print("\nGenerating Model_Comparison_Matrix.png...")
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], wspace=0.3, hspace=0.4)
    
    # Plot A: Grouped Bar Chart
    ax_bar = fig.add_subplot(gs[0, 0])
    metrics_list = ['RMSE', 'MAE', 'R2']
    x = np.arange(len(metrics_list))
    width = 0.25
    
    for i, m in enumerate(models):
        values = [metrics[met][m] for met in metrics_list]
        ax_bar.bar(x + (i-1)*width, values, width, label=m, color=colors[m])
        
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_list, fontsize=12, fontweight='bold')
    ax_bar.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax_bar.legend()
    ax_bar.grid(axis='y', alpha=0.3)
    
    # Plot B: Time-Series Overlay
    ax_ts = fig.add_subplot(gs[0, 1:])
    ax_ts.plot(df_all.index, df_all['y_true'], label='Actual Soil Moisture (SWC_5)', color='black', linewidth=2.5)
    
    line_styles = {'LSTM': '--', 'XGBoost': '-.', 'SVM': ':'}
    for m in models:
        ax_ts.plot(df_all.index, df_all[m], label=f'{m} Prediction', 
                   color=colors[m], linestyle=line_styles[m], linewidth=1.5, alpha=0.8)
                   
    ax_ts.set_title('Time-Series Overlaid Predictions (TXS01)', fontsize=14, fontweight='bold')
    ax_ts.set_ylabel('Volumetric Soil Moisture ($m^3/m^3$)', fontsize=12)
    ax_ts.legend(loc='upper right', ncol=2)
    ax_ts.grid(True, alpha=0.3)
    
    # Plot C: Scatter Matrix (1x3)
    for i, m in enumerate(models):
        ax_scatter = fig.add_subplot(gs[1, i])
        ax_scatter.scatter(df_all['y_true'], df_all[m], color=colors[m], alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # 1:1 Line
        min_val = min(df_all['y_true'].min(), df_all[m].min())
        max_val = max(df_all['y_true'].max(), df_all[m].max())
        lims = [min_val, max_val]
        
        ax_scatter.plot(lims, lims, 'r--', label='1:1 Ideal Fit', linewidth=2)
        ax_scatter.set_title(f'{m} vs Actual', fontsize=14, fontweight='bold')
        ax_scatter.set_xlabel('Actual ($m^3/m^3$)', fontsize=12)
        ax_scatter.set_ylabel('Predicted ($m^3/m^3$)', fontsize=12)
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        
    plt.savefig('Model_Comparison_Matrix.png', dpi=300, bbox_inches='tight')
    print("Done! Visualization saved to Model_Comparison_Matrix.png")

if __name__ == "__main__":
    main()
