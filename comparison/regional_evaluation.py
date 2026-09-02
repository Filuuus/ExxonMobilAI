import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("Aggregating regional data from all TxSON stations...")

# ==========================================
# 1. LOAD & AGGREGATE ALL REGIONAL DATA
# ==========================================
df_smap = pd.read_csv("datasets/TxSON_SMAP_Weather_Merged.csv")
df_smap['Date'] = pd.to_datetime(df_smap['Date'])

regional_data = []
stations = ['TXS01', 'TXS02', 'TXS03', 'TXS04', 'TXS05', 'TXS06']

for i, station in enumerate(stations, start=1):
    # Load specific station target data
    try:
        df_target = pd.read_csv(f"datasets/TxSON-Station-Files/Station{i}_filled_Data.csv")
        df_target['Date'] = pd.to_datetime(df_target['Unnamed: 0']).dt.normalize()
        target_daily = df_target.groupby('Date')[['SWC_5']].mean().reset_index()
        
        # Load specific station features
        features_daily = df_smap[df_smap['Station_ID'] == station]
        
        # Merge
        merged = pd.merge(features_daily, target_daily, on='Date', how='inner').sort_values('Date')
        regional_data.append(merged)
    except FileNotFoundError:
        print(f"Warning: Data for {station} not found. Skipping.")

# Combine into one massive regional dataframe
df_regional = pd.concat(regional_data, ignore_index=True)

# ==========================================
# 2. CHRONOLOGICAL REGIONAL SPLIT & SCALING
# ==========================================
test_start_date = '2020-01-01'
train_df = df_regional[df_regional['Date'] < test_start_date].copy()
test_df = df_regional[df_regional['Date'] >= test_start_date].copy()

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']

# Fit scalers STRICTLY on the regional training data
feature_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaler.fit(train_df[features].values)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(train_df[['SWC_5']].values)

# ==========================================
# 3. BUILD STATION-ISOLATED SEQUENCES
# ==========================================
look_back = 14

def create_sequences(df):
    X_lstm, X_ml, y = [], [], []
    # Process each station individually to prevent sequence overlap
    for station in df['Station_ID'].unique():
        station_data = df[df['Station_ID'] == station].sort_values('Date')
        
        scaled_features = feature_scaler.transform(station_data[features].values)
        scaled_target = target_scaler.transform(station_data[['SWC_5']].values)
        
        for i in range(look_back, len(scaled_features)):
            X_lstm.append(scaled_features[i-look_back:i, :])
            X_ml.append(scaled_features[i, :]) # Current day features for ML models
            y.append(scaled_target[i, 0])
            
    return np.array(X_lstm), np.array(X_ml), np.array(y)

print("Building regional training and testing tensors...")
X_train_lstm, X_train_ml, y_train = create_sequences(train_df)
X_test_lstm, X_test_ml, y_test = create_sequences(test_df)

print(f"Total Regional Training Samples: {len(y_train)}")
print(f"Total Regional Testing Samples:  {len(y_test)}")

# ==========================================
# 4. TRAIN REGIONAL MODELS
# ==========================================
print("\nTraining Regional XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_ml, y_train)

print("Training Regional SVM...")
svm_model = SVR(C=1.0, epsilon=0.01)
svm_model.fit(X_train_ml, y_train)

print("Training Regional LSTM...")
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())

early_stopping = EarlyStopping(monitor='loss', patience=12, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

# ==========================================
# 5. GENERATE REGIONAL PREDICTIONS
# ==========================================
print("\nGenerating out-of-sample predictions...")
models = ['LSTM', 'XGBoost', 'SVM']
colors = {'LSTM': 'Crimson', 'XGBoost': 'ForestGreen', 'SVM': 'SteelBlue'}

y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

preds = {
    'LSTM': target_scaler.inverse_transform(lstm_model.predict(X_test_lstm, verbose=0)).flatten(),
    'XGBoost': target_scaler.inverse_transform(xgb_model.predict(X_test_ml).reshape(-1, 1)).flatten(),
    'SVM': target_scaler.inverse_transform(svm_model.predict(X_test_ml).reshape(-1, 1)).flatten()
}

# ==========================================
# 6. CALCULATE METRICS & VISUALIZE
# ==========================================
metrics = {'RMSE': {}, 'MAE': {}, 'R2': {}}
for m in models:
    metrics['RMSE'][m] = np.sqrt(mean_squared_error(y_true, preds[m]))
    metrics['MAE'][m] = mean_absolute_error(y_true, preds[m])
    metrics['R2'][m] = r2_score(y_true, preds[m])
    print(f"{m} - RMSE: {metrics['RMSE'][m]:.4f} | R2: {metrics['R2'][m]:.4f}")

plt.style.use('default')
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1], wspace=0.3)

# Plot A: Overall Regional Metrics Bar Chart
ax_bar = fig.add_subplot(gs[0])
metrics_list = ['RMSE', 'MAE', 'R2']
x = np.arange(len(metrics_list))
width = 0.25

for i, m in enumerate(models):
    values = [metrics[met][m] for met in metrics_list]
    ax_bar.bar(x + (i-1)*width, values, width, label=m, color=colors[m])

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics_list, fontsize=11, fontweight='bold')
ax_bar.set_title('Unified Regional Performance', fontsize=13, fontweight='bold')
ax_bar.legend()
ax_bar.grid(axis='y', alpha=0.3)

# Plot B, C, D: Scatter Matrices
for i, m in enumerate(models, start=1):
    ax_scatter = fig.add_subplot(gs[i])
    ax_scatter.scatter(y_true, preds[m], color=colors[m], alpha=0.3, edgecolors='none', s=10)
    
    lims = [min(y_true.min(), preds[m].min()), max(y_true.max(), preds[m].max())]
    ax_scatter.plot(lims, lims, 'r--', label='1:1 Ideal Fit', linewidth=2)
    ax_scatter.set_title(f'{m} vs Actual', fontsize=12, fontweight='bold')
    ax_scatter.set_xlabel('Actual ($m^3/m^3$)', fontsize=10)
    if i == 1:
        ax_scatter.set_ylabel('Predicted ($m^3/m^3$)', fontsize=10)
    ax_scatter.grid(True, alpha=0.3)

plt.savefig('Unified_Regional_Comparison.png', dpi=300, bbox_inches='tight')
print("\nDone! Regional comparison saved to 'Unified_Regional_Comparison.png'")