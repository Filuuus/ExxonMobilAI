import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("Phase 1: Initializing Production Training (2015 - Sept 2021)...")

# ==========================================
# 1. LOAD HISTORICAL DATA (TRAINING)
# ==========================================
# Features: SMAP + Daymet
df_hist_features = pd.read_csv("datasets/TxSON_SMAP_Weather_Merged.csv")
df_hist_features['Date'] = pd.to_datetime(df_hist_features['Date'])
txs01_features = df_hist_features[df_hist_features['Station_ID'] == 'TXS01'].copy()

# Target: TxSON Ground Truth
df_txson = pd.read_csv("datasets/TxSON-Station-Files/Station1_filled_Data.csv")
df_txson['Date'] = pd.to_datetime(df_txson['Unnamed: 0']).dt.normalize()
txs01_target = df_txson.groupby('Date')[['SWC_5']].mean().reset_index()

# Inner merge automatically isolates the 2015-2021 period where sensors were active
train_df = pd.merge(txs01_features, txs01_target, on='Date', how='inner').sort_values('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
X_train_raw = train_df[features].values
y_train_raw = train_df[['SWC_5']].values

# ==========================================
# 2. FIT PRODUCTION SCALERS
# ==========================================
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_X_train = feature_scaler.fit_transform(X_train_raw)

target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_y_train = target_scaler.fit_transform(y_train_raw)

# ==========================================
# 3. BUILD TRAINING SEQUENCES
# ==========================================
look_back = 14
X_train, y_train = [], []
for i in range(look_back, len(scaled_X_train)):
    X_train.append(scaled_X_train[i-look_back:i, :])
    y_train.append(scaled_y_train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# ==========================================
# 4. TRAIN & FREEZE THE MODEL
# ==========================================
# Training on 100% of the physical ground truth to maximize robustness
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# At this point, the model and scalers can be saved (e.g., model.save()) for live deployment.

print("\nPhase 2: Executing Blind Inference (Sept 2021 - Dec 2025)...")

# ==========================================
# 5. PREPARE FUTURE ESTIMATION DATA
# ==========================================
# Load the 2024-2025 new datasets
smap_new = pd.read_csv("datasets/TxSON24TDSMAP-SPL3SMP-E-006-results.csv")
weather_new = pd.read_csv("datasets/NASA_Weather/TxSON-Daymet-Weather-24-TD-DAYMET-004-results.csv")

smap_new = smap_new.rename(columns={'ID': 'Station_ID', 'SPL3SMP_E_006_Soil_Moisture_Retrieval_Data_AM_soil_moisture': 'SMAP_Moisture'})
weather_new = weather_new.rename(columns={'ID': 'Station_ID', 'DAYMET_004_prcp': 'Precipitation', 'DAYMET_004_tmax': 'T_Max', 'DAYMET_004_tmin': 'T_Min'})

smap_new['SMAP_Moisture'] = smap_new['SMAP_Moisture'].replace(-9999.0, np.nan)
smap_new['Date'] = pd.to_datetime(smap_new['Date'])
weather_new['Date'] = pd.to_datetime(weather_new['Date'])

df_new = pd.merge(weather_new, smap_new[['Station_ID', 'Date', 'SMAP_Moisture']], on=['Station_ID', 'Date'], how='left')
txs01_new = df_new[df_new['Station_ID'] == 'TXS01'].copy()
txs01_new['SMAP_Moisture'] = txs01_new['SMAP_Moisture'].ffill().bfill()

# Isolate the gap period (late 2021 through 2023) from the original merged file
gap_features = txs01_features[txs01_features['Date'] > train_df['Date'].max()].copy()
gap_features['SMAP_Moisture'] = gap_features['SMAP_Moisture'].ffill().bfill()

# Combine gap period with the new 2024-2025 data to create one continuous future timeline
future_df = pd.concat([gap_features[features + ['Date']], txs01_new[features + ['Date']]]).sort_values('Date').reset_index(drop=True)

# ==========================================
# 6. RUN CONTINUOUS INFERENCE
# ==========================================
# TRANSFORM using the locked historical scalers
scaled_X_future = feature_scaler.transform(future_df[features].values)

X_future, future_dates = [], []
for i in range(look_back, len(scaled_X_future)):
    X_future.append(scaled_X_future[i-look_back:i, :])
    future_dates.append(future_df['Date'].iloc[i])

X_future = np.array(X_future)

# Predict the physical ground truth mathematically
scaled_preds = model.predict(X_future)
y_pred_future = target_scaler.inverse_transform(scaled_preds).flatten()

# ==========================================
# 7. VISUALIZATION FOR PRESENTATION
# ==========================================
plt.figure(figsize=(16, 6))

# Plot historical training context
plt.plot(train_df['Date'], train_df['SWC_5'], label='Actual TxSON Ground Truth (Training Phase)', color='black', alpha=0.6, linewidth=1)

# Plot the multi-year blind estimation
plt.plot(future_dates, y_pred_future, label='LSTM Blind Estimation (Deployment Phase)', color='crimson', linewidth=1.5)

# Formatting
plt.axvline(x=pd.to_datetime('2021-09-01'), color='gray', linestyle='--', linewidth=2, label='Sensor Deactivation (Inference Starts)')
plt.title('Production Deployment: Multi-Year Soil Moisture Estimation via Satellite Forcing (TXS01)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Calculated Volumetric Soil Moisture ($m^3/m^3$)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Final_Production_Estimation.png', dpi=300)
print("\nSuccess! Production plot saved as 'Final_Production_Estimation.png'.")