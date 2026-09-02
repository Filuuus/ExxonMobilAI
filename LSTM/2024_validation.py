import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. LOAD HISTORICAL DATA (2015-2023)
# ==========================================
print("Loading historical data...")
df_hist = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
txs01_hist = df_hist[df_hist['Station_ID'] == 'TXS01'].sort_values('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
look_back = 14

# ==========================================
# 2. STRICT SCALER FITTING (NO DATA LEAKAGE)
# ==========================================
# We fit the scalers ONLY on the 2015-2023 data. 
# This prevents the minimums/maximums of 2024 from influencing the training environment.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(txs01_hist[features].values)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(txs01_hist[['SMAP_Moisture']].values)

# ==========================================
# 3. PREPARE TRAINING TENSOR (2015-2023)
# ==========================================
# Apply the fitted scaler to the historical data
scaled_hist = scaler.transform(txs01_hist[features].values)
X_train, y_train = [], []
for i in range(look_back, len(scaled_hist)):
    X_train.append(scaled_hist[i-look_back:i, :])
    y_train.append(scaled_hist[i, 3])

X_train, y_train = np.array(X_train), np.array(y_train)

# ==========================================
# 4. LOAD & PREPARE UNSEEN DATA (2024-2025)
# ==========================================
print("Processing 2024-2025 unseen datasets...")
smap_new = pd.read_csv("TxSON24TDSMAP-SPL3SMP-E-006-results.csv")
weather_new = pd.read_csv("TxSON-Daymet-Weather-24-TD-DAYMET-004-results.csv")

smap_new = smap_new.rename(columns={
    'ID': 'Station_ID',
    'SPL3SMP_E_006_Soil_Moisture_Retrieval_Data_AM_soil_moisture': 'SMAP_Moisture'
})
weather_new = weather_new.rename(columns={
    'ID': 'Station_ID',
    'DAYMET_004_prcp': 'Precipitation',
    'DAYMET_004_tmax': 'T_Max',
    'DAYMET_004_tmin': 'T_Min'
})

# Handle NASA's missing data flags and merge
smap_new['SMAP_Moisture'] = smap_new['SMAP_Moisture'].replace(-9999.0, np.nan)
smap_new['Date'] = pd.to_datetime(smap_new['Date'])
weather_new['Date'] = pd.to_datetime(weather_new['Date'])

df_new = pd.merge(weather_new, smap_new[['Station_ID', 'Date', 'SMAP_Moisture']], on=['Station_ID', 'Date'], how='left')
txs01_new = df_new[df_new['Station_ID'] == 'TXS01'].sort_values('Date').copy()
txs01_new['SMAP_Moisture'] = txs01_new['SMAP_Moisture'].interpolate(method='linear').bfill().ffill()

# ==========================================
# 5. PREPARE TESTING TENSOR (2024-2025)
# ==========================================
# IMPORTANT: We use transform(), NOT fit_transform(), to apply the historical bounds to the new data
scaled_new = scaler.transform(txs01_new[features].values)
X_test, y_test_actual = [], []
test_dates = txs01_new['Date'].values[look_back:]

for i in range(look_back, len(scaled_new)):
    X_test.append(scaled_new[i-look_back:i, :])
    y_test_actual.append(scaled_new[i, 3])

X_test = np.array(X_test)
# Inverse transform the actual targets so we can calculate real-world metrics later
y_test_actual = target_scaler.inverse_transform(np.array(y_test_actual).reshape(-1, 1)).flatten()

# ==========================================
# 6. BUILD & TRAIN MODEL (2015-2023 Only)
# ==========================================
print("\nTraining LSTM strictly on 2015-2023 data...")
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
# Notice there is no validation_data parameter here, preventing any leakage during training
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# ==========================================
# 7. EVALUATE ON UNSEEN 2024-2025 DATA
# ==========================================
print("\nEvaluating on 2024-2025 out-of-sample data...")
scaled_preds = model.predict(X_test)
y_pred = target_scaler.inverse_transform(scaled_preds).flatten()

rmse = mean_squared_error(y_test_actual, y_pred) ** 0.5
r2 = r2_score(y_test_actual, y_pred)

print(f"\n==========================================")
print(f"Strict Out-of-Sample Metrics (TXS01 - 2024/2025):")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"==========================================")

# ==========================================
# 8. VISUALIZE
# ==========================================
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_actual, label='Actual SMAP (2024-2025)', color='black', alpha=0.7)
plt.plot(test_dates, y_pred, label='LSTM Forecast', color='crimson', linestyle='--')
plt.title(f'Strict Out-of-Sample Validation: TXS01 (RMSE: {rmse:.4f} | R²: {r2:.4f})')
plt.xlabel('Date')
plt.ylabel('Volumetric Soil Moisture ($m^3/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Strict_Validation_2024.png', dpi=300)
print("\nSaved plot as 'Strict_Validation_2024.png'.")