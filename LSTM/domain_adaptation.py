import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("Aligning satellite features with physical ground targets...")

# 1. Load Satellite/Weather Features
smap_df = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
smap_df['Date'] = pd.to_datetime(smap_df['Date'])
txs01_features = smap_df[smap_df['Station_ID'] == 'TXS01'].copy()

# 2. Load and Aggregate Ground Targets (TxSON)
txson_df = pd.read_csv("Station1_filled_Data.csv")
# Extract daily date from hourly timestamps
txson_df['Date'] = pd.to_datetime(txson_df['Unnamed: 0']).dt.normalize()
ground_target = txson_df.groupby('Date')[['SWC_5']].mean().reset_index()

# 3. Merge Datasets 
# An inner join automatically drops any days missing the physical sensor data
merged_df = pd.merge(txs01_features, ground_target, on='Date', how='inner').sort_values('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
dataset_features = merged_df[features].values
dataset_target = merged_df[['SWC_5']].values

# 4. Independent Scaling
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(dataset_features)

target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target = target_scaler.fit_transform(dataset_target)

# 5. Sequence Generation (14-day window)
look_back = 14
X, y = [], []
dates = merged_df['Date'].values[look_back:]

for i in range(look_back, len(scaled_features)):
    X.append(scaled_features[i-look_back:i, :])
    y.append(scaled_target[i, 0])

X, y = np.array(X), np.array(y)

# 6. Chronological 80/20 Split (2015-2020 Train | 2020-2021 Test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = dates[split:]

# 7. Model Architecture
print("Training Domain Adaptation Model...")
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# 8. Evaluate on Holdout
scaled_preds = model.predict(X_test)
y_pred = target_scaler.inverse_transform(scaled_preds).flatten()
y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2 = r2_score(y_true, y_pred)

print("\n==========================================")
print("DOMAIN ADAPTATION METRICS (Target: TxSON Physical Sensor)")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print("==========================================")

# 9. Visualization
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_true, label='Actual Ground Target (TxSON SWC_5)', color='black', alpha=0.7)
plt.plot(test_dates, y_pred, label='LSTM Forecast (Satellite Inputs)', color='crimson', linestyle='--')
plt.title(f'Domain Adaptation: Satellite to Ground Truth\nRMSE: {rmse:.4f} | R²: {r2:.4f}')
plt.xlabel('Date')
plt.ylabel('Volumetric Soil Moisture ($m^3/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Domain_Adaptation_Validation.png', dpi=300)
print("\nPlot saved as 'Domain_Adaptation_Validation.png'.")
