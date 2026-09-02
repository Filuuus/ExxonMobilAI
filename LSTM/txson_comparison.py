import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the Merged SMAP/Daymet Data (Your Model's Training Data)
df_smap = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
df_smap['Date'] = pd.to_datetime(df_smap['Date'])
txs01_smap = df_smap[df_smap['Station_ID'] == 'TXS01'].sort_values('Date')

# 2. Load the TxSON Ground-Truth Data (Station 1)
df_txson = pd.read_csv("Station1_filled_Data.csv")
# The first column 'Unnamed: 0' contains the hourly timestamps
df_txson['Date'] = pd.to_datetime(df_txson['Unnamed: 0']).dt.normalize() # Strip time to get daily dates

# Aggregate hourly readings to daily averages
txs01_ground_truth = df_txson.groupby('Date')[['SWC_5']].mean().reset_index()

# 3. Align the Datasets
# Merge the SMAP/Daymet features with the TxSON ground truth on the exact same dates
merged_data = pd.merge(txs01_smap, txs01_ground_truth, on='Date', how='inner').set_index('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
dataset = merged_data[features].values
txson_actuals = merged_data['SWC_5'].values # The true ground target

# 4. Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Scalers for inverse transformations later
smap_target_scaler = MinMaxScaler(feature_range=(0, 1))
smap_target_scaler.fit(dataset[:, [3]])

# 5. Create Sequences (14-day lookback)
look_back = 14
X, y_smap, y_txson = [], [], []
dates = merged_data.index[look_back:]

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, :])
    y_smap.append(scaled_data[i, 3])      # SMAP target for training
    y_txson.append(txson_actuals[i])      # TxSON true value for evaluation

X = np.array(X)
y_smap = np.array(y_smap)
y_txson = np.array(y_txson)

# 6. Chronological Train/Test Split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train_smap, y_test_smap = y_smap[:split], y_smap[split:]
y_test_txson = y_txson[split:] # We evaluate against this!
test_dates = dates[split:]

# 7. Build & Train the LSTM (Trained on SMAP/Satellite proxy)
print("Training LSTM on SMAP Proxy Data...")
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train_smap, epochs=40, batch_size=32, verbose=0) # verbose=0 to keep terminal clean

# 8. Generate Predictions and Evaluate Against TxSON Ground Truth
scaled_preds = model.predict(X_test)
y_pred_smap = smap_target_scaler.inverse_transform(scaled_preds).flatten()

rmse = mean_squared_error(y_test_txson, y_pred_smap) ** 0.5
r2 = r2_score(y_test_txson, y_pred_smap)

print(f"\nSatellite Model vs Ground Truth (Station 1):")
print(f"True RMSE: {rmse:.4f}")
print(f"True R²: {r2:.4f}")

# 9. Visualization
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_txson, label='Actual TxSON Sensor (SWC_5)', color='black')
plt.plot(test_dates, y_pred_smap, label='LSTM Prediction (Trained on SMAP)', color='crimson', linestyle='--')
plt.title(f'Satellite LSTM Forecast vs Physical Ground Sensors (Station TXS01)\nTrue RMSE: {rmse:.4f} | R²: {r2:.4f}')
plt.xlabel('Date')
plt.ylabel('Volumetric Soil Moisture ($m^3/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('LSTM_vs_TxSON.png', dpi=300)
print("\nPlot saved as 'LSTM_vs_TxSON.png'.")