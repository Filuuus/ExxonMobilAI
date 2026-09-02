import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load data & recreate the exact feature set
df = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
txs01_data = df[df['Station_ID'] == 'TXS01'].copy()
txs01_data['Date'] = pd.to_datetime(txs01_data['Date'])
txs01_data = txs01_data.sort_values('Date').set_index('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
dataset = txs01_data[features].values

# 2. Fit scaler on features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Dedicated scaler for inverse transforming the target (SMAP_Moisture is index 3)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(dataset[:, [3]])

# 3. Construct 14-day sequences
look_back = 14
X, y = [], []
dates = txs01_data.index[look_back:]

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, :])
    y.append(scaled_data[i, 3])

X, y = np.array(X), np.array(y)

# Chronological 80/20 split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = dates[split:]

# 4. Define and compile the LSTM Architecture
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 5. Train the model
print("\nTraining LSTM network...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Generate predictions and invert scaling
scaled_predictions = model.predict(X_test)
y_pred = target_scaler.inverse_transform(scaled_predictions).flatten()
y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 7. Evaluate RMSE
lstm_rmse = mean_squared_error(y_true, y_pred) ** 0.5
print(f"\n==========================================")
print(f"LSTM Test RMSE: {lstm_rmse:.4f}")
print(f"==========================================")

# 8. Visualization
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_true, label='Actual Ground Truth (SMAP)', color='black', alpha=0.7)
plt.plot(test_dates, y_pred, label='LSTM Forecast (Multivariate)', color='crimson', linestyle='--')
plt.title('TxSON (TXS01) Soil Moisture Prediction: LSTM with Daymet Weather Forcing')
plt.xlabel('Date')
plt.ylabel('Volumetric Soil Moisture (m³/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()