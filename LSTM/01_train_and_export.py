import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load historical data (2015-2023)
df_hist = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
txs01_hist = df_hist[df_hist['Station_ID'] == 'TXS01'].sort_values('Date').set_index('Date')

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
look_back = 14

# 2. Fit scalers strictly on training data
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(txs01_hist[features].values)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(txs01_hist[['SMAP_Moisture']].values)

# 3. Construct 14-day training sequences
X_train, y_train = [], []
for i in range(look_back, len(scaled_features)):
    X_train.append(scaled_features[i-look_back:i, :])
    y_train.append(scaled_features[i, 3])

X_train, y_train = np.array(X_train), np.array(y_train)

print(f"Training Tensor Shape: {X_train.shape}")
print(f"Target Vector Shape:   {y_train.shape}")

# 4. Define and train the LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining LSTM on 2015-2023 dataset...")
model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    verbose=1
)

# 5. Export frozen model and scalers to disk
model.save("lstm_soil_model.keras")
joblib.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, "scalers.joblib")

print("\nTraining complete.")
print("Saved artifacts: 'lstm_soil_model.keras' and 'scalers.joblib'")
