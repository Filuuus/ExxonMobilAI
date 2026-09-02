import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

merged_df = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")


# Isolate features for a single station (e.g., TXS01) to train the prototype
txs01_data = merged_df[merged_df['Station_ID'] == 'TXS01'].set_index('Date')
features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
dataset = txs01_data[features].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create sequences (14-day lookback)
look_back = 14
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, :]) # All features for 14 days
    y.append(scaled_data[i, 3])             # Predicting SMAP_Moisture (index 3)

X, y = np.array(X), np.array(y)

# Chronological Train/Test Split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 1. Check array dimensions
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 2. Check scaling boundaries
print(f"\nScaled Data Min: {scaled_data.min()}")
print(f"Scaled Data Max: {scaled_data.max()}")