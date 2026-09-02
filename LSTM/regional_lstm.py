import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the merged dataset
df = pd.read_csv("TxSON_SMAP_Weather_Merged.csv")
df['Date'] = pd.to_datetime(df['Date'])

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
look_back = 14

# We need a global scaler for the target variable to inverse-transform later
global_target_scaler = MinMaxScaler(feature_range=(0, 1))
global_target_scaler.fit(df[['SMAP_Moisture']])

X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

print("Processing sequences for all 6 stations...")
for station in df['Station_ID'].unique():
    station_data = df[df['Station_ID'] == station].sort_values('Date').set_index('Date')
    dataset = station_data[features].values
    
    # Scale features per station
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, 3])
        
    X, y = np.array(X), np.array(y)
    
    # Chronological 80/20 split for this specific station
    split = int(len(X) * 0.8)
    X_train_list.append(X[:split])
    y_train_list.append(y[:split])
    X_test_list.append(X[split:])
    y_test_list.append(y[split:])

# Concatenate all stations into massive regional tensors
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"Regional Training Tensor Shape: {X_train.shape}")
print(f"Regional Testing Tensor Shape: {X_test.shape}")

# 2. Build and Train the Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining Regional LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64, # Increased batch size for the larger dataset
    validation_data=(X_test, y_test),
    verbose=1
)

# 3. Generate Regional Predictions
scaled_predictions = model.predict(X_test)
y_pred = global_target_scaler.inverse_transform(scaled_predictions).flatten()
y_true = global_target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 4. Calculate Final Presentation Metrics
rmse = mean_squared_error(y_true, y_pred) ** 0.5
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n" + "="*50)
print("FINAL REGIONAL LSTM METRICS (ALL 6 STATIONS)")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.4f}")
print(f"R-squared (R2) Score:           {r2:.4f}")
print("="*50)

# 5. Presentation Visualization: Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue', label='Predictions')

# Plot the ideal 1:1 perfect prediction line
min_val = min(np.min(y_true), np.min(y_pred))
max_val = max(np.max(y_true), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit (1:1)')

plt.title(f'LSTM Regional Model: Actual vs Predicted Soil Moisture\n(RMSE: {rmse:.4f} | R²: {r2:.4f})')
plt.xlabel('Actual Soil Moisture ($m^3/m^3$)')
plt.ylabel('Predicted Soil Moisture ($m^3/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('LSTM_Regional_Scatter.png', dpi=300)
print("\nPlot saved as 'LSTM_Regional_Scatter.png'. Ready for the presentation!")