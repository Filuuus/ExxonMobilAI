import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# 1. Load frozen model and scalers
print("Loading saved model and scalers...")
model = load_model("lstm_soil_model.keras")
scalers = joblib.load("scalers.joblib")
feature_scaler = scalers['feature_scaler']
target_scaler = scalers['target_scaler']

# 2. Ingest 2024-2025 out-of-sample data
print("Processing 2024-2025 out-of-sample dataset...")
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

smap_new['SMAP_Moisture'] = smap_new['SMAP_Moisture'].replace(-9999.0, np.nan)
smap_new['Date'] = pd.to_datetime(smap_new['Date'])
weather_new['Date'] = pd.to_datetime(weather_new['Date'])

df_new = pd.merge(weather_new, smap_new[['Station_ID', 'Date', 'SMAP_Moisture']], on=['Station_ID', 'Date'], how='left')
txs01_new = df_new[df_new['Station_ID'] == 'TXS01'].sort_values('Date').copy()

# Forward-fill missing satellite passes so no future data is referenced during imputation
txs01_new['SMAP_Moisture'] = txs01_new['SMAP_Moisture'].ffill().bfill()

features = ['Precipitation', 'T_Max', 'T_Min', 'SMAP_Moisture']
look_back = 14

# 3. Transform features using strictly the historical scalers
scaled_new = feature_scaler.transform(txs01_new[features].values)

X_test, y_test_scaled = [], []
dates_test = txs01_new['Date'].values[look_back:]

for i in range(look_back, len(scaled_new)):
    X_test.append(scaled_new[i-look_back:i, :])
    y_test_scaled.append(scaled_new[i, 3])

X_test = np.array(X_test)
y_test_scaled = np.array(y_test_scaled)

# 4. Generate predictions on unseen data
print("\nRunning inference on 2024-2025 sequences...")
scaled_preds = model.predict(X_test)

y_pred = target_scaler.inverse_transform(scaled_preds).flatten()
y_true = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# 5. Calculate validation metrics
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2 = r2_score(y_true, y_pred)

print("\n" + "="*50)
print(f"STRICT OUT-OF-SAMPLE RESULTS (2024-2025 - TXS01)")
print(f"RMSE: {rmse:.4f} m³/m³")
print(f"R²:   {r2:.4f}")
print("="*50)

# 6. Plot the forecast vs reality
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_true, label='Actual Satellite Observation (SMAP)', color='black', alpha=0.7)
plt.plot(dates_test, y_pred, label='LSTM Out-of-Sample Forecast', color='crimson', linestyle='--')
plt.title(f'Strict Out-of-Sample Validation (2024–2025)\nRMSE: {rmse:.4f} | R²: {r2:.4f}')
plt.xlabel('Date')
plt.ylabel('Volumetric Soil Moisture ($m^3/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Strict_2024_2025_Validation.png", dpi=300)
print("\nPlot saved as 'Strict_2024_2025_Validation.png'.")
