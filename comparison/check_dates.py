import pandas as pd
import numpy as np

# Check LSTM Dates
df_smap = pd.read_csv("datasets/TxSON_SMAP_Weather_Merged.csv")
df_smap['Date'] = pd.to_datetime(df_smap['Date'])
txs01_smap = df_smap[df_smap['Station_ID'] == 'TXS01'].sort_values('Date')

df_txson = pd.read_csv("datasets/TxSON-Station-Files/Station1_filled_Data.csv")
df_txson['Date'] = pd.to_datetime(df_txson['Unnamed: 0']).dt.normalize()
txs01_ground_truth = df_txson.groupby('Date')[['SWC_5']].mean().reset_index()

merged_data = pd.merge(txs01_smap, txs01_ground_truth, on='Date', how='inner').set_index('Date')
dates = merged_data.index[14:]
split = int(len(dates) * 0.8)
test_dates_lstm = dates[split:]
print(f"LSTM Test Dates: {test_dates_lstm.min()} to {test_dates_lstm.max()} (Total: {len(test_dates_lstm)})")

# Check XGBoost/SVM Dates
import sys
sys.path.append("Xgboost y svm")
from soil_moisture_models import build_panel, make_target, FEATURES_FULL

panel = build_panel()
df = make_target(panel, 1) # h=1
df = df.dropna(subset=FEATURES_FULL + ["y"])
te = df[(df.Date >= "2020-01-01") & (df.Station_ID == "TXS01")]
print(f"XGB/SVM Test Dates: {te.Date.min()} to {te.Date.max()} (Total: {len(te.Date)})")

# Let's find a common test timeframe
common_start = max(test_dates_lstm.min(), te.Date.min())
common_end = min(test_dates_lstm.max(), te.Date.max())
print(f"Common Test Frame: {common_start} to {common_end}")
