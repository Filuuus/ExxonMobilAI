import pandas as pd

# 1. Load datasets
weather_df = pd.read_csv("TxSON-Daymet-Weather-Extraction-DAYMET-004-results.csv")
smap_df = pd.read_csv("SMAP_Cleaned.csv")

# 2. Standardize column names for the merge
weather_df = weather_df.rename(columns={
    'ID': 'Station_ID', 
    'DAYMET_004_prcp': 'Precipitation', 
    'DAYMET_004_tmax': 'T_Max', 
    'DAYMET_004_tmin': 'T_Min'
})

# 3. Convert dates to datetime objects
weather_df['Date'] = pd.to_datetime(weather_df['Date'])
smap_df['Date'] = pd.to_datetime(smap_df['Date'])

# 4. Merge datasets on Station and Date
merged_df = pd.merge(
    weather_df, 
    smap_df[['Station_ID', 'Date', 'SMAP_Moisture']], 
    on=['Station_ID', 'Date'], 
    how='left'
)

# 5. Interpolate missing moisture values per station (Forward and Backward Fill for edges)
merged_df['SMAP_Moisture'] = merged_df.groupby('Station_ID')['SMAP_Moisture'].transform(
    lambda x: x.interpolate(method='linear').bfill().ffill()
)

# 6. Save the final merged dataset to a new CSV file
output_filename = "TxSON_SMAP_Weather_Merged.csv"
merged_df.to_csv(output_filename, index=False)
print(f"Success! Merged dataset saved as '{output_filename}'")