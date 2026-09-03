import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def scan_cleaned_stations(clean_dir="datasets/final_clean/CSV"):
    """
    Scans and inventories all cleaned TxSON station CSV files.
    Returns a DataFrame with coverage metrics for each station.
    """
    clean_path = Path(clean_dir)
    files = sorted(clean_path.glob("*_filled_Data.csv"))
    
    records = []
    print(f"[+] Scanning {len(files)} cleaned TxSON station files in '{clean_dir}'...")
    
    for f in files:
        station_name = f.stem.replace("_filled_Data", "")
        try:
            # Read first and last lines for date span, and sample data
            df = pd.read_csv(f)
            time_col = df.columns[0]
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col]).sort_values(time_col)
            
            start_date = df[time_col].min()
            end_date = df[time_col].max()
            total_hours = len(df)
            
            has_swc5 = 'SWC_5' in df.columns
            swc5_valid_pct = (df['SWC_5'].notna().mean() * 100.0) if has_swc5 else 0.0
            swc5_mean = df['SWC_5'].mean() if has_swc5 else np.nan
            
            has_ppt = 'Ppt' in df.columns
            has_tair = 'Tair' in df.columns
            
            records.append({
                'Station_ID': station_name,
                'File': f.name,
                'Start_Date': start_date.strftime('%Y-%m-%d') if pd.notnull(start_date) else 'N/A',
                'End_Date': end_date.strftime('%Y-%m-%d') if pd.notnull(end_date) else 'N/A',
                'Total_Hours': total_hours,
                'Total_Days': round(total_hours / 24.0, 1),
                'Has_SWC_5cm': has_swc5,
                'SWC_5_Valid_Pct': round(swc5_valid_pct, 2),
                'Mean_SWC_5cm': round(swc5_mean, 4) if pd.notnull(swc5_mean) else np.nan,
                'Has_Precipitation': has_ppt,
                'Has_AirTemp': has_tair
            })
        except Exception as e:
            records.append({
                'Station_ID': station_name,
                'File': f.name,
                'Error': str(e)
            })
            
    catalog_df = pd.DataFrame(records)
    out_catalog = clean_path / "txson_expanded_station_catalog.csv"
    catalog_df.to_csv(out_catalog, index=False)
    print(f"[+] Saved catalog summary for {len(catalog_df)} stations to: {out_catalog}")
    return catalog_df

def load_daily_station_data(station_file):
    """
    Loads an hourly station CSV from datasets/cleanup/CSV and aggregates it to daily resolution.
    Returns DataFrame indexed by Date with daily mean SWC_5, total Ppt, and min/max temps.
    """
    df = pd.read_csv(station_file)
    time_col = df.columns[0]
    df['Date'] = pd.to_datetime(df[time_col]).dt.normalize()
    
    agg_dict = {}
    if 'SWC_5' in df.columns:
        agg_dict['SWC_5'] = 'mean'
    if 'SWC_10' in df.columns:
        agg_dict['SWC_10'] = 'mean'
    if 'Ppt' in df.columns:
        agg_dict['Ppt'] = 'sum'
    if 'Tair' in df.columns:
        agg_dict['T_Mean'] = 'mean'
        agg_dict['T_Max'] = 'max'
        agg_dict['T_Min'] = 'min'
    elif 'T_5' in df.columns:
        agg_dict['T_5'] = 'mean'
        
    daily = df.groupby('Date').agg(agg_dict).reset_index()
    return daily

if __name__ == "__main__":
    print("=" * 75)
    print("  TxSON CLEANED STATION DATASET INTEGRATOR & AUDIT")
    print("=" * 75)
    catalog = scan_cleaned_stations()
    print("\n[+] Station Integration Overview:")
    cols_to_show = ['Station_ID', 'Start_Date', 'End_Date', 'Total_Days', 'Has_SWC_5cm', 'Mean_SWC_5cm']
    available_cols = [c for c in cols_to_show if c in catalog.columns]
    print(catalog[available_cols].to_string(index=False))
    print("=" * 75)
