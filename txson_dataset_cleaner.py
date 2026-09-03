import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def clean_station_file(input_file, output_dir):
    """
    Cleans a single station CSV file by applying physical bounds,
    interpolating short gaps, and dropping unrecoverable rows.
    """
    df = pd.read_csv(input_file)
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    
    original_len = len(df)
    
    # 1. Apply Physical Bounds Filter
    swc_cols = [c for c in df.columns if c.startswith('SWC_')]
    temp_cols = [c for c in df.columns if c.startswith('T_') or c == 'Tair']
    ppt_cols = ['Ppt'] if 'Ppt' in df.columns else []

    # Enforce SWC bounds (0.01 to 0.6)
    for col in swc_cols:
        df.loc[(df[col] <= 0.01) | (df[col] > 0.6), col] = np.nan
        
        # Detect artificial linear interpolation (constant slope)
        first_diff = df[col].diff()
        second_diff = first_diff.diff().abs()
        # If the slope is perfectly constant but non-zero, it's artificially interpolated
        is_fake_slope = (first_diff.abs() > 1e-6) & (second_diff < 1e-8)
        df.loc[is_fake_slope, col] = np.nan
    for col in temp_cols:
        df.loc[(df[col] < -20.0) | (df[col] > 60.0), col] = np.nan
        
    # Enforce Precipitation bounds (>= 0.0)
    for col in ppt_cols:
        df.loc[df[col] < 0.0, col] = 0.0

    # 2. Smart Time-Series Interpolation
    # We want to interpolate short gaps (<= 12 hours) linearly.
    # Pandas `interpolate(limit=12)` does exactly this.
    
    cols_to_interpolate = swc_cols + temp_cols
    for col in cols_to_interpolate:
        df[col] = df[col].interpolate(method='linear', limit=12, limit_direction='both')

    # Precipitation should NOT be interpolated, missing precipitation is usually 0,
    # but to be safe we can fill short gaps with 0.
    for col in ppt_cols:
        df[col] = df[col].fillna(0.0)

    # 3. Aggressive Pruning
    # If the primary target (SWC_5) is still NaN, we drop the row.
    if 'SWC_5' in df.columns:
        df = df.dropna(subset=['SWC_5'])
    
    dropped = original_len - len(df)
        
    # Save to output directory
    station_name = Path(input_file).name
    out_path = Path(output_dir) / station_name
    df.to_csv(out_path, index=False)
    
    return original_len, dropped, len(df)

def main():
    input_dir = "datasets/cleanup/CSV"
    output_dir = "datasets/final_clean/CSV"
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(Path(input_dir).glob("*_filled_Data.csv"))
    print("=" * 75)
    print("  TxSON DATASET CLEANER & PHYSICAL BOUNDS ENFORCER")
    print("=" * 75)
    print(f"[+] Found {len(files)} files to clean.")
    
    total_original = 0
    total_dropped = 0
    total_retained = 0
    
    for f in files:
        orig, dropped, retained = clean_station_file(f, output_dir)
        total_original += orig
        total_dropped += dropped
        total_retained += retained
        
        pct_dropped = (dropped / orig * 100) if orig > 0 else 0
        print(f" -> {f.stem}: Dropped {dropped} rows ({pct_dropped:.1f}%), Retained {retained} rows")
        
    print("=" * 75)
    print(f"[+] CLEANING COMPLETE")
    print(f"    Total Rows Evaluated: {total_original}")
    if total_original > 0:
        print(f"    Total Rows Dropped:   {total_dropped} ({(total_dropped/total_original)*100:.1f}%)")
    print(f"    Total Rows Retained:  {total_retained}")
    print(f"[+] Clean files saved to '{output_dir}'")
    
if __name__ == "__main__":
    main()
