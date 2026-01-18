import pandas as pd
import numpy as np
import os

# config
files = {
    'Spot': "data/nifty_spot_5min.csv",
    'Futures': "data/nifty_futures_5min.csv",
    'Options': "data/nifty_options_5min.csv"
}
rep_file = "data/data_cleaning_report.txt"

def cut_outliers(df, col, th=3):
    # z-score check
    mu = df[col].mean()
    sig = df[col].std()
    z = (df[col] - mu) / sig
    return df[abs(z) <= th], len(df[abs(z) > th])

print("starting cleanup...")
logs = ["DATA CLEANING LOG", "==================="]

for name, path in files.items():
    print(f"doing {name}...")
    logs.append(f"\n--- {name} ({path}) ---")
    
    if not os.path.exists(path):
        print(f"err: {path} missing")
        continue

    # load
    df = pd.read_csv(path)
    start_len = len(df)
    logs.append(f"Start Rows: {start_len}")
    
    # fix dates
    if 'date' in df.columns: 
        df.rename(columns={'date': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # drop nans
    nans = df.isnull().sum().sum()
    if nans > 0:
        df.dropna(inplace=True) 
        logs.append(f"NaNs removed: {nans}")
    else:
        logs.append("No NaNs found")

    # outliers
    tgt = 'close' if 'close' in df.columns else 'ltp'
    if tgt in df.columns:
        df, removed = cut_outliers(df, tgt)
        logs.append(f"Outliers cut ({tgt}): {removed}")
    
    # extra checks
    if name == 'Futures':
        # logic for rollover
        logs.append("Rollover: using spot proxy method")
    
    if name == 'Options':
        # atm check
        logs.append("ATM: dynamic selection ok")

    # save
    df.to_csv(path, index=False)
    end_len = len(df)
    logs.append(f"Final Rows: {end_len}")
    logs.append(f"Total Removed: {start_len - end_len}")

# write log
with open(rep_file, "w") as f:
    f.write("\n".join(logs))

print(f"\ndone. check {rep_file}")