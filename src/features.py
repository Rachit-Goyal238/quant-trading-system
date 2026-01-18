import pandas as pd
import numpy as np
from datetime import timedelta
from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks

# setup
infile = "data/nifty_merged_5min.csv"
outfile = "data/nifty_features_5min.csv"
r = 0.065 

print("starting...")

# read data
df = pd.read_csv(infile)
df['datetime'] = pd.to_datetime(df['datetime'])

# calc emas
# do it on unique times first so its faster
spot_df = df[['datetime', 'close_spot']].drop_duplicates().set_index('datetime').sort_index()
spot_df['ema_5'] = spot_df['close_spot'].ewm(span=5, adjust=False).mean()
spot_df['ema_15'] = spot_df['close_spot'].ewm(span=15, adjust=False).mean()

# join back
df = pd.merge(df, spot_df[['ema_5', 'ema_15']], on='datetime', how='left')

# expiry calc
def get_years_to_expiry(row_date):
    # last thurs logic
    next_month = row_date.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    offset = (last_day.weekday() - 3) % 7
    expiry = last_day - timedelta(days=offset)
    
    if row_date.date() > expiry.date():
        expiry = expiry + timedelta(days=30) 
        
    days = (expiry - row_date).total_seconds() / (24*3600)
    return max(days, 0.01) / 365.0 

df['t'] = df['datetime'].apply(get_years_to_expiry)

# fix flags for lib
type_col = 'right' if 'right' in df.columns else 'type'
df['flag'] = df[type_col].map({'CE': 'c', 'PE': 'p', 'Call': 'c', 'Put': 'p'})

print("doing greeks...")

# calc iv
try:
    df['implied_iv'] = vectorized_implied_volatility(
        df['close'].values,       
        df['close_spot'].values,  
        df['strike_price'].values,
        df['t'].values,           
        r,          
        df['flag'].values,        
        q=0,                                    
        return_as='numpy'
    )
except:
    print("iv calc failed, using default")
    df['implied_iv'] = 0.2

df['implied_iv'] = df['implied_iv'].fillna(0)

# get greeks
greeks = get_all_greeks(
    df['flag'].values,
    df['close_spot'].values,
    df['strike_price'].values,
    df['t'].values,
    r,
    df['implied_iv'].values, 
    q=0,
    model='black_scholes',
    return_as='dataframe'
)

df['delta'] = greeks['delta']
df['gamma'] = greeks['gamma']
df['theta'] = greeks['theta']
df['vega'] = greeks['vega']
df['rho'] = greeks['rho']

# derived stats
print("calc pcr etc...")

pivoted = df.pivot_table(
    index='datetime', 
    columns=type_col, 
    values=['open_interest', 'volume', 'implied_iv'], 
    aggfunc='sum'
)

ce_oi = pivoted['open_interest']['CE']
pe_oi = pivoted['open_interest']['PE']
ce_vol = pivoted['volume']['CE']
pe_vol = pivoted['volume']['PE']
ce_iv = pivoted['implied_iv']['CE']
pe_iv = pivoted['implied_iv']['PE']

# map back to main df
df['pcr_oi'] = df['datetime'].map(pe_oi / ce_oi.replace(0, 1))
df['pcr_volume'] = df['datetime'].map(pe_vol / ce_vol.replace(0, 1))
df['average_iv'] = df['datetime'].map((ce_iv + pe_iv) / 2)
df['iv_spread'] = df['datetime'].map(ce_iv - pe_iv)

df['futures_basis'] = (df['close_fut'] - df['close_spot']) / df['close_spot']
df['spot_returns'] = df['close_spot'].pct_change()
df['gamma_exposure'] = df['close_spot'] * df['gamma'] * df['open_interest']
df['delta_neutral_ratio'] = abs(df['delta']) 

# dump to csv
df.fillna(0, inplace=True)
df.to_csv(outfile, index=False)
print(f"done. saved to {outfile}")