import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import sys

# fix encoding just in case
sys.stdout.reconfigure(encoding='utf-8')

# config
feats = "data/nifty_regimes_5min.csv"
t_log = "results/trade_log.csv"
out_file = "results/ml_comparison.txt"

print("starting ml run...")

# load data
print("loading...")
if not os.path.exists(feats) or not os.path.exists(t_log):
    print("missing files, exiting")
    sys.exit()

# market data
features_df = pd.read_csv(feats)
features_df['datetime'] = pd.to_datetime(features_df['datetime'])
features_df = features_df.sort_values('datetime')
features_df = features_df.drop_duplicates(subset=['datetime'], keep='first')
print(f"market data: {len(features_df)} rows")

# trade logs
trades = pd.read_csv(t_log)
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['target'] = (trades['pnl_points'] > 0).astype(int)
trades = trades.sort_values('entry_time')
print(f"trades: {len(trades)}")

# merge
print("merging data...")

# matching trades to previous candle
merged_df = pd.merge_asof(
    trades,
    features_df,
    left_on='entry_time',
    right_on='datetime',
    direction='backward',
    tolerance=pd.Timedelta("10min") 
)

# drop bad matches
valid_trades = merged_df.dropna(subset=['close_spot', 'average_iv'])
print(f"matched {len(valid_trades)} trades")

if len(valid_trades) < 50:
    print("not enough data matched. check timezones.")
    sys.exit()

# prep vectors
cols = ['close_spot', 'ema_5', 'ema_15', 'average_iv', 'pcr_oi', 'delta', 'gamma', 'regime']
X_static = valid_trades[cols].values
y = valid_trades['target'].values

# train/test split
split = int(len(y) * 0.70)
X_train, X_test = X_static[:split], X_static[split:]
y_train, y_test = y[:split], y[split:]
test_trades = valid_trades.iloc[split:].copy()

# xgboost
print("training xgb...")
model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
model_xgb.fit(X_train, y_train)

# predict
probs = model_xgb.predict_proba(X_test)[:, 1]
test_trades['xgb_signal'] = (probs > 0.55).astype(int)
print("xgb done")

# lstm / nn
print("training nn...")
lstm_probs = np.zeros(len(y_test))

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    
    # simple nn as lstm proxy
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Sequential()
    model.add(Dense(64, input_dim=len(cols), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_s, y_train, epochs=10, batch_size=16, verbose=0)
    lstm_probs = model.predict(X_test_s, verbose=0).flatten()
    print("nn done")
    
except Exception as e:
    print(f"skipping nn: {e}")

test_trades['lstm_signal'] = (lstm_probs > 0.55).astype(int)

# results
print("generating report...")
def get_stats(df, name):
    if len(df) == 0: return [name, 0, 0, 0]
    return [
        name, 
        round(df['pnl_points'].sum(), 2), 
        round((len(df[df['pnl_points'] > 0]) / len(df)) * 100, 2), 
        round(df['pnl_points'].mean(), 2)
    ]

res_base = get_stats(test_trades, "Baseline")
res_xgb = get_stats(test_trades[test_trades['xgb_signal']==1], "XGBoost")
res_lstm = get_stats(test_trades[test_trades['lstm_signal']==1], "LSTM")

results_df = pd.DataFrame([res_base, res_xgb, res_lstm], columns=["Strategy", "Total PnL", "Win Rate %", "Avg PnL"])
print("\n" + str(results_df))

with open(out_file, "w") as f:
    f.write(results_df.to_string())

print(f"saved to {out_file}")