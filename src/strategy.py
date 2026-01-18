import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "data/nifty_regimes_5min.csv"
OUTPUT_TRADES = "results/trade_log.csv"
OUTPUT_METRICS = "results/performance_metrics.txt"
EQUITY_CURVE_PLOT = "plots/equity_curve.png"

# Ensure results folder exists
import os
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("--- STARTING STRATEGY BACKTEST (Task 4.1 & 4.2) ---")

# 1. LOAD DATA
print("1. Loading Data...")
df = pd.read_csv(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# 2. IMPLEMENT STRATEGY SIGNALS (Task 4.1)
print("2. Generating Signals...")

# Define Crossovers
# Cross Over: 5 EMA was below 15, now is above
df['crossover_up'] = (df['ema_5'] > df['ema_15']) & (df['ema_5'].shift(1) <= df['ema_15'].shift(1))
# Cross Under: 5 EMA was above 15, now is below
df['crossover_down'] = (df['ema_5'] < df['ema_15']) & (df['ema_5'].shift(1) >= df['ema_15'].shift(1))

# Initialize Position Column (1=Long, -1=Short, 0=Flat)
df['position'] = 0

# Iterative Backtest (To handle "Enter at Next Open" logic correctly)
# We loop through to manage state
current_pos = 0 # 0=Flat, 1=Long, -1=Short
positions = []

# Convert to numpy for speed
regimes = df['regime'].values
cross_up = df['crossover_up'].values
cross_down = df['crossover_down'].values
opens = df['open'].values # We enter at OPEN of next candle
closes = df['close_spot'].values
timestamps = df.index

trade_log = []

for i in range(len(df) - 1):
    # Logic applies to determining position for i+1
    
    # 1. CHECK EXIT CONDITIONS FIRST
    if current_pos == 1: # We are Long
        # Exit if 5 crosses below 15
        if cross_down[i]:
            current_pos = 0
            # Record Trade Exit
            trade_log[-1]['exit_time'] = timestamps[i+1]
            trade_log[-1]['exit_price'] = opens[i+1]
            
    elif current_pos == -1: # We are Short
        # Exit if 5 crosses above 15
        if cross_up[i]:
            current_pos = 0
            # Record Trade Exit
            trade_log[-1]['exit_time'] = timestamps[i+1]
            trade_log[-1]['exit_price'] = opens[i+1]

    # 2. CHECK ENTRY CONDITIONS (Only if Flat)
    if current_pos == 0:
        # Long Entry: Cross Up AND Regime +1
        if cross_up[i] and regimes[i] == 1:
            current_pos = 1
            trade_log.append({
                'entry_time': timestamps[i+1],
                'type': 'LONG',
                'entry_price': opens[i+1],
                'exit_time': None,
                'exit_price': None
            })
            
        # Short Entry: Cross Down AND Regime -1
        elif cross_down[i] and regimes[i] == -1:
            current_pos = -1
            trade_log.append({
                'entry_time': timestamps[i+1],
                'type': 'SHORT',
                'entry_price': opens[i+1],
                'exit_time': None,
                'exit_price': None
            })

    positions.append(current_pos)

# Add last position
positions.append(current_pos)
df['position'] = positions

# Close any open trade at the end
if trade_log and trade_log[-1]['exit_price'] is None:
    trade_log[-1]['exit_time'] = timestamps[-1]
    trade_log[-1]['exit_price'] = closes[-1]

# Create Trade DataFrame
trades_df = pd.DataFrame(trade_log)

# Calculate PnL per trade
# Long: Exit - Entry
# Short: Entry - Exit
trades_df['pnl_points'] = np.where(
    trades_df['type'] == 'LONG',
    trades_df['exit_price'] - trades_df['entry_price'],
    trades_df['entry_price'] - trades_df['exit_price']
)

trades_df['pnl_pct'] = trades_df['pnl_points'] / trades_df['entry_price']

# Save Trade Log
trades_df.to_csv(OUTPUT_TRADES, index=False)
print(f"   Generated {len(trades_df)} trades. Saved to {OUTPUT_TRADES}")

# 3. SPLIT TRAIN/TEST (Task 4.2)
# "Training: First 70%, Testing: Last 30%"
split_idx = int(len(trades_df) * 0.70)
train_trades = trades_df.iloc[:split_idx]
test_trades = trades_df.iloc[split_idx:]

print(f"   Split: {len(train_trades)} Train Trades, {len(test_trades)} Test Trades")

# 4. CALCULATE METRICS
def calculate_metrics(trade_data, label="Dataset"):
    if len(trade_data) == 0: return f"No trades in {label}"
    
    total_trades = len(trade_data)
    win_rate = len(trade_data[trade_data['pnl_points'] > 0]) / total_trades
    
    # Cumulative Return (Points)
    total_return_points = trade_data['pnl_points'].sum()
    
    # Drawdown Calculation
    cum_pnl = trade_data['pnl_points'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()
    
    # Sharpe (Simplified annualized using trade returns)
    avg_return = trade_data['pnl_pct'].mean()
    std_return = trade_data['pnl_pct'].std()
    sharpe = (avg_return / std_return) * np.sqrt(252 * 75) if std_return != 0 else 0 # Approx scaling
    
    # Sortino (Downside Deviation)
    neg_returns = trade_data[trade_data['pnl_pct'] < 0]['pnl_pct']
    downside_std = neg_returns.std()
    sortino = (avg_return / downside_std) * np.sqrt(252 * 75) if downside_std != 0 else 0
    
    # Calmar (Return / Max DD) - Using Points for approximation
    calmar = abs(total_return_points / max_drawdown) if max_drawdown != 0 else 0
    
    profit_factor = abs(trade_data[trade_data['pnl_points'] > 0]['pnl_points'].sum() / 
                        trade_data[trade_data['pnl_points'] < 0]['pnl_points'].sum())

    report = f"""
--- {label} PERFORMANCE ---
Total Trades: {total_trades}
Total Return (Points): {total_return_points:.2f}
Win Rate: {win_rate*100:.2f}%
Profit Factor: {profit_factor:.2f}
Max Drawdown (Points): {max_drawdown:.2f}
Sharpe Ratio: {sharpe:.2f}
Sortino Ratio: {sortino:.2f}
Calmar Ratio: {calmar:.2f}
"""
    return report

report_train = calculate_metrics(train_trades, "TRAINING (70%)")
report_test = calculate_metrics(test_trades, "TESTING (30%)")
report_full = calculate_metrics(trades_df, "FULL DATASET")

print(report_train)
print(report_test)

# Save Metrics
with open(OUTPUT_METRICS, "w") as f:
    f.write(report_train + "\n" + report_test + "\n" + report_full)

# 5. PLOT EQUITY CURVE
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(trades_df['entry_time']), trades_df['pnl_points'].cumsum())
plt.title("Strategy Equity Curve (PnL Points)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.grid(True)
plt.axvline(x=pd.to_datetime(test_trades.iloc[0]['entry_time']), color='r', linestyle='--', label='Train/Test Split')
plt.legend()
plt.savefig(EQUITY_CURVE_PLOT)

print(f"✔ SUCCESS! Backtest Complete. Metrics saved to {OUTPUT_METRICS}")