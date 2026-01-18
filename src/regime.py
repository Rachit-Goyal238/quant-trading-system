import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# config
infile = "data/nifty_features_5min.csv"
outfile = "data/nifty_regimes_5min.csv"
modelfile = "models/hmm_model.pkl"
plotdir = "plots"

# make dirs
os.makedirs("models", exist_ok=True)
os.makedirs(plotdir, exist_ok=True)

print("starting regime detection...")

# load
print("loading data...")
df = pd.read_csv(infile)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# cleanup
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# select features
cols = [
    'average_iv', 'iv_spread', 'pcr_oi', 
    'delta', 'gamma', 'vega', 
    'futures_basis', 'spot_returns'
]
print(f"using cols: {cols}")
X = df[cols].values

# scale it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train hmm
print("training hmm...")
train_size = int(len(X_scaled) * 0.70)
X_train = X_scaled[:train_size]

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model.fit(X_train)
joblib.dump(model, modelfile)

# predict
hidden_states = model.predict(X_scaled)
df['hmm_state'] = hidden_states

# map states to labels
# logic: high return = up, low = down, mid = sideways
means = df.groupby('hmm_state')['spot_returns'].mean()
sorted_idx = means.sort_values().index.tolist()

down_s = sorted_idx[0]    # lowest ret
side_s = sorted_idx[1]    # mid
up_s = sorted_idx[2]      # highest

mapping = {up_s: 1, down_s: -1, side_s: 0}
df['regime'] = df['hmm_state'].map(mapping)

print(f"mapped: {up_s}->Up, {down_s}->Down, {side_s}->Sideways")

# plots
print("making plots...")

# 1. overlay
plt.figure(figsize=(15, 6))
sub = df.tail(1000)
x_rng = range(len(sub))
plt.plot(x_rng, sub['close_spot'], color='black', alpha=0.6, label='Price')

# color segments
for i in range(len(sub)):
    reg = sub.iloc[i]['regime']
    c = 'lightgray' if reg == 0 else ('green' if reg == 1 else 'red')
    plt.axvspan(i, i+1, color=c, alpha=0.2, lw=0)

plt.title("Regime Overlay")
plt.savefig(f"{plotdir}/regime_overlay.png")
print("saved overlay")

# 2. transition matrix
plt.figure(figsize=(8, 6))
trans = pd.DataFrame(model.transmat_, 
                     index=[f"State {i}" for i in range(3)],
                     columns=[f"State {i}" for i in range(3)])
sns.heatmap(trans, annot=True, cmap="Blues", fmt=".2f")
plt.title("Transition Matrix")
plt.savefig(f"{plotdir}/transition_matrix.png")
print("saved matrix")

# 3. stats boxplots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='regime', y='average_iv', data=df)
plt.title("IV by Regime")

plt.subplot(1, 2, 2)
sns.boxplot(x='regime', y='vega', data=df)
plt.title("Vega by Regime")
plt.savefig(f"{plotdir}/regime_statistics.png")
print("saved boxplots")

# 4. duration
df['grp'] = (df['regime'] != df['regime'].shift()).cumsum()
durs = df.groupby(['grp', 'regime']).size().reset_index(name='duration')

plt.figure(figsize=(10, 6))
sns.histplot(data=durs, x='duration', hue='regime', bins=30, kde=True, palette={1:'green', -1:'red', 0:'gray'})
plt.title("Regime Duration")
plt.savefig(f"{plotdir}/duration_histogram.png")
print("saved hist")

# save
df.to_csv(outfile)
print(f"done. saved to {outfile}")