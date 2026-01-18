Quantitative Trading System: Regime-Filtered EMA Strategy

 📌 Project Overview
This project implements an end-to-end quantitative trading system for NIFTY 50 derivatives. It features a complete data engineering pipeline, a Hidden Markov Model (HMM) for market regime detection, and a machine learning-enhanced EMA crossover strategy.

Key Objectives:
1.  Data Engineering: Processed 1 year of 5-minute interval data for NIFTY Spot, Futures, and Options.
2.  Regime Detection: Classified market states (Uptrend, Downtrend, Sideways) using Gaussian HMM on Option Greeks.
3.  Strategy: Executed a 5/15 EMA Crossover strategy filtered by the detected regime.
4.  ML Enhancement: Optimized win rates using XGBoost and Neural Networks (LSTM/DNN).
5.  Outlier Analysis: Analyzed high-performance "Jackpot" trades (>3 Sigma).

---

 📂 Repository Structure
```bash
├── data/                  # Processed Data (Spot, Fut, Opt, Regimes)
├── models/                # Saved Models (HMM, XGBoost)
├── notebooks/             # Jupyter Notebooks for each stage
│   ├── 1_Data_Acquisition.ipynb
│   ├── 2_Data_Cleaning.ipynb
│   ├── 3_Feature_Engineering.ipynb
│   ├── 4_Regime_Detection.ipynb
│   ├── 5_Baseline_Strategy.ipynb
│   ├── 6_ML_Models.ipynb
│   └── 7_Outlier_Analysis.ipynb
├── plots/                 # Visualizations (Regime Charts, Equity Curves)
├── results/               # Trade Logs, Metrics Reports
├── src/                   # Python Modules
│   ├── data_utils.py      # Cleaning & Merging
│   ├── features.py        # Greeks & Technical Indicators
│   ├── regime.py          # HMM Implementation
│   ├── strategy.py        # Signal Generation
│   └── ml_models.py       # XGBoost & Deep Learning
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation

How to Run
Install Dependencies:

Bash

pip install -r requirements.txt

Run the Pipeline: You can run the Jupyter notebooks in order (1 to 7) or execute the python modules in src/.

Bash

python src/data_utils.py
python src/features.py
python src/regime.py
python src/strategy.py
python src/ml_models.py

Key Results Summary
Regime Detection: Successfully identified volatility regimes using Average IV and Vega.

Baseline Strategy: Generated ~20,000 points profit with a 26% win rate (Trend Following profile).

ML Enhancement: * XGBoost: Improved win rate to ~31% by filtering false signals.

Neural Network: Achieved 100% win rate by selecting only highest-confidence trades.

Outlier Analysis: Trades with Z-Score > 3 generated 10.3x more profit than average trades, typically occurring during high-momentum regimes with expanding EMA gaps.